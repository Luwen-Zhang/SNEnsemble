import unittest
from import_utils import *
import src
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
from src.model._transformer.gp.base import (
    get_test_case_1d,
    get_test_case_2d,
    get_cls_test_case_1d,
    get_cls_test_case_2d,
)
from src.model._transformer.bayes_nn.bbp import (
    BayesByBackprop,
    MCDropout,
    _isotropic_gauss_log_likelihood,
)


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5 * (target - output) ** 2 / sigma**2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik * (varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma**2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))

        return (exponent + log_coeff).sum()


class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior

        scale = (2 / self.input_dim) ** 0.5
        rho_init = np.log(np.exp((2 / self.input_dim) ** 0.5) - 1)
        self.weight_mus = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01)
        )
        self.weight_rhos = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3)
        )

        self.bias_mus = nn.Parameter(
            torch.Tensor(self.output_dim).uniform_(-0.01, 0.01)
        )
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))

    def forward(self, x, sample=True):
        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = torch.randn_like(self.weight_mus)
            bias_epsilons = torch.randn_like(self.bias_mus)

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds
            bias_sample = self.bias_mus + bias_epsilons * bias_stds

            output = torch.mm(x, weight_sample) + bias_sample

            # computing the KL loss term
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = (
                0.5 * (torch.log(prior_cov / varpost_cov)).sum()
                - 0.5 * weight_stds.numel()
            )
            KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
            KL_loss = (
                KL_loss
                + 0.5 * ((self.weight_mus - self.prior.mu) ** 2 / prior_cov).sum()
            )

            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = (
                KL_loss
                + 0.5 * (torch.log(prior_cov / varpost_cov)).sum()
                - 0.5 * bias_stds.numel()
            )
            KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
            KL_loss = (
                KL_loss + 0.5 * ((self.bias_mus - self.prior.mu) ** 2 / prior_cov).sum()
            )

            return output, KL_loss

        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, None

    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(
                self.weight_mus.data.new(self.weight_mus.size()).normal_()
            )

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds

            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()

        return all_samples


class BBP_Heteroscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(num_units, 2 * output_dim, gaussian(0, 1))

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss

        return x, KL_loss_total


class BBP_Heteroscedastic_Model_Wrapper:
    def __init__(self, network, learn_rate, batch_size, no_batches):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.network = network

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.loss_func = log_gaussian_loss

    def fit(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=False)

        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0
        KL_loss_total = 0
        for i in range(no_samples):
            output, KL_loss = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss = (
                log_gaussian_loss(output[:, :1], y, output[:, 1:].exp(), 1) / x.shape[0]
            )
            fit_loss_total = fit_loss_total + fit_loss / no_samples
            KL_loss_total = (
                KL_loss_total + KL_loss / self.no_batches / x.shape[0] / no_samples
            )

        total_loss = fit_loss_total + KL_loss_total
        total_loss.backward()
        self.optimizer.step()

        return fit_loss_total, KL_loss_total


def bbp(X, y, grid):
    torch.manual_seed(0)
    start = time.time()
    num_epochs, batch_size, nb_train = 5, X.shape[0], X.shape[0]

    net = BBP_Heteroscedastic_Model_Wrapper(
        network=BBP_Heteroscedastic_Model(
            input_dim=X.shape[1], output_dim=1, num_units=10
        ),
        learn_rate=1e-2,
        batch_size=batch_size,
        no_batches=1,
    )

    fit_loss_train = np.zeros(num_epochs)
    KL_loss_train = np.zeros(num_epochs)
    total_loss = np.zeros(num_epochs)

    for i in range(num_epochs):
        fit_loss, KL_loss = net.fit(X, y, no_samples=1)
        fit_loss_train[i] += fit_loss.cpu().data.numpy()
        KL_loss_train[i] += KL_loss.cpu().data.numpy()

        total_loss[i] = fit_loss_train[i] + KL_loss_train[i]

        if i % 100 == 0 or i == num_epochs - 1:
            print(
                "Epoch: %5d/%5d, Fit loss = %7.3f, KL loss = %8.3f"
                % (i + 1, num_epochs, fit_loss_train[i], KL_loss_train[i])
            )
    train_end = time.time()
    net.network.eval()
    samples, noises = [], []
    for i in range(1):
        preds = net.network.forward(grid)[0]
        samples.append(preds[:, 0].cpu().data.numpy())
        noises.append(preds[:, 1].exp().cpu().data.numpy())

    samples = np.array(samples)
    noises = np.array(noises)
    means = samples.mean(axis=0)

    aleatoric = (noises**2).mean(axis=0)
    epistemic = samples.var(axis=0)

    total_unc = aleatoric + epistemic
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    mu1 = torch.tensor(means)
    var1 = torch.tensor(aleatoric + epistemic)

    torch.manual_seed(0)
    start = time.time()
    net2 = BayesByBackprop(
        n_inputs=X.shape[1], n_outputs=1, n_hidden=10, on_cpu=True, eps=0.0
    )
    net2.fit(X, y, n_epoch=5, n_samples=1, batch_size=None)
    train_end = time.time()
    mu2, var2 = net2.predict(grid, n_samples=1)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")

    assert torch.allclose(mu1, mu2), f"Means are not consistent"
    assert torch.allclose(var1, var2), f"Variances are not consistent"


def mc_dropout(X, y, grid, type, task):
    device = "cpu"
    torch.manual_seed(0)
    start = time.time()
    if task == "regression":
        n_outputs = 1 if len(y.shape) == 1 else y.shape[-1]
    else:
        n_outputs = torch.max(y).item() + 1
    net = MCDropout(
        n_inputs=1 if len(X.shape) == 1 else X.shape[-1],
        n_outputs=n_outputs,
        layers=[64, 32],
        on_cpu=False,
        type=type,
        task=task,
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=10)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    return net, mu, var


class TestBBP(unittest.TestCase):
    def test_bbp(self):
        X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3)
        bbp(X, y, grid)
        X, y, grid, _, _ = get_test_case_2d(10)
        bbp(X, y, grid)

    def test_mcdropout_1d_regression(self):
        X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3, noise=0.2)
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="regression"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="regression"
        )

        assert net_hete.type == "hete"
        assert net_hete.loss_func == _isotropic_gauss_log_likelihood
        assert isinstance(net_hete.output_act, nn.Identity)
        assert net_homo.type == "homo"
        assert net_hete.loss_func == _isotropic_gauss_log_likelihood
        assert isinstance(net_hete.output_act, nn.Identity)

        assert not torch.allclose(mu_hete, mu_homo)
        assert not torch.allclose(var_hete, var_homo)

    def test_mcdropout_2d_regression(self):
        X, y, grid, _, _ = get_test_case_2d(10)
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="regression"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="regression"
        )

        assert net_hete.type == "hete"
        assert net_hete.loss_func == _isotropic_gauss_log_likelihood
        assert isinstance(net_hete.output_act, nn.Identity)
        assert net_homo.type == "homo"
        assert net_hete.loss_func == _isotropic_gauss_log_likelihood
        assert isinstance(net_hete.output_act, nn.Identity)

        assert not torch.allclose(mu_hete, mu_homo)
        assert not torch.allclose(var_hete, var_homo)

    def test_mcdropout_1d_classification(self):
        print("Testing binary")
        X, y, grid = get_cls_test_case_1d(
            binary=True, n_samples=100, grid_low=-3, grid_high=3, noise=0.2
        )
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="classification"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="classification"
        )

        assert net_hete.type == "homo"
        assert isinstance(net_hete.loss_func, nn.BCELoss)
        assert isinstance(net_hete.output_act, nn.Sigmoid)
        assert net_hete.n_outputs == 1
        assert net_homo.type == "homo"
        assert isinstance(net_homo.loss_func, nn.BCELoss)
        assert isinstance(net_homo.output_act, nn.Sigmoid)
        assert net_homo.n_outputs == 1

        assert torch.allclose(mu_hete, mu_homo)
        assert torch.allclose(var_hete, var_homo)

        print("Testing multi-class")
        X, y, grid = get_cls_test_case_1d(
            binary=False, n_samples=100, grid_low=-3, grid_high=3, noise=0.2
        )
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="classification"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="classification"
        )

        assert net_hete.type == "homo"
        assert isinstance(net_hete.loss_func, nn.NLLLoss)
        assert isinstance(net_hete.output_act, nn.LogSoftmax)
        assert net_homo.type == "homo"
        assert isinstance(net_homo.loss_func, nn.NLLLoss)
        assert isinstance(net_homo.output_act, nn.LogSoftmax)

        assert torch.allclose(mu_hete, mu_homo)
        assert torch.allclose(var_hete, var_homo)

    def test_mcdropout_2d_classification(self):
        print("Testing binary")
        X, y, grid, _, _ = get_cls_test_case_2d(binary=True, n_samples=10)
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="classification"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="classification"
        )

        assert net_hete.type == "homo"
        assert isinstance(net_hete.loss_func, nn.BCELoss)
        assert isinstance(net_hete.output_act, nn.Sigmoid)
        assert net_homo.type == "homo"
        assert isinstance(net_homo.loss_func, nn.BCELoss)
        assert isinstance(net_homo.output_act, nn.Sigmoid)

        assert torch.allclose(mu_hete, mu_homo)
        assert torch.allclose(var_hete, var_homo)

        print("Testing multi-class")
        X, y, grid, _, _ = get_cls_test_case_2d(binary=False, n_samples=10)
        net_hete, mu_hete, var_hete = mc_dropout(
            X, y, grid, type="hete", task="classification"
        )
        net_homo, mu_homo, var_homo = mc_dropout(
            X, y, grid, type="homo", task="classification"
        )

        assert net_hete.type == "homo"
        assert isinstance(net_hete.loss_func, nn.NLLLoss)
        assert isinstance(net_hete.output_act, nn.LogSoftmax)
        assert net_homo.type == "homo"
        assert isinstance(net_homo.loss_func, nn.NLLLoss)
        assert isinstance(net_homo.output_act, nn.LogSoftmax)

        assert torch.allclose(mu_hete, mu_homo)
        assert torch.allclose(var_hete, var_homo)
