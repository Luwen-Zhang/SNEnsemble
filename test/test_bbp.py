import unittest
from import_utils import *
import src
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
from src.model._transformer.gp.base import get_test_case_1d, get_test_case_2d
from src.model._transformer.bayes_nn.bbp import HeteroscedasticBBP


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

        for i in range(no_samples):
            output, KL_loss_total = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1)
            fit_loss_total = fit_loss_total + fit_loss

        KL_loss_total = KL_loss_total / self.no_batches
        total_loss = (fit_loss_total + KL_loss_total) / (no_samples * x.shape[0])
        total_loss.backward()
        self.optimizer.step()

        return fit_loss_total / no_samples, KL_loss_total

    def get_loss_and_rmse(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=False)

        means, stds = [], []
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)
            means.append(output[:, :1, None])
            stds.append(output[:, 1:, None].exp())

        means, stds = torch.cat(means, 2), torch.cat(stds, 2)
        mean = means.mean(dim=2)
        std = (means.var(dim=2) + stds.mean(dim=2) ** 2) ** 0.5

        # calculate fit loss based on mean and standard deviation of output
        logliks = self.loss_func(
            output[:, :1], y, output[:, 1:].exp(), 1, sum_reduce=False
        )
        rmse = float((((mean - y) ** 2).mean() ** 0.5).cpu().data)

        return logliks, rmse


class TestBBP(unittest.TestCase):
    def test_bbp(self):
        X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3)
        torch.manual_seed(0)
        start = time.time()
        num_epochs, batch_size, nb_train = 5, X.shape[0], X.shape[0]

        net = BBP_Heteroscedastic_Model_Wrapper(
            network=BBP_Heteroscedastic_Model(input_dim=1, output_dim=1, num_units=10),
            learn_rate=1e-2,
            batch_size=batch_size,
            no_batches=1,
        )

        fit_loss_train = np.zeros(num_epochs)
        KL_loss_train = np.zeros(num_epochs)
        total_loss = np.zeros(num_epochs)

        for i in range(num_epochs):
            fit_loss, KL_loss = net.fit(X, y, no_samples=10)
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
        for i in range(5):
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
        net2 = HeteroscedasticBBP(n_inputs=1, n_outputs=1, n_hidden=10, on_cpu=True)
        net2.fit(X, y, n_epoch=5, n_samples=10, batch_size=100)
        train_end = time.time()
        mu2, var2 = net2.predict(grid, n_samples=5)
        print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")

        assert torch.allclose(mu1, mu2), f"Means are not consistent"
        assert torch.allclose(var1, var2), f"Variances are not consistent"
