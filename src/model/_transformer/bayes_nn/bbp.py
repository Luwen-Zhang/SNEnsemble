import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch


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


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma**2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))
        return (exponent + log_coeff).sum()


class BayesLinearLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, prior):
        super(BayesLinearLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.prior = prior

        scale = (2 / self.n_inputs) ** 0.5
        rho_init = np.log(np.exp((2 / self.n_inputs) ** 0.5) - 1)
        self.weight_mus = nn.Parameter(
            torch.Tensor(self.n_inputs, self.n_outputs).uniform_(-0.01, 0.01)
        )
        self.weight_rhos = nn.Parameter(
            torch.Tensor(self.n_inputs, self.n_outputs).uniform_(-3, -3)
        )

        self.bias_mus = nn.Parameter(torch.Tensor(self.n_outputs).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.n_outputs).uniform_(-4, -3))

    def forward(self, x, sample=True):
        if sample:
            # sample Gaussian noise for each weight and each bias
            weight_epsilons = torch.randn_like(self.weight_mus, device=x.device)
            bias_epsilons = torch.randn_like(self.bias_mus, device=x.device)

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
            # sample Gaussian noise for each weight and each bias
            weight_epsilons = Variable(
                self.weight_mus.data.new(self.weight_mus.size()).normal_()
            )

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds

            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()

        return all_samples


class HeteroscedasticBBP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, on_cpu=False, **kwargs):
        super(HeteroscedasticBBP, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # network with two hidden and one output layer
        self.layer1 = BayesLinearLayer(n_inputs, n_hidden, Gaussian(0, 1))
        self.layer2 = BayesLinearLayer(n_hidden, 2 * n_outputs, Gaussian(0, 1))

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

        self.optimizer = None
        self.kwargs = kwargs
        self.on_cpu = on_cpu

    def to_cpu(self, x: torch.Tensor):
        if self.on_cpu:
            self.to("cpu")
            return x.device, x.to("cpu")
        else:
            return x.device, x

    def to_device(self, x: torch.Tensor, device):
        if self.on_cpu:
            return x.to(device)
        else:
            return x

    def forward(self, x):
        device, x = self.to_cpu(x)
        KL_loss_total = 0
        x = x.view(-1, self.n_inputs)

        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.to_device(x, device)
        KL_loss_total = self.to_device(KL_loss_total, device)
        return x, KL_loss_total

    def _get_optimizer(self, lr=0.01, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def fit(self, X, y, batch_size=None, n_epoch=100, n_samples=10):
        if self.optimizer is None:
            self.optimizer = self._get_optimizer(**self.kwargs)
        if batch_size is None:
            batch_size = X.shape[0]
        # loader = Data.DataLoader(
        #     Data.TensorDataset(X, y), shuffle=True, batch_size=batch_size
        # )
        # n_batches = len(loader)
        for epoch in range(n_epoch):
            # for x, y_hat in loader:
            self.optimizer.zero_grad()
            fit_loss_total = 0
            kl_loss = 0
            for i in range(n_samples):
                output, kl_loss = self(X)
                # calculate fit loss based on mean and standard deviation of output
                fit_loss = log_gaussian_loss(output[:, :1], y, output[:, 1:].exp(), 1)
                fit_loss_total = fit_loss_total + fit_loss

            total_loss = (fit_loss_total + kl_loss / 1) / (n_samples * X.shape[0])
            total_loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0 or epoch == n_epoch - 1:
                print(
                    "Epoch: %5d/%5d, Fit loss = %7.3f, KL loss = %8.3f"
                    % (
                        epoch + 1,
                        n_epoch,
                        fit_loss_total / n_samples,
                        kl_loss,
                    )
                )

    def predict(self, X, n_samples=100):
        samples, noises = [], []
        for i in range(n_samples):
            preds = self(X)[0]
            samples.append(preds[:, :1])
            noises.append(preds[:, 1:])
        mean = torch.mean(torch.concat(samples, dim=-1), dim=-1)
        epistemic_var = torch.var(torch.concat(samples, dim=-1), dim=-1, unbiased=False)
        stds = torch.exp(torch.concat(noises, dim=-1))
        aleatoric_var = torch.mean(stds**2, dim=-1)
        var = epistemic_var + aleatoric_var
        return mean, var


if __name__ == "__main__":
    from src.model._transformer.gp.base import (
        get_test_case_1d,
        plot_mu_var_1d,
        get_test_case_2d,
        plot_mu_var_2d,
    )

    device = "cuda"
    X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3)
    torch.manual_seed(0)
    start = time.time()
    net = HeteroscedasticBBP(n_inputs=1, n_outputs=1, n_hidden=400, on_cpu=False)
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=2000, n_samples=10, batch_size=100)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=100)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(
        100, grid_low_x=-3, grid_high_x=3, grid_low_y=-3, grid_high_y=3
    )
    torch.manual_seed(0)
    start = time.time()
    net = HeteroscedasticBBP(n_inputs=2, n_outputs=1, n_hidden=400, on_cpu=False)
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=2000, n_samples=10, batch_size=100)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=100)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)
