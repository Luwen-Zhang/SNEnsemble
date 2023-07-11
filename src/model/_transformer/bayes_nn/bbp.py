"""
This is a modification of https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/regression/bbp_hetero.ipynb
improved by comparing https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop/model.py (for stability and classification tasks)
and further improved by https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py (for local reparameterization)
"""

import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import warnings


def _isotropic_gauss_log_likelihood(x, mu, sigma, eps=1e-8):
    # https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
    # https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/1f867a5bcbd1abfecede99807eb0b5f97ed8be7c/src/priors.py#L6
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu).float()
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma).float()
    # The second term is the determinant of the covariance matrix (with only diagonal terms for the isotropic case)
    return (
        -0.5 * np.log(2 * np.pi)
        - torch.log(sigma)
        - 0.5 * _safe_pow(x - mu, 2) / _safe_positive(_safe_pow(sigma, 2), eps=eps)
    )


def _safe_pow(x, exp):
    return torch.nan_to_num(
        torch.pow(torch.nan_to_num(x, posinf=1e8, neginf=-1e8), exp),
        posinf=1e8,
        neginf=-1e8,
    )


def _safe_exp(x):
    return torch.exp(torch.clamp(x, max=50))


def _safe_positive(x, eps=1e-8):
    return torch.clamp(x, min=eps)


def _gauss_kl(mu_1: torch.Tensor, sigma_1: torch.Tensor, mu_2, sigma_2, eps=1e-8):
    # https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/torchbnn/functional.py
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py#L25
    # For isotropic gaussian prior, the function reduces to a simple case: https://arxiv.org/abs/1312.6114 Appendix B
    if not isinstance(mu_2, torch.Tensor):
        mu_2 = torch.tensor(mu_2).float()
    if not isinstance(sigma_2, torch.Tensor):
        sigma_2 = torch.tensor(sigma_2).float()
    kl = (
        torch.log(sigma_2)
        - torch.log(sigma_1)
        + (_safe_pow(sigma_1, 2) + _safe_pow(mu_1 - mu_2, 2))
        / (2 * _safe_positive(_safe_pow(sigma_2, 2), eps=eps))
        - 0.5
    )
    return kl


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        return _isotropic_gauss_log_likelihood(weights, self.mu, self.sigma).sum()


class BayesLinearLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, prior, eps=1e-8, local_reparam=False):
        super(BayesLinearLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.prior = prior
        self.local_reparam = local_reparam
        self.weight_mus = nn.Parameter(
            torch.Tensor(self.n_inputs, self.n_outputs).uniform_(-0.01, 0.01)
        )
        self.weight_rhos = nn.Parameter(
            torch.Tensor(self.n_inputs, self.n_outputs).uniform_(-3, -3)
        )

        self.bias_mus = nn.Parameter(torch.Tensor(self.n_outputs).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.n_outputs).uniform_(-4, -3))

        self.eps = eps

    def forward(self, x, sample=True):
        if sample:
            # calculate the weight and bias stds from the rho parameters
            weight_stds = _safe_positive(
                F.softplus(self.weight_rhos, beta=1, threshold=20), eps=self.eps
            )
            bias_stds = _safe_positive(
                F.softplus(self.bias_rhos, beta=1, threshold=20), eps=self.eps
            )
            if self.local_reparam:
                act_weight_mus = torch.mm(x, self.weight_mus)
                act_weight_stds = torch.sqrt(
                    _safe_positive(
                        torch.mm(_safe_pow(x, 2), _safe_pow(weight_stds, 2)),
                        eps=self.eps,
                    )
                )
                weight_epsilons = torch.randn_like(act_weight_mus, device=x.device)
                out_weight_sample = act_weight_mus + act_weight_stds * weight_epsilons
            else:
                # sample Gaussian noise for each weight and each bias
                weight_epsilons = torch.randn_like(self.weight_mus, device=x.device)
                # calculate samples from the posterior from the sampled noise and mus/stds
                weight_sample = self.weight_mus + weight_epsilons * weight_stds
                out_weight_sample = torch.mm(x, weight_sample)
            bias_epsilons = torch.randn_like(self.bias_mus, device=x.device)
            bias_sample = self.bias_mus + bias_epsilons * bias_stds
            output = out_weight_sample + bias_sample

            # computing the KL loss term
            weight_kl = _gauss_kl(
                self.weight_mus,
                weight_stds,
                self.prior.mu,
                self.prior.sigma,
                eps=self.eps,
            )
            bias_kl = _gauss_kl(
                self.bias_mus, bias_stds, self.prior.mu, self.prior.sigma, eps=self.eps
            )
            """
            The minimization of KL divergence is equivalent to the maximization of likelihood, see
            https://jaketae.github.io/study/kl-mle/, so we can see https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop/model.py#L69-L71
            the loss function is the following one:
            weight_kl = _isotropic_gauss_log_likelihood(
                weight_sample, self.weight_mus, weight_stds
            ) - _isotropic_gauss_log_likelihood(weight_sample, 0, 1)
            bias_kl = _isotropic_gauss_log_likelihood(
                bias_sample, self.bias_mus, bias_stds
            ) - _isotropic_gauss_log_likelihood(bias_sample, 0, 1)
            """
            return output, weight_kl.sum() + bias_kl.sum()
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, None


class AbstractBNN(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        on_cpu=False,
        task="regression",
        type="hete",
        eps=1e-6,
        init_log_noise=0.0,
        ignore_aleatoric=False,
        **kwargs,
    ):
        super(AbstractBNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.on_cpu = on_cpu
        self.task = task
        self.type = type
        self.eps = eps
        self.optimizer = None
        self.ignore_aleatoric = ignore_aleatoric
        self.kwargs = kwargs
        self.loss_func = None
        if ignore_aleatoric:
            if self.type == "hete":
                warnings.warn(f"ignore_aleatoric=True, so Homoscedastic model is used.")
                self.type = "homo"
            if self.task == "regression":
                self.loss_func = nn.MSELoss()
            elif self.task == "classification":
                self.loss_func = nn.CrossEntropyLoss()
        else:
            if self.task == "regression":
                self.loss_func = _isotropic_gauss_log_likelihood
                if self.type == "homo":
                    self.log_noise = nn.Parameter(torch.tensor([init_log_noise]))
            elif self.task == "classification":
                warnings.warn(
                    f"For classification tasks, the aleatoric variance will be ignored."
                )
                self.type = "homo"
                self.ignore_aleatoric = True
                self.loss_func = nn.CrossEntropyLoss()

        if self.loss_func is None:
            raise Exception(f"Type {self.type} for {task} tasks is not implemented. ")

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

    def _get_optimizer(self, lr=0.01, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def fit(self, X, y, batch_size=None, n_epoch=100, n_samples=10):
        if self.optimizer is None:
            self.optimizer = self._get_optimizer(**self.kwargs)
        if batch_size is None:
            loader = [(X, y)]
            n_batches = 1
        else:
            loader = Data.DataLoader(
                Data.TensorDataset(X, y), shuffle=True, batch_size=batch_size
            )
            n_batches = len(loader)
        for epoch in range(n_epoch):
            self.on_train_epoch_start(X, y)
            for x, y_hat in loader:
                self.optimizer.zero_grad()
                total_loss = self._train_step(
                    x, y_hat, n_samples=n_samples, n_batches=n_batches
                )
                total_loss.backward()
                self.optimizer.step()
            self.on_train_epoch_end(X, y, epoch, n_epoch)

    def predict(self, X, n_samples=100):
        samples, noises = [], []
        for i in range(n_samples):
            preds = self._predict_step(X)
            if self.type == "hete":
                samples.append(preds[:, :1])
                noises.append(preds[:, 1:])
            else:
                samples.append(preds)
        mean = torch.mean(torch.concat(samples, dim=-1), dim=-1)
        epistemic_var = torch.var(torch.concat(samples, dim=-1), dim=-1, unbiased=False)
        if self.type == "hete":
            stds = torch.exp(torch.concat(noises, dim=-1))
            aleatoric_var = torch.mean(_safe_pow(stds, 2), dim=-1)
        else:
            if not self.ignore_aleatoric:
                stds = _safe_exp(self.log_noise)
                aleatoric_var = torch.ones_like(epistemic_var, device=X.device) * stds
            else:
                aleatoric_var = torch.zeros_like(epistemic_var, device=X.device)
        var = epistemic_var + aleatoric_var
        return mean, var

    def get_sample_kl_loss(self, kl_loss, batch_size, n_batches, n_samples):
        return kl_loss / batch_size / n_batches / n_samples

    def get_sample_fitness_loss(self, output, y_hat, n_samples):
        if self.loss_func == _isotropic_gauss_log_likelihood:
            if self.type == "hete":
                fit_loss = -self.loss_func(
                    output[:, :1], y_hat, _safe_exp(output[:, 1:]), eps=self.eps
                ).mean()
            else:
                fit_loss = -self.loss_func(
                    output, y_hat, _safe_exp(self.log_noise), eps=self.eps
                ).mean()
        else:
            fit_loss = self.loss_func(output, y_hat)
        return fit_loss / n_samples

    def _train_step(self, x, y, n_samples, n_batches, **kwargs):
        raise NotImplementedError

    def _predict_step(self, X):
        raise NotImplementedError

    def on_train_epoch_start(self, X, y):
        pass

    def on_train_epoch_end(self, X, y, i_epoch, n_epoch):
        pass


class BayesByBackprop(AbstractBNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers=2,
        on_cpu=False,
        task="regression",
        type="hete",
        ignore_aleatoric=False,
        eps=1e-6,
        kl_weight=1.0,
        local_reparam=False,
        **kwargs,
    ):
        super(BayesByBackprop, self).__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            on_cpu=on_cpu,
            task=task,
            type=type,
            eps=eps,
            ignore_aleatoric=ignore_aleatoric,
            **kwargs,
        )
        self.activation = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            if i == 0:
                n_in = n_inputs
                n_out = n_hidden
            elif i == n_layers - 1:
                n_in = n_hidden
                n_out = 2 * n_outputs if self.type == "hete" else n_outputs
            else:
                n_in = n_out = n_hidden

            self.layers.append(
                BayesLinearLayer(
                    n_in, n_out, Gaussian(0, 1), eps=eps, local_reparam=local_reparam
                )
            )
        self.kl_weight = kl_weight
        self.local_reparam = local_reparam

    def forward(self, x):
        device, x = self.to_cpu(x)
        kl_loss_total = 0
        x = x.view(-1, self.n_inputs)

        for idx, layer in enumerate(self.layers):
            x, kl_loss = layer(x)
            kl_loss_total = kl_loss_total + kl_loss
            if idx != len(self.layers) - 1:
                x = self.activation(x)

        x = self.to_device(x, device)
        kl_loss_total = self.to_device(kl_loss_total, device)
        return x, kl_loss_total

    def get_sample_loss(self, output, y_hat, kl_loss, batch_size, n_batches, n_samples):
        return self.get_sample_fitness_loss(
            output, y_hat, n_samples
        ), self.get_sample_kl_loss(kl_loss, batch_size, n_batches, n_samples)

    def _train_step(self, x, y, n_samples, n_batches, **kwargs):
        fit_loss_step = 0
        kl_loss_step = 0
        for i in range(n_samples):
            output, kl_loss = self(x)
            fit_loss_sample, kl_loss_sample = self.get_sample_loss(
                output,
                y,
                kl_loss,
                batch_size=x.shape[0],
                n_batches=n_batches,
                n_samples=n_samples,
            )
            fit_loss_step = fit_loss_step + fit_loss_sample
            kl_loss_step = kl_loss_step + kl_loss_sample
        total_loss = fit_loss_step + self.kl_weight * kl_loss_step
        self.fit_loss_epoch += fit_loss_step.item() * x.shape[0]
        self.kl_loss_epoch += kl_loss_step.item() * x.shape[0]
        return total_loss

    def _predict_step(self, X):
        return self(X)[0]

    def on_train_epoch_start(self, X, y):
        self.fit_loss_epoch = 0
        self.kl_loss_epoch = 0

    def on_train_epoch_end(self, X, y, i_epoch, n_epoch):
        self.fit_loss_epoch /= X.shape[0]
        self.kl_loss_epoch /= X.shape[0]
        if i_epoch % 100 == 0 or i_epoch == n_epoch - 1:
            print(
                "Epoch: %5d/%5d, Fit loss = %7.3f, KL loss = %8.3f"
                % (
                    i_epoch + 1,
                    n_epoch,
                    self.fit_loss_epoch,
                    self.kl_loss_epoch,
                )
            )


class MCDropout(AbstractBNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        on_cpu=False,
        task="regression",
        type="hete",
        eps=1e-6,
        **kwargs,
    ):
        super(MCDropout, self).__init__(
            n_inputs, n_outputs, on_cpu, task, type, eps, **kwargs
        )
        from src.model.base import get_sequential

        # network with two hidden and one output layer
        self.layers = get_sequential(
            layers=layers,
            n_inputs=n_inputs,
            n_outputs=2 * n_outputs if self.type == "hete" else n_outputs,
            use_norm=False,
            dropout=0.2,
            act_func=nn.ReLU,
            dropout_keep_training=True,
        )

    def forward(self, x):
        device, x = self.to_cpu(x)
        x = x.view(-1, self.n_inputs)
        x = self.layers(x)
        x = self.to_device(x, device)
        return x

    def _train_step(self, x, y, n_samples, n_batches, **kwargs):
        output = self(x)
        fit_loss_step = self.get_sample_fitness_loss(output, y, n_samples=1)
        self.fit_loss_epoch += fit_loss_step * x.shape[0]
        return fit_loss_step

    def _predict_step(self, X):
        return self(X)

    def on_train_epoch_start(self, X, y):
        self.fit_loss_epoch = 0

    def on_train_epoch_end(self, X, y, i_epoch, n_epoch):
        self.fit_loss_epoch /= X.shape[0]
        if i_epoch % 100 == 0 or i_epoch == n_epoch - 1:
            print(
                "Epoch: %5d/%5d, Fit loss = %7.3f"
                % (
                    i_epoch + 1,
                    n_epoch,
                    self.fit_loss_epoch,
                )
            )


if __name__ == "__main__":
    from src.model._transformer.gp.base import (
        get_test_case_1d,
        plot_mu_var_1d,
        get_test_case_2d,
        plot_mu_var_2d,
    )

    device = "cuda"
    X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3, noise=0.2)
    torch.manual_seed(0)
    start = time.time()
    net = BayesByBackprop(
        n_inputs=1, n_outputs=1, n_layers=2, n_hidden=20, on_cpu=False, type="homo"
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=2000, n_samples=10, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=100)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    # X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(
    #     100, grid_low_x=-3, grid_high_x=3, grid_low_y=-3, grid_high_y=3, noise=0.2
    # )
    # torch.manual_seed(0)
    # start = time.time()
    # net = BayesByBackprop(
    #     n_inputs=2, n_outputs=1, n_layers=4, n_hidden=50, on_cpu=False, type="homo"
    # )
    # X = X.to(device)
    # y = y.to(device)
    # net.to(device)
    # net.fit(X, y, n_epoch=2000, n_samples=10, batch_size=None)
    # train_end = time.time()
    # mu, var = net.predict(grid.to(device), n_samples=100)
    # print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    # plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)
