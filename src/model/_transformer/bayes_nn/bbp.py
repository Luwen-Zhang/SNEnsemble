"""
This is a modification of https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/regression/bbp_hetero.ipynb
improved by comparing https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop/model.py (for stability and classification tasks)
and further improved by https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py (for local reparameterization)
The MC Dropout model is implemented under the consistent framework. https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/regression/mc_dropout_hetero.ipynb
For classification tasks, the uncertainty is accessed by the approach proposed by two papers written by Y Kwon et al.:
    1. Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation
    2. Uncertainty quantification using Bayesian neural networks in classification: Application to ischemic stroke lesion segmentation
Parallel MC sampling is also implemented.
WARNING: Multi-output regression/classification tasks are not well tested because it is not the case for our paper.
"""

import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import warnings
from src.model.base import KeepDropout


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
                act_weight_mus = torch.matmul(x, self.weight_mus)
                act_weight_stds = torch.sqrt(
                    _safe_positive(
                        torch.matmul(_safe_pow(x, 2), _safe_pow(weight_stds, 2)),
                        eps=self.eps,
                    )
                )
                weight_epsilons = torch.randn_like(act_weight_mus, device=x.device)
                out_weight_sample = act_weight_mus + act_weight_stds * weight_epsilons
            else:
                # sample Gaussian noise for each weight and each bias
                weight_epsilons = torch.randn(
                    *list(x.shape[:-2] + self.weight_mus.shape), device=x.device
                )
                # calculate samples from the posterior from the sampled noise and mus/stds
                weight_sample = self.weight_mus + weight_epsilons * weight_stds
                out_weight_sample = torch.matmul(x, weight_sample)
            bias_epsilons = torch.randn(
                *list(x.shape[:-2] + self.bias_mus.shape), device=x.device
            )
            bias_sample = self.bias_mus + bias_epsilons * bias_stds
            output = out_weight_sample + bias_sample.unsqueeze(1)

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
            output = torch.matmul(x, self.weight_mus) + self.bias_mus
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
        verbose=True,
        **kwargs,
    ):
        super(AbstractBNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.on_cpu = on_cpu
        self.task = task
        self.type = type
        self.eps = eps
        self.verbose = verbose
        self.optimizer = None
        self.kwargs = kwargs
        self.loss_func = None
        self.output_act = nn.Identity()
        if self.task == "classification":
            self._force_homo(f"For classification tasks")
            if self.n_outputs <= 2:
                # Binary classification is reduced to one-dimensional, and the output is passed through sigmoid.
                self.n_outputs = 1
                self.output_act = nn.Sigmoid()
                self.loss_func = nn.BCELoss()
            else:
                self.output_act = nn.LogSoftmax(dim=-1)
                self.loss_func = nn.NLLLoss()
        else:
            if self.n_outputs > 1:
                self._force_homo(f"For multi-output regression tasks")
                self.loss_func = nn.MSELoss()
            else:
                self.loss_func = _isotropic_gauss_log_likelihood
            if self.type == "homo":
                self.log_noise = nn.Parameter(torch.tensor([init_log_noise]))
        if self.n_outputs > 1:
            warnings.warn(
                f"Multi-output tasks are not verified because it is not the case for our paper, so I will not take "
                f"responsibility for the results if you use it in your research."
            )
        if self.loss_func is None:
            raise Exception(f"Type {self.type} for {task} tasks is not implemented. ")

    def _force_homo(self, msg):
        if self.type == "hete":
            warnings.warn(f"{msg}, homoscedastic model is used.")
        self.type = "homo"

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

    def _get_optimizer(self, lr=0.01, weight_decay=1e-9, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, X, y, batch_size=None, n_epoch=100, n_samples=10):
        self.train()
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
        self.eval()
        with torch.no_grad():
            preds = self._predict_step(X, n_samples).to(X.device)
            if self.type == "hete":
                samples = preds[:, :, :1]
                noises = preds[:, :, 1:]
            else:
                samples = preds

        if self.task == "classification" and self.n_outputs > 1:
            exp_samples = torch.exp(samples)
            mean = torch.argmax(torch.mean(exp_samples, dim=0), dim=-1)
            epistemic_var = torch.var(exp_samples, dim=0, unbiased=False)[
                torch.arange(mean.shape[0]), mean
            ]
        else:
            mean = torch.mean(samples, dim=0)
            epistemic_var = torch.var(samples, dim=0, unbiased=False)
        if self.task == "classification":
            # See https://github.com/ykwon0407/UQ_BNN/tree/master
            # For epistemic_var, the described np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2 is
            # consistent with the above torch.var line.
            if self.n_outputs == 1:
                p_hat = samples
                aleatoric_var = torch.mean(p_hat * (1 - p_hat), dim=0)
            else:
                p_hat = torch.exp(samples)
                aleatoric_var = torch.mean(p_hat * (1 - p_hat), dim=0)[
                    torch.arange(mean.shape[0]), mean
                ]
        else:
            if self.type == "homo":
                stds = _safe_exp(self.log_noise)
                aleatoric_var = torch.ones_like(epistemic_var, device=X.device) * stds
            else:
                stds = torch.exp(noises)
                aleatoric_var = torch.mean(_safe_pow(stds, 2), dim=0)
        var = epistemic_var + aleatoric_var
        mean = mean.float()
        return mean.squeeze(), var.squeeze()

    def get_sample_kl_loss(self, kl_loss, batch_size, n_batches):
        return kl_loss / batch_size / n_batches

    def get_sample_fitness_loss(self, output, y_hat):
        if self.loss_func == _isotropic_gauss_log_likelihood:
            # _isotropic_gauss_log_likelihood is compatible with batched output.
            if self.type == "hete":
                fit_loss = -self.loss_func(
                    output[:, :, :1],
                    y_hat.unsqueeze(0),
                    _safe_exp(output[:, :, 1:]),
                    eps=self.eps,
                ).mean()
            else:
                fit_loss = -self.loss_func(
                    output, y_hat.unsqueeze(0), _safe_exp(self.log_noise), eps=self.eps
                ).mean()
        else:
            fit_loss = 0
            for i_batch in range(output.shape[0]):
                if isinstance(self.loss_func, nn.NLLLoss):
                    batch_fit_loss = self.loss_func(
                        output[i_batch], y_hat.flatten().long()
                    )
                elif isinstance(self.loss_func, nn.BCELoss):
                    batch_fit_loss = self.loss_func(
                        output[i_batch].flatten(), y_hat.flatten().float()
                    )
                else:
                    batch_fit_loss = self.loss_func(output[i_batch], y_hat)
                fit_loss = fit_loss + batch_fit_loss
            fit_loss = fit_loss / output.shape[0]
        return fit_loss

    def _train_step(self, x, y, n_samples, n_batches, **kwargs):
        raise NotImplementedError

    def _predict_step(self, X, n_samples):
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
        eps=1e-6,
        kl_weight=1.0,
        local_reparam=False,
        **kwargs,
    ):
        super(BayesByBackprop, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, eps=eps, **kwargs
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
        if self.n_inputs == 1 and x.shape[-1] != self.n_inputs:
            x = x.unsqueeze(-1)

        for idx, layer in enumerate(self.layers):
            x, kl_loss = layer(x)
            kl_loss_total = kl_loss_total + kl_loss
            if idx != len(self.layers) - 1:
                x = self.activation(x)
            else:
                x = self.output_act(x)

        x = self.to_device(x, device)
        kl_loss_total = self.to_device(kl_loss_total, device)
        return x, kl_loss_total

    def get_sample_loss(self, output, y_hat, kl_loss, batch_size, n_batches):
        return self.get_sample_fitness_loss(output, y_hat), self.get_sample_kl_loss(
            kl_loss, batch_size, n_batches
        )

    def _train_step(self, x, y, n_samples, n_batches, **kwargs):
        sample_x = x.unsqueeze(0).repeat(n_samples, 1, 1)
        output, kl_loss = self(sample_x)
        fit_loss_step, kl_loss_step = self.get_sample_loss(
            output,
            y,
            kl_loss,
            batch_size=x.shape[0],
            n_batches=n_batches,
        )
        total_loss = fit_loss_step + self.kl_weight * kl_loss_step
        if self.verbose:
            self.fit_loss_epoch += fit_loss_step.item() * x.shape[0]
            self.kl_loss_epoch += kl_loss_step.item() * x.shape[0]
        return total_loss

    def _predict_step(self, X, n_samples):
        sample_x = X.unsqueeze(0).repeat(n_samples, 1, 1)
        return self(sample_x)[0]

    def on_train_epoch_start(self, X, y):
        if self.verbose:
            self.fit_loss_epoch = 0
            self.kl_loss_epoch = 0

    def on_train_epoch_end(self, X, y, i_epoch, n_epoch):
        if self.verbose:
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
    def __init__(self, n_inputs, n_outputs, layers, dropout=0.2, **kwargs):
        super(MCDropout, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, **kwargs
        )
        from src.model.base import get_sequential

        # network with two hidden and one output layer
        self.layers = get_sequential(
            layers=layers,
            n_inputs=n_inputs,
            n_outputs=2 * self.n_outputs if self.type == "hete" else self.n_outputs,
            use_norm=False,
            dropout=dropout,
            act_func=nn.ReLU,
            dropout_keep_training=True,
        )

    def forward(self, x):
        device, x = self.to_cpu(x)
        if self.n_inputs == 1 and x.shape[-1] != self.n_inputs:
            x = x.unsqueeze(-1)
        x = self.layers(x)
        x = self.output_act(x)
        x = self.to_device(x, device)
        return x

    def _train_step(self, x, y, **kwargs):
        output = self(x)
        fit_loss_step = self.get_sample_fitness_loss(output.unsqueeze(0), y)
        if self.verbose:
            self.fit_loss_epoch += fit_loss_step * x.shape[0]
        return fit_loss_step

    def _predict_step(self, X, n_samples):
        sample_x = X.unsqueeze(0).repeat(n_samples, 1, 1)
        with KeepDropout():
            res = self(sample_x)
        return res

    def on_train_epoch_start(self, X, y):
        if self.verbose:
            self.fit_loss_epoch = 0

    def on_train_epoch_end(self, X, y, i_epoch, n_epoch):
        if self.verbose:
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
        get_cls_test_case_1d,
        get_cls_test_case_2d,
    )

    device = "cpu"
    X, y, grid = get_test_case_1d(100, grid_low=-3, grid_high=3, noise=0.2)
    torch.manual_seed(0)
    start = time.time()
    net = BayesByBackprop(
        n_inputs=1, n_outputs=1, n_layers=2, n_hidden=20, on_cpu=False, type="homo"
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=1000, n_samples=10, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=100)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(
        100, grid_low_x=-3, grid_high_x=3, grid_low_y=-3, grid_high_y=3, noise=0.2
    )
    torch.manual_seed(0)
    start = time.time()
    net = BayesByBackprop(
        n_inputs=2, n_outputs=1, n_layers=4, n_hidden=50, on_cpu=False, type="homo"
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=2000, n_samples=10, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=100)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)

    X, y, grid = get_test_case_1d(100, grid_low=-5, grid_high=5, noise=0.0)
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=1,
        n_outputs=1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="homo",
        dropout=0.5,
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=1000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(
        100, grid_low_x=-3, grid_high_x=3, grid_low_y=-3, grid_high_y=3, noise=0.2
    )
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=2,
        n_outputs=1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="hete",
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)

    X, y, grid = get_cls_test_case_1d(
        binary=True, n_samples=1000, grid_low=-5, grid_high=10, noise=0.1
    )
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=1,
        n_outputs=torch.max(y).item() + 1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="hete",
        task="classification",
        dropout=0.5,
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_cls_test_case_2d(
        binary=True,
        n_samples=100,
        grid_low_x=-3,
        grid_high_x=3,
        grid_low_y=-3,
        grid_high_y=3,
        noise=0.2,
    )
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=2,
        n_outputs=torch.max(y).item() + 1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="hete",
        task="classification",
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)

    X, y, grid = get_cls_test_case_1d(
        binary=False, n_samples=1000, grid_low=-5, grid_high=10, noise=0.1
    )
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=1,
        n_outputs=torch.max(y).item() + 1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="hete",
        task="classification",
        dropout=0.5,
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_cls_test_case_2d(
        binary=False,
        n_samples=100,
        grid_low_x=-3,
        grid_high_x=3,
        grid_low_y=-3,
        grid_high_y=3,
        noise=0.2,
    )
    torch.manual_seed(0)
    start = time.time()
    net = MCDropout(
        n_inputs=2,
        n_outputs=torch.max(y).item() + 1,
        layers=[128, 64, 32],
        on_cpu=False,
        type="hete",
        task="classification",
        lr=0.01,
    )
    X = X.to(device)
    y = y.to(device)
    net.to(device)
    net.fit(X, y, n_epoch=5000, batch_size=None)
    train_end = time.time()
    mu, var = net.predict(grid.to(device), n_samples=1000)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)
