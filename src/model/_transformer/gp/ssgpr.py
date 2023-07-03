"""
This is a modified version of https://github.com/linesd/SSGPR/tree/master
The repo is under MIT license.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from .base import AbstractGP


class MiniBatchSSGPR(AbstractGP):
    def __init__(self, input_dim, num_basis_func=100, **kwargs):
        self.input_dim = input_dim
        self.num_basis_func = num_basis_func
        super(MiniBatchSSGPR, self).__init__(**kwargs)

    @property
    def amplitude(self):
        return torch.abs(self._amplitude)

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def length_scales(self):
        return torch.abs(self._length_scales)

    @property
    def variance(self):
        return torch.abs(self._variance)

    def _register_params(self, **kwargs):
        self._length_scales = nn.Parameter(torch.ones(self.input_dim, 1))
        self._frequencies = nn.Parameter(
            torch.ones(self.num_basis_func, self.input_dim)
        )
        self._amplitude = nn.Parameter(torch.tensor(1.0))
        self._variance = nn.Parameter(torch.tensor(1.0))

    def _record_params(self):
        return [
            self._length_scales.clone(),
            self._frequencies.clone(),
            self._amplitude.clone(),
            self._variance.clone(),
        ]

    def _get_optimizer(self, lr=0.1, **kwargs):
        return Adam(self.parameters(), lr=lr)

    def _get_default_results(self, x, name):
        if name == "L":
            return torch.eye(x.shape[0], device=x.device)
        elif name == "alpha":
            return torch.ones(x.shape[0], 1, device=x.device)
        else:
            raise RuntimeError

    def _set_requires_grad(self, requires_grad):
        self._length_scales.requires_grad_(requires_grad)
        self._frequencies.requires_grad_(requires_grad)
        self._amplitude.requires_grad_(requires_grad)
        self._variance.requires_grad_(requires_grad)

    def _train(self, X, y):
        N, D = X.shape
        phi = self.design_matrix(X)
        A = (
            self.amplitude / self.num_basis_func
        ) * phi.T @ phi + self.variance * torch.eye(
            2 * self.num_basis_func, device=X.device
        )
        L = torch.linalg.cholesky(A.to(torch.float64))
        Rtiphity = torch.linalg.solve(L, (phi.T @ y).to(torch.float64))
        alpha = (
            self.amplitude
            / self.num_basis_func
            * torch.linalg.solve(L.T, Rtiphity).to(torch.float32)
        )
        # negative marginal log likelihood
        nmll = (
            y.T @ y - (self.amplitude / self.num_basis_func) * torch.sum(Rtiphity**2)
        ) / (2 * self.variance)
        nmll += torch.log(torch.diag(L)).sum() + (
            (N / 2) - self.num_basis_func
        ) * torch.log(self.variance)
        nmll += (N / 2) * np.log(2 * np.pi)

        self.L = L
        self.alpha = alpha
        return nmll

    def _predict(self, X, x):
        L = self.request_param(x, "L")
        alpha = self.request_param(x, "alpha")

        phi_star = self.design_matrix(x)
        mu = (phi_star @ alpha).reshape(-1, 1)  # predictive mean
        var = self.variance * (
            1
            + self.amplitude
            / self.num_basis_func
            * torch.sum(
                (phi_star @ torch.linalg.inv(L.T).to(torch.float32)) ** 2, dim=1
            )
        )
        return mu, var

    def _hp_converge_crit(self, previous_recorded, current_recorded):
        return all(
            [
                torch.linalg.norm(x - y) / torch.linalg.norm(y) < 1e-5
                for x, y in zip(previous_recorded, current_recorded)
            ]
        )

    def design_matrix(self, X):
        """
        Create Trigonometric Basis Function (TBF) design matrix.
        """
        N = X.shape[0]
        W = self.frequencies / self.length_scales.T
        phi_x = torch.zeros(N, 2 * self.num_basis_func, device=X.device)
        phi_x[:, : self.num_basis_func] = torch.cos(X @ W.T)
        phi_x[:, self.num_basis_func :] = torch.sin(X @ W.T)
        return phi_x


if __name__ == "__main__":
    import time
    from base import get_test_case_1d, plot_mu_var_1d

    X, y, grid = get_test_case_1d(100, 1)

    torch.manual_seed(0)
    start = time.time()
    ssgpr = MiniBatchSSGPR(input_dim=1, num_basis_func=100)
    ssgpr.fit(X, y, batch_size=None, n_iter=1000)
    train_end = time.time()
    mu, var = ssgpr.predict(grid)
    print(f"Train {train_end-start} s, Predict {time.time()-train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)
