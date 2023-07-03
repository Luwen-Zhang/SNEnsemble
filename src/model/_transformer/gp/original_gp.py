"""
This is a modified version of https://github.com/swyoon/pytorch-minimal-gaussian-process/blob/main/gp.py
The repo does not follow any open source license.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from base import AbstractGP


class MiniBatchGP(AbstractGP):
    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def _register_params(
        self, length_scale=1.0, noise_scale=1.0, amplitude_scale=1.0, **kwargs
    ):
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

    def _record_params(self):
        self.previous_hp = [
            self.length_scale_.item(),
            self.noise_scale_.item(),
            self.amplitude_scale_.item(),
        ]

    def _get_optimizer(self, lr=0.1, **kwargs):
        return Adam(self.parameters(), lr=lr)

    def _get_default_param(self, x, name):
        if name == "L":
            return torch.eye(x.shape[0], device=x.device)
        elif name == "alpha":
            return torch.ones(x.shape[0], 1, device=x.device)
        else:
            raise RuntimeError

    def _set_requires_grad(self, requires_grad):
        self.noise_scale_.requires_grad_(requires_grad)
        self.amplitude_scale_.requires_grad_(requires_grad)
        self.length_scale_.requires_grad_(requires_grad)

    def _train(self, X, y):
        D = X.shape[1]
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K.to(torch.float64))
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y.to(torch.float64))).to(
            torch.float32
        )
        marginal_likelihood = (
            -0.5 * y.T.mm(alpha)
            - torch.log(torch.diag(L)).sum()
            - D * 0.5 * np.log(2 * np.pi)
        )
        self.L = L
        self.alpha = alpha
        return -marginal_likelihood.sum()

    def _predict(self, X, x):
        L = self.request_param(x, "L")
        alpha = self.request_param(x, "alpha")
        k = self.kernel_mat(X, x)
        v = torch.linalg.solve(L.to(torch.float64), k.to(torch.float64)).to(
            torch.float32
        )
        mu = k.T.mm(alpha)
        var = -torch.diag(v.T.mm(v)) + self.amplitude_scale + self.noise_scale
        return mu, var

    def _hp_converge_crit(self, previous_recorded, current_recorded):
        return (
            np.linalg.norm(np.array(previous_recorded) - np.array(current_recorded))
            / np.linalg.norm(np.array(current_recorded))
            < 1e-5
        )

    def kernel_mat_self(self, X):
        sq = (X**2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 * X.mm(X.T)
        return torch.mul(
            self.amplitude_scale, torch.exp(-0.5 * sqdist / self.length_scale)
        ) + torch.mul(self.noise_scale, torch.eye(len(X), device=X.device))

    def kernel_mat(self, X, Z):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return torch.mul(
            self.amplitude_scale, torch.exp(-0.5 * sqdist / self.length_scale)
        )
