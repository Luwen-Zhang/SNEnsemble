"""
This is a modified version of https://github.com/swyoon/pytorch-minimal-gaussian-process/blob/main/gp.py
The repo does not follow any open source license.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np


class MiniBatchGP(nn.Module):
    """
    A gaussian process implementation for mini-batches. Hyperparameters are optimized by an Adam optimizer on each
    training step. Final results are based on all given batches during the first epoch (so the result is consistent
    with that obtained by training on the entire dataset).

    The procedure can be described as followed:
        During each step:
            When training:
            (1) If in the first epoch, append the batch to a buffer (create an empty one if it is the first step), then
            use the buffer as the data. If not, use the buffer directly.
            (2) Calculate all parameters.
            (3) Optimize hyperparameters.
            When predicting:
            (1) Load the data from the buffer and parameters.
            Finally, predict using the data and parameters.
        On epoch end: Check whether to stop optimizing hyperparameters.
    """

    def __init__(
        self,
        length_scale=1.0,
        noise_scale=1.0,
        amplitude_scale=1.0,
        dynamic_input=False,
    ):
        super().__init__()
        self.trained = False
        self.optim_hp = True
        self.dynamic_input = dynamic_input
        self.input_changing = dynamic_input
        self._records = {}
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self.previous_hp = [
            self.length_scale_.item(),
            self.noise_scale_.item(),
            self.amplitude_scale_.item(),
        ]
        self.data_buffer_ls = []
        self.optimizer = Adam(self.parameters(), lr=0.8)

    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def request_param(self, x, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            # Running sanity check.
            if name == "L":
                return torch.eye(x.shape[0], device=x.device)
            elif name == "X":
                return x
            elif name == "y":
                return torch.ones(x.shape[0], 1, device=x.device)
            elif name == "alpha":
                return torch.ones(x.shape[0], 1, device=x.device)
            else:
                return getattr(self, name)

    def on_train_start(self):
        self.trained = False
        self.optim_hp = True
        self.noise_scale_.requires_grad_(True)
        self.amplitude_scale_.requires_grad_(True)
        self.length_scale_.requires_grad_(True)

    def on_epoch_start(self):
        if self.input_changing and self.trained:
            self.trained = False
            for param in self.data_buffer_ls:
                self._records[param] = getattr(self, param)
                try:
                    self.__delattr__(param)
                except:
                    pass

    def on_epoch_end(self):
        self.trained = True
        previous_hp = self.previous_hp
        self.previous_hp = [
            self.length_scale_.item(),
            self.noise_scale_.item(),
            self.amplitude_scale_.item(),
        ]
        if self.input_changing and len(self._records) > 0:
            norm_previous = [
                torch.linalg.norm(self._records[x]) for x in self.data_buffer_ls
            ]
            norm_current = [
                torch.linalg.norm(getattr(self, x)) for x in self.data_buffer_ls
            ]
            if all(
                [
                    torch.abs(x - y) / y < 1e-5
                    for x, y in zip(norm_previous, norm_current)
                ]
            ):
                self.input_changing = False
        if (
            np.linalg.norm(np.array(previous_hp) - np.array(self.previous_hp))
            / np.linalg.norm(np.array(self.previous_hp))
            < 1e-5
        ) and not self.input_changing:
            self.optim_hp = False
            self.noise_scale_.requires_grad_(False)
            self.amplitude_scale_.requires_grad_(False)
            self.length_scale_.requires_grad_(False)

    def forward(self, x, y=None):
        if self.training and not self.trained:
            if y is not None:
                X = self.append_to_data_buffer(x, "X")
                y = self.append_to_data_buffer(y, "y")
            else:
                raise Exception(
                    f"Gaussian Process is training but the label is not provided."
                )
        else:
            X = self.request_param(x, "X")
            y = self.request_param(x, "y")
        if self.training and self.optim_hp:
            D = X.shape[1]
            K = self.kernel_mat_self(X)
            L = torch.linalg.cholesky(K)
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
            marginal_likelihood = (
                -0.5 * y.T.mm(alpha)
                - torch.log(torch.diag(L)).sum()
                - D * 0.5 * np.log(2 * np.pi)
            )
            self.L = L
            self.alpha = alpha
            self.nll_loss = -marginal_likelihood.sum()
        else:
            L = self.request_param(x, "L")
            alpha = self.request_param(x, "alpha")
        k = self.kernel_mat(X, x)
        v = torch.linalg.solve(L, k)
        mu = k.T.mm(alpha)
        var = -torch.diag(v.T.mm(v)) + self.amplitude_scale + self.noise_scale
        return mu, var

    def append_to_data_buffer(self, x, name):
        if name not in self.data_buffer_ls:
            self.data_buffer_ls.append(name)
        if not hasattr(self, name):
            self.register_buffer(name, x)
        else:
            previous = getattr(self, name)
            setattr(self, name, torch.concat([previous, x], dim=0))
        return getattr(self, name)

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

    def fit(self, X, y, batch_size: int = None, n_iter: int = None):
        self.on_train_start()
        self.train()
        if batch_size is None:
            dataloader = [(X, y)]
        else:
            from torch.utils import data as Data

            dataloader = Data.DataLoader(
                Data.TensorDataset(X, y), batch_size=batch_size, shuffle=True
            )
        for i_iter in range(n_iter if n_iter is not None else 1):
            self.on_epoch_start()
            for x, y_hat in dataloader:
                self(x, y_hat)
                if n_iter is not None:
                    self.optim_step()
            self.on_epoch_end()
        mu, var = self.predict(X)
        return mu, var

    def predict(self, X):
        self.eval()
        mu, var = self(X)
        return mu, var

    def optim_step(self):
        if self.optim_hp:
            self.optimizer.zero_grad()
            self.nll_loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(0)

    X = torch.randn(100, 1)
    f = torch.sin(X * 2 * np.pi / 4).flatten()
    y = f + torch.randn_like(f) * 0.1
    y = y[:, None]
    grid = torch.linspace(-5, 5, 200)[:, None]

    gp = MiniBatchGP()
    gp.fit(X, y, batch_size=10, n_iter=100)
    mu, var = gp.predict(grid)
    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()
    plt.figure()
    plt.plot(X.flatten(), y, ".", markersize=2)
    plt.plot(grid.flatten(), mu)
    plt.fill_between(grid.flatten(), y1=mu + std, y2=mu - std, alpha=0.3)
    plt.show()
    plt.close()
