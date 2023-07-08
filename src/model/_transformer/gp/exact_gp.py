import gpytorch
from torch.optim import Adam
import numpy as np
from typing import List
import torch

try:
    from .base import AbstractGPyTorch
except:
    from base import AbstractGPyTorch


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(AbstractGPyTorch):
    def __init__(self, **kwargs):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = _ExactGPModel(train_x=None, train_y=None, likelihood=likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        super(ExactGPModel, self).__init__(
            gp=gp, likelihood=likelihood, loss_func=mll, **kwargs
        )

    def _record_params(self):
        return [
            self.gp.covar_module.base_kernel.lengthscale.item(),
            self.gp.likelihood.noise.item(),
        ]

    def _get_optimizer(self, lr=0.1, **kwargs):
        return Adam(self.parameters(), lr=lr)

    def _hp_converge_crit(self, previous_recorded: List, current_recorded: List):
        return (
            np.linalg.norm(np.array(previous_recorded) - np.array(current_recorded))
            / np.linalg.norm(np.array(current_recorded))
            < 1e-5
        )


def train_exact_gp(
    model, likelihood, loss_func, optimizer, X, y, training_iter=50, verbose=True
):
    model.train()
    likelihood.train()

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -loss_func(output, y.flatten())
        loss.backward()
        if verbose:
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item(),
                )
            )
        optimizer.step()


def predict_exact_gp(model, likelihood, grid):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(grid))

    mu = observed_pred.mean
    var = observed_pred.variance
    return mu, var


if __name__ == "__main__":
    import time
    from base import get_test_case_1d, plot_mu_var_1d, get_test_case_2d, plot_mu_var_2d
    import matplotlib.pyplot as plt
    import src.utils

    X, y, grid = get_test_case_1d(100, grid_low=-10, grid_high=10, noise=1)

    torch.manual_seed(0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    start = time.time()
    model = _ExactGPModel(X, y.flatten(), likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    train_exact_gp(
        model, likelihood, mll, optimizer, X, y, training_iter=200, verbose=False
    )
    train_end = time.time()
    mu, var = predict_exact_gp(model, likelihood, grid)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    torch.manual_seed(0)
    start = time.time()
    gp = ExactGPModel(on_cpu=False)
    gp.fit(X, y, batch_size=None, n_iter=200)
    train_end = time.time()
    mu, var = gp.predict(grid)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(
        100, grid_high_x=3, grid_low_x=-3, grid_low_y=-3, grid_high_y=3
    )
    torch.manual_seed(0)
    start = time.time()
    gp = ExactGPModel(on_cpu=False)
    gp.fit(X, y, batch_size=None, n_iter=200)
    train_end = time.time()
    mu, var = gp.predict(grid)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_2d(X, y, grid, mu, var, plot_grid_x, plot_grid_y)
