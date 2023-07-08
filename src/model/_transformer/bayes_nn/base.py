import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class AbstractBNN(nn.Module):
    def __init__(self):
        super(AbstractBNN, self).__init__()


def get_test_case_1d(
    n_samples=100,
    noise=0.1,
    sample_std=1,
    grid_low=-5,
    grid_high=5,
    n_grid=200,
    func=None,
):
    torch.manual_seed(0)

    X = torch.randn(n_samples, 1) / sample_std
    if func is None:
        f = (1 + X + 3 * X**2 + 0.5 * X**3).flatten()
    else:
        f = func(X).flatten()
    if f.shape[0] != n_samples:
        raise Exception(
            f"Get the target with {f.shape[0]} components from `func`, but expect n_samples={n_samples} components."
        )
    y = f + torch.randn_like(f) * noise
    y = y[:, None]
    grid = torch.linspace(grid_low, grid_high, n_grid)[:, None]
    return X, y, grid


def plot_mu_var_1d(X, y, grid, mu, var, markersize=2, alpha=0.3):
    X = X.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()
    grid = grid.detach().cpu().numpy().flatten()
    mu = mu.detach().cpu().numpy().flatten()
    var = var.detach().cpu().numpy().flatten()
    std = np.sqrt(var)
    plt.figure()
    plt.plot(X, y, ".", markersize=markersize)
    plt.plot(grid, mu)
    plt.fill_between(grid, y1=mu + std, y2=mu - std, alpha=alpha)
    plt.show()
    plt.close()
