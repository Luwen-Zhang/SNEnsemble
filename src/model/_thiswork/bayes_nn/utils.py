import torch
import matplotlib.pyplot as plt
import numpy as np


def get_test_case_1d(
    n_samples=100,
    noise=1,
    sample_std=1,
    grid_low=-5,
    grid_high=5,
    n_grid=200,
    func=None,
):
    torch.manual_seed(0)

    X = torch.randn(n_samples, 1) / sample_std
    if func is None:
        f = (
            3 * (1 - X) ** 2 * torch.exp(-(X**2) - 1)
            - 10 * (X / 5 - X**3) * torch.exp(-(X**2))
            - 1 / 3 * torch.exp(-((X + 1) ** 2))
        ).flatten()
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


def get_test_case_2d(
    n_samples=10,
    noise=0.1,
    sample_std_x=1,
    sample_std_y=1,
    grid_low_x=-3,
    grid_high_x=3,
    grid_low_y=-3,
    grid_high_y=3,
    n_grid_x=100,
    n_grid_y=100,
):
    torch.manual_seed(0)

    def make_grid(x1, x2):
        re_x1 = x1.repeat(1, len(x2))
        re_x2 = x2.repeat(1, len(x1)).T
        return torch.vstack([re_x1.flatten(), re_x2.flatten()]).T, re_x1, re_x2

    X1 = torch.randn(n_samples) / sample_std_x
    X2 = torch.randn(n_samples) / sample_std_y
    X = torch.vstack([X1, X2]).T
    x1 = X[:, 0]
    x2 = X[:, 1]
    f = (
        3 * (1 - x1) ** 2 * torch.exp(-(x1**2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * torch.exp(-(x1**2) - x2**2)
        - 1 / 3 * torch.exp(-((x1 + 1) ** 2) - x2**2)
    )
    y = f + torch.randn_like(f) * noise
    y = y[:, None]
    grid_x1 = torch.linspace(grid_low_x, grid_high_x, n_grid_x)[:, None]
    grid_x2 = torch.linspace(grid_low_y, grid_high_y, n_grid_y)[:, None]
    grid, plot_grid_x, plot_grid_y = make_grid(grid_x1, grid_x2)
    return X, y, grid, plot_grid_x, plot_grid_y


def _convert_to_cls(y: torch.Tensor, binary: bool):
    if binary:
        y = (y > torch.mean(y)).long()
    else:
        y = torch.round(y).long()
        y = y + torch.abs(torch.min(y))
    return y


def get_cls_test_case_1d(binary=True, *args, **kwargs):
    X, y, grid = get_test_case_1d(*args, **kwargs)
    y = _convert_to_cls(y, binary)
    return X, y, grid


def get_cls_test_case_2d(binary=True, *args, **kwargs):
    X, y, grid, plot_grid_x, plot_grid_y = get_test_case_2d(*args, **kwargs)
    y = _convert_to_cls(y, binary)
    return X, y, grid, plot_grid_x, plot_grid_y


def plot_mu_var_1d(X, y, grid, mu, var, markersize=2, alpha=0.3, limit_y=True):
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
    if limit_y:
        r = np.max(y) - np.min(y)
        plt.ylim([np.min(y) - 0.2 * r, np.max(y) + 0.2 * r])
    plt.show()
    # plt.close()


def plot_mu_var_2d(
    X, y, grid, mu, var, plot_grid_x, plot_grid_y, markersize=2, alpha=0.3, limit_y=True
):
    import copy

    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy().flatten()
    grid = grid.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy().flatten()
    var = var.detach().cpu().numpy().flatten()
    std = np.sqrt(var)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, ".", c="r", s=markersize)
    if limit_y:
        r = np.max(y) - np.min(y)
        lim = [np.min(y) - 0.2 * r, np.max(y) + 0.2 * r]
        ax.set_zlim(lim)

    def plot_once(value):
        cliped = copy.deepcopy(value)
        if limit_y:
            cliped[cliped < lim[0]] = np.nan
            cliped[cliped > lim[1]] = np.nan
        ax.plot_surface(
            plot_grid_x, plot_grid_y, cliped.reshape(*plot_grid_x.shape), alpha=alpha
        )

    plot_once(mu)
    plot_once(mu + std)
    plot_once(mu - std)
    plt.show()
    # plt.close()
