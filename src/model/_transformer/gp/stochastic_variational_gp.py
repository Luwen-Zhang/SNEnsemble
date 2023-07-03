from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch
import torch
from torch.optim import Adam
from typing import List

try:
    from .base import AbstractGPyTorch, get_test_case_1d, plot_mu_var_1d
except:
    from base import AbstractGPyTorch, get_test_case_1d, plot_mu_var_1d


class _StochasticVariationalModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(_StochasticVariationalModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StochasticVariationalModel(AbstractGPyTorch):
    @property
    def incremental(self):
        return True

    def __init__(self, inducing_points, num_data, **kwargs):
        gp = _StochasticVariationalModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=num_data)
        super(StochasticVariationalModel, self).__init__(
            gp=gp, likelihood=likelihood, loss_func=mll, **kwargs
        )

    def _record_params(self):
        return [param.clone() for param in self.parameters() if param.requires_grad]

    def _get_optimizer(self, lr=0.01, **kwargs):
        return Adam(
            [
                {"params": self.gp.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
        )

    def _hp_converge_crit(self, previous_recorded: List, current_recorded: List):
        return all(
            [
                torch.linalg.norm(x - y) / torch.linalg.norm(y) < 1e-5
                for x, y in zip(previous_recorded, current_recorded)
            ]
        )


def train_approx_gp(
    model, likelihood, loss_func, optimizer, X, y, training_iter=50, batch_size=10
):
    from torch.utils import data as Data

    model.train()
    likelihood.train()

    data_loader = Data.DataLoader(
        Data.TensorDataset(X, y), batch_size=batch_size, shuffle=True
    )

    for i in range(training_iter):
        for x, y_hat in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = -loss_func(output, y_hat.flatten())
            loss.backward()
            optimizer.step()


def predict_approx_gp(model, likelihood, grid):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(grid))

    mu = observed_pred.mean
    var = observed_pred.variance
    return mu, var


if __name__ == "__main__":
    import time

    X, y, grid = get_test_case_1d(100, 1)

    inducing_points = X[:10, :]
    torch.manual_seed(0)
    start = time.time()
    model = _StochasticVariationalModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.01,
    )

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))

    train_approx_gp(
        model, likelihood, mll, optimizer, X, y, training_iter=10, batch_size=10
    )
    train_end = time.time()
    mu, var = predict_approx_gp(model, likelihood, grid)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)

    torch.manual_seed(0)
    start = time.time()
    gp = StochasticVariationalModel(inducing_points=inducing_points, num_data=y.size(0))
    gp.fit(X, y, batch_size=10, n_iter=10)
    train_end = time.time()
    mu, var = gp.predict(grid)
    print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")
    plot_mu_var_1d(X, y, grid, mu, var)
