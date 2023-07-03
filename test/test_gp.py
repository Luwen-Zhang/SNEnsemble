import unittest
from import_utils import *
import src
from src.model._transformer.gp.exact_gp import (
    _ExactGPModel,
    ExactGPModel,
    train_exact_gp,
    predict_exact_gp,
)
from src.model._transformer.gp.stochastic_variational_gp import (
    _StochasticVariationalModel,
    StochasticVariationalModel,
    train_approx_gp,
    predict_approx_gp,
)
import time
import torch
import gpytorch
from src.model._transformer.gp.base import get_test_case_1d, plot_mu_var_1d


class TestGP(unittest.TestCase):
    def test_exact_gp(self):
        X, y, grid = get_test_case_1d(100, 1)

        torch.manual_seed(0)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        start = time.time()
        model = _ExactGPModel(X, y.flatten(), likelihood)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        train_exact_gp(
            model, likelihood, mll, optimizer, X, y, training_iter=50, verbose=False
        )
        train_end = time.time()
        mu1, var1 = predict_exact_gp(model, likelihood, grid)
        print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")

        torch.manual_seed(0)
        start = time.time()
        gp = ExactGPModel()
        gp.fit(X, y, batch_size=None, n_iter=50)
        train_end = time.time()
        mu2, var2 = gp.predict(grid)
        print(f"Train {train_end - start} s, Predict {time.time() - train_end} s")

        assert torch.allclose(mu1, mu2), f"Means are not consistent"
        assert torch.allclose(var1, var2), f"Variances are not consistent"

    def test_variational_gp(self):
        X, y, grid = get_test_case_1d(100, 1)

        inducing_points = X[:10, :]
        torch.manual_seed(0)
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
        mu1, var1 = predict_approx_gp(model, likelihood, grid)

        torch.manual_seed(0)
        gp = StochasticVariationalModel(
            inducing_points=inducing_points, num_data=y.size(0)
        )
        gp.fit(X, y, batch_size=10, n_iter=10)
        mu2, var2 = gp.predict(grid)

        assert torch.allclose(mu1, mu2), f"Means are not consistent"
        assert torch.allclose(var1, var2), f"Variances are not consistent"
