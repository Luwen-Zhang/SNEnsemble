import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from typing import Union
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP, ApproximateGP
import matplotlib.pyplot as plt


class AbstractGP(nn.Module):
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
        dynamic_input=False,
        on_cpu=False,
        **kwargs,
    ):
        super().__init__()
        self.trained = False
        self.optim_hp = True
        self.dynamic_input = dynamic_input
        self.input_changing = dynamic_input
        self.on_cpu = on_cpu
        self._records = {}
        self._register_params(**kwargs)
        try:
            self.previous_hp = self._record_params()
        except NotImplementedError:
            raise NotImplementedError(f"_record_params is not implemented.")
        except:
            self.previous_hp = []
        self.data_buffer_ls = []
        self.kwargs = kwargs
        self.optimizer = None

    def _register_params(self, **kwargs):
        """
        Register torch.nn.Parameter to self. It is also ok to register them in ``__init__`` after calling
        ``super().__init__(**kwargs)``. These parameters will be optimized by the optimizer from
        ``self._get_optimizer`` if they ``requires_grad``.

        Parameters
        -------
        kwargs
            kwargs passed to ``__init__``
        """
        raise NotImplementedError

    def _record_params(self) -> List:
        """
        Record hyperparameters registered in ``_register_params``. These records are used to check the convergence and
        will be passed to ``_hp_converge_crit``.
        """
        raise NotImplementedError

    def _get_optimizer(self, **kwargs):
        """
        Get an optimizer from torch.optim.

        Parameters
        -------
        kwargs
            kwargs passed to ``__init__``

        Returns
        -------
        optimizer
            An optimizer for nn.Module.
        """
        raise NotImplementedError

    def _get_default_results(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        For pytorch_lightning, the sanity check runs several evaluation steps before the first training step. It is
        possible that some attributes are not initialized; therefore, when implementing ``_train``, it is better to use
        ``self.request_param`` instead of directly calling ``self.XXX`` to get default attributes from this method.
        The returned value should reveal the dimensions of the requested attribute to prevent runtime error. These
        values will not be used during actual training or evaluating stages.

        Parameters
        ----------
        x
            The tensor passed to ``forward``
        name
            The name of requested attribute passed to ``self.request_param``.

        Returns
        -------
        value
            A tensor that reveal the dimensions of the requested attribute.
        """
        raise NotImplementedError

    def _set_requires_grad(self, requires_grad: bool):
        """
        Set requires_grad for parameters registered in ``_register_params``. It is used to control the optimization of
        hyperparameters. The simplest way could be self.train(requires_grad) if no other params are trained after
        hyperparameters converge.
        """
        raise NotImplementedError

    def _train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        The main training step. An objective value (loss) should be returned. Necessary results should be stored as
        attributes of the instance for ``_predict``.

        Parameters
        ----------
        X
            All received data in previous steps during this epoch and the current step if it is the first epoch or
            (``self.dynamic_input=True`` and ``self.input_changing=True``). Otherwise, it is all received data during
            the first epoch if ``self.dynamic_input=False``; or the converged data
            if ``self.dynamic_input=True`` and `self.input_changing=False``.
        y
            The corresponding target.

        Returns
        -------
        loss
            A scalar loss value
        """
        raise NotImplementedError

    def _predict(
        self, X: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction. It is called after calling ``_train``. It is possible that the model is under the sanity
        check of pytorch_lightning, so in this method, it is better to call attributes using ``self.request_param``.
        See also ``_get_default_results``.

        Parameters
        ----------
        X
            The training data. See the docstring of ``_train``.
        x
            The new data.

        Returns
        -------
        mu, var
            Means and variances.
        """
        raise NotImplementedError

    def _hp_converge_crit(
        self, previous_recorded: List, current_recorded: List
    ) -> bool:
        """
        Check convergence of hyperparameters recorded by ``_record_params``. Inputs are lists recorded before and after
        optimization respectively.
        """
        raise NotImplementedError

    def request_param(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            # Running sanity check.
            if name == "X":
                return x
            elif name == "y":
                return torch.ones(x.shape[0], 1, device=x.device)
            else:
                try:
                    return self._get_default_results(x, name).to(x.device)
                except:
                    return getattr(self, name)

    def on_train_start(self):
        self.trained = False
        self.optim_hp = True
        self._set_requires_grad(True)

    def on_epoch_start(self):
        if self.input_changing and self.trained:
            self.trained = False
            for param in self.data_buffer_ls:
                self._records[param] = getattr(self, param)
                try:
                    self.__delattr__(param)
                except:
                    pass
            torch.cuda.empty_cache()

    def on_epoch_end(self):
        self.trained = True
        previous_hp = self.previous_hp
        self.previous_hp = self._record_params()
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
        if len(previous_hp) != 0:
            hp_converged = self._hp_converge_crit(previous_hp, self.previous_hp)
        else:
            hp_converged = False
        if hp_converged and not self.input_changing:
            self.optim_hp = False
            self._set_requires_grad(False)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, return_prediction=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, x = self.to_cpu(x)
        if y is not None:
            _, y = self.to_cpu(y)
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
            self.loss = self._train(X, y)
        if return_prediction:
            mu, var = self._predict(X, x)
            mu = self.to_device(mu, device)
            var = self.to_device(var, device)
            return mu, var

    def append_to_data_buffer(self, x: torch.Tensor, name: str):
        if name not in self.data_buffer_ls:
            self.data_buffer_ls.append(name)
        if not hasattr(self, name):
            self.register_buffer(name, x)
        else:
            previous = getattr(self, name)
            setattr(self, name, torch.concat([previous, x], dim=0))
        return getattr(self, name)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = None,
        n_iter: int = None,
    ):
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
                self(x, y_hat, return_prediction=False)
                if n_iter is not None:
                    self.optim_step()
            self.on_epoch_end()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        mu, var = self(X)
        return mu, var

    def optim_step(self):
        if self.optimizer is None:
            self.optimizer = self._get_optimizer(**self.kwargs)
        if self.optim_hp:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

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






def get_test_case_1d(
    n_samples=100,
    n_dim=1,
    noise=0.1,
    sample_std=1,
    grid_low=-5,
    grid_high=5,
    n_grid=200,
    func=None,
):
    torch.manual_seed(0)

    X = torch.randn(n_samples, n_dim) / sample_std
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
