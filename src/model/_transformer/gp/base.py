import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import List, Tuple


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
        sampling_on_hpo=False,
        **kwargs,
    ):
        super().__init__()
        self.trained = False
        self.optim_hp = True
        self.dynamic_input = dynamic_input
        self.input_changing = dynamic_input
        self.on_cpu = on_cpu
        self.sampling_on_hpo = sampling_on_hpo
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
        raise NotImplementedError

    def _record_params(self):
        raise NotImplementedError

    def _get_optimizer(self, **kwargs):
        raise NotImplementedError

    def _get_default_results(self, x: torch.Tensor, name: str) -> torch.Tensor:
        raise NotImplementedError

    def _set_requires_grad(self, requires_grad: bool):
        raise NotImplementedError

    def _train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _predict(self, X: torch.Tensor, x: torch.Tensor):
        raise NotImplementedError

    def _hp_converge_crit(self, previous_recorded: List, current_recorded: List):
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from ssgpr import MiniBatchSSGPR
    from original_gp import MiniBatchGP

    torch.manual_seed(0)

    X = torch.randn(100, 1) / 3
    f = (1 + X + 3 * X**2 + 0.5 * X**3).flatten()
    y = f + torch.randn_like(f) * 0.1
    y = y[:, None] + 1
    grid = torch.linspace(-5, 5, 200)[:, None]

    torch.manual_seed(0)
    start = time.time()
    gp = MiniBatchGP(on_cpu=False)
    gp.fit(X, y, batch_size=None, n_iter=100)
    train_end = time.time()
    mu, var = gp.predict(grid)
    print(f"GP: Train {train_end-start} s, Predict {time.time()-train_end} s")
    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()
    plt.figure()
    plt.plot(X.flatten(), y, ".", markersize=2)
    plt.plot(grid.flatten(), mu)
    plt.fill_between(grid.flatten(), y1=mu + std, y2=mu - std, alpha=0.3)
    plt.show()
    plt.close()

    torch.manual_seed(0)
    start = time.time()
    ssgpr = MiniBatchSSGPR(input_dim=1, num_basis_func=100, on_cpu=False)
    ssgpr.fit(X, y, batch_size=None, n_iter=100)
    train_end = time.time()
    mu, var = ssgpr.predict(grid)
    print(f"SSGPR: Train {train_end-start} s, Predict {time.time()-train_end} s")
    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()

    plt.figure()
    plt.plot(X.flatten(), y, ".", markersize=2)
    plt.plot(grid.flatten(), mu)
    plt.fill_between(grid.flatten(), y1=mu + std, y2=mu - std, alpha=0.3)
    plt.show()
    plt.close()
