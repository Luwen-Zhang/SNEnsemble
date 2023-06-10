"""
This is a pytorch version of sklearn.decomposition.IncrementalPCA.
"""

from sklearn.decomposition import IncrementalPCA as skIncrementalPCA
from torch import nn
import torch
import numpy as np


def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs
    v *= signs.unsqueeze(-1)
    return u, v


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """
    A pytorch version of sklearn.utils.extmath._incremental_mean_and_var
    """
    last_sum = last_mean * last_sample_count
    X_nan_mask = torch.isnan(X)
    if torch.any(X_nan_mask):
        sum_op = torch.nansum
    else:
        sum_op = torch.sum

    new_sum = sum_op(X, dim=0)
    n_samples = X.shape[0]
    new_sample_count = n_samples - torch.sum(X_nan_mask, dim=0)

    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        correction = sum_op(temp, dim=0)
        temp **= 2
        new_unnormalized_variance = sum_op(temp, dim=0)

        # correction term of the corrected 2 pass algorithm.
        # See "Algorithms for computing the sample variance: analysis
        # and recommendations", by Chan, Golub, and LeVeque.
        new_unnormalized_variance -= correction**2 / new_sample_count

        last_unnormalized_variance = last_variance * last_sample_count

        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance
            + new_unnormalized_variance
            + last_over_new_count
            / updated_sample_count
            * (last_sum / last_over_new_count - new_sum) ** 2
        )

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


class IncrementalPCA(nn.Module):
    def __init__(self, n_components: int = None, on_cpu: bool = True):
        super(IncrementalPCA, self).__init__()
        self.components_ = None
        self.n_components_ = n_components
        self.register_buffer("n_samples_seen_", torch.zeros((1,)))
        self.register_buffer("mean_", torch.zeros((1,)))
        self.register_buffer("var_", torch.zeros((1,)))
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        self.initialized = False
        self.on_cpu = on_cpu

    def forward(self, x):
        # i.e. partial_fit
        device, x = self.to_cpu(x)
        if not self.initialized:
            self.initialize(x)
        if self.training:
            n_samples, n_features = x.shape
            col_mean, col_var, n_total_samples = _incremental_mean_and_var(
                x,
                last_mean=self.mean_,
                last_variance=self.var_,
                last_sample_count=self.n_samples_seen_.repeat(x.shape[1]),
            )
            n_total_samples = n_total_samples[0]

            # Whitening
            if self.n_samples_seen_ == 0:
                # If it is the first step, simply whiten X
                X = x - col_mean
            else:
                col_batch_mean = torch.mean(x, dim=0)
                X = x - col_batch_mean
                # Build matrix of combined previous basis and new data
                mean_correction = torch.sqrt(
                    (self.n_samples_seen_ / n_total_samples) * n_samples
                ) * (self.mean_ - col_batch_mean)
                X = torch.vstack(
                    (
                        self.singular_values_.view((-1, 1)) * self.components_,
                        X,
                        mean_correction,
                    )
                )

            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            U, Vt = svd_flip(U, Vt, u_based_decision=False)
            explained_variance = S**2 / (n_total_samples - 1)
            explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

            self.n_samples_seen_ = n_total_samples
            self.components_ = Vt[: self.n_components_]
            self.singular_values_ = S[: self.n_components_]
            self.mean_ = col_mean
            self.var_ = col_var
            self.explained_variance_ = explained_variance[: self.n_components_]
            self.explained_variance_ratio_ = explained_variance_ratio[
                : self.n_components_
            ]
            if self.n_components_ < n_features:
                self.noise_variance_ = explained_variance[self.n_components_ :].mean()
            else:
                self.noise_variance_ = 0.0
        x = x - self.mean_
        output = torch.matmul(x, self.components_.T)
        output = self.to_device(output, device)
        return output

    def initialize(self, x):
        self.batch_size = x.shape[0]
        if self.n_components_ is None:
            self.n_components_ = np.min(x.shape)
        elif not 1 <= self.n_components_ <= x.shape[1]:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components_, x.shape[1])
            )
        elif not self.n_components_ <= x.shape[0]:
            raise ValueError(
                "n_components=%r must be less or equal to "
                "the batch number of samples "
                "%d." % (self.n_components_, x.shape[0])
            )
        self.components_ = torch.zeros(
            self.n_components_, x.shape[1], device=x.device, requires_grad=False
        )
        self.initialized = True

    def to_cpu(self, x):
        if self.on_cpu:
            self.to("cpu")
            return x.device, x.to("cpu")
        else:
            return x.device, x

    def to_device(self, x, device):
        if self.on_cpu:
            return x.to(device)
        else:
            return x


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    X, _ = load_digits(return_X_y=True)
    skpca = skIncrementalPCA()
    torchpca = IncrementalPCA()
    skpca.partial_fit(X[:100, :])
    skpca.partial_fit(X[100:200, :])
    x_skpca = skpca.transform(X[:200, :])
    torchpca.train()
    torchpca(torch.tensor(X[:100, :]))
    torchpca(torch.tensor(X[100:200, :]))
    torchpca.eval()
    x_torchpca = torchpca(torch.tensor(X[:200, :]))
    # torch.matmul(torch.tensor(X[:100, :]), res[2])
