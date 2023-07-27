"""
This is a modification of sklearn.mixture.BayesianGaussianMixture for pytorch.
"""
from .gmm import GMM, Cluster
import torch
import warnings
from src.model._thiswork.pca.incremental_pca import IncrementalPCA
from .base import AbstractMultilayerClustering
import numpy as np
from typing import List, Union
from torch import nn


class BMM(GMM):
    def __init__(self, *args, **kwargs):
        super(BMM, self).__init__(*args, **kwargs)
        self.register_buffer("degrees_of_freedom_", torch.ones(self.n_clusters))
        self.register_buffer(
            "covariance_prior_", torch.ones(self.n_input, self.n_input)
        )
        self.register_buffer("mean_precision_", torch.ones(self.n_clusters))
        self.register_buffer("mean_prior_", torch.ones(self.n_input))
        self.register_buffer("weight_concentration_0", torch.ones(self.n_clusters))
        self.register_buffer("weight_concentration_1", torch.ones(self.n_clusters))
        # Ref: sklearn.mixture.BayesianGaussianMixture._check_weights_parameters
        self.weight_concentration_prior_ = 1.0 / self.n_clusters
        # Ref: sklearn.mixture.BayesianGaussianMixture._check_means_parameters
        self.mean_precision_prior = 1.0
        # Ref: sklearn.mixture.BayesianGaussianMixture._check_precision_parameters
        self.degrees_of_freedom_prior_ = self.n_input

    def _estimate_log_prob(
        self, x: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ):
        # Ref: sklearn.mixture.BayesianGaussianMixture._estimate_log_prob
        log_gauss = self._estimate_log_gaussian_prob(
            x, means, covariances
        ) - 0.5 * self.n_input * torch.log(self.degrees_of_freedom_)
        log_lambda = self.n_input * torch.log(
            torch.tensor([2.0], device=x.device)
        ) + torch.sum(
            torch.special.digamma(
                0.5
                * (
                    self.degrees_of_freedom_
                    - torch.arange(0, self.n_input, device=x.device).unsqueeze(-1)
                )
            ),
            dim=0,
        )
        return log_gauss + 0.5 * (log_lambda - self.n_input / self.mean_precision_)

    def _estimate_log_weights(self):
        # Ref: sklearn.mixture.BayesianGaussianMixture._estimate_log_weights
        digamma_sum = torch.special.digamma(
            self.weight_concentration_0 + self.weight_concentration_1
        )
        digamma_a = torch.special.digamma(self.weight_concentration_0)
        digamma_b = torch.special.digamma(self.weight_concentration_1)
        device = self.weight_concentration_0.device
        return (
            digamma_a
            - digamma_sum
            + torch.hstack(
                (
                    torch.tensor([0], device=device),
                    torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1],
                )
            )
        )

    def _estimate_parameters(self, x: torch.Tensor, resp: torch.Tensor):
        # Ref: sklearn.mixture.BayesianGaussianMixture._initialize
        nk, xk, sk = self._estimate_gaussian_parameters(x, resp)
        # Ref: sklearn.mixture.BayesianGaussianMixture._estimate_weights
        self.weight_concentration_0, self.weight_concentration_1 = (
            1.0 + nk,
            self.weight_concentration_prior_
            + torch.hstack(
                (
                    torch.flip(
                        torch.cumsum(torch.flip(nk, dims=(0,)), dim=0), dims=(0,)
                    )[1:],
                    torch.tensor([0], device=x.device),
                )
            ),
        )
        # Ref: sklearn.mixture.BayesianGaussianMixture._set_parameters
        weight_dirichlet_sum = self.weight_concentration_0 + self.weight_concentration_1
        tmp = self.weight_concentration_1 / weight_dirichlet_sum
        weights = (
            self.weight_concentration_0
            / weight_dirichlet_sum
            * torch.hstack(
                (torch.tensor([1], device=x.device), torch.cumprod(tmp[:-1], dim=0))
            )
        )
        weights /= torch.sum(weights)
        # Ref: sklearn.mixture.BayesianGaussianMixture._estimate_means
        self.mean_precision_ = self.mean_precision_prior + nk
        self.mean_prior_ = torch.mean(x, dim=0)
        means = (
            self.mean_precision_prior * self.mean_prior_ + nk.unsqueeze(-1) * xk
        ) / self.mean_precision_.unsqueeze(-1)
        # Ref: sklearn.mixture.BayesianGaussianMixture._check_precision_parameters
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk
        # Ref: sklearn.mixture.BayesianGaussianMixture._checkcovariance_prior_parameter
        self.covariance_prior_ = torch.cov(x.T).reshape(
            self.n_input, self.n_input
        )  # To avoid empty cov.
        # Ref: sklearn.mixture.BayesianGaussianMixture._estimate_precisions (_estimate_wishart_full)
        covariances = torch.zeros(
            (self.n_clusters, self.n_input, self.n_input), device=x.device
        )
        for k in range(self.n_clusters):
            diff = xk[k] - self.mean_prior_
            covariances[k, :, :] = (
                self.covariance_prior_
                + nk[k] * sk[k, :, :]
                + nk[k]
                * self.mean_precision_prior
                / self.mean_precision_[k]
                * torch.outer(diff, diff)
            ) + self.eps * torch.eye(self.n_input, device=x.device)
        covariances /= self.degrees_of_freedom_.unsqueeze(-1).unsqueeze(-1)
        return weights, means, covariances


class PCABMM(BMM):
    def __init__(self, n_input, n_pca_dim: int = None, on_cpu: bool = True, **kwargs):
        if n_pca_dim is not None:
            if n_input <= n_pca_dim:
                msg = f"Expecting n_pca_dim lower than n_input {n_input}, but got {n_pca_dim}."
                if n_input < n_pca_dim:
                    raise Exception(msg)
                elif n_input == n_pca_dim:
                    print(msg)
                super(PCABMM, self).__init__(n_input=n_input, on_cpu=on_cpu, **kwargs)
            else:
                self.n_clustering_features = np.min([n_input, n_pca_dim])
                super(PCABMM, self).__init__(
                    n_input=self.n_clustering_features, on_cpu=on_cpu, **kwargs
                )
                self.pca = IncrementalPCA(
                    n_components=self.n_clustering_features, on_cpu=on_cpu
                )
        else:
            super(PCABMM, self).__init__(n_input=n_input, on_cpu=on_cpu, **kwargs)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "pca"):
            x = self.pca(x)
        return super(PCABMM, self).forward(x)


class FirstBMMCluster(Cluster):
    def __init__(
        self,
        n_input_outer: int,
        n_input_inner: int,
        exp_avg_factor: float = 1.0,
        **kwargs,
    ):
        super(FirstBMMCluster, self).__init__(
            n_input=n_input_outer, exp_avg_factor=exp_avg_factor
        )
        self.inner_layer = BMM(
            exp_avg_factor=exp_avg_factor, n_input=n_input_inner, **kwargs
        )


class TwolayerBMM(AbstractMultilayerClustering):
    def __init__(self, **kwargs):
        super(TwolayerBMM, self).__init__(
            algorithm_class=PCABMM,
            first_layer_cluster_class=FirstBMMCluster,
            **kwargs,
        )
