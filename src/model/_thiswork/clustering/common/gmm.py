"""
This is a modification of sklearn.mixture.GaussianMixture for pytorch.
"""

import torch
from torch import nn
import numpy as np
from typing import List, Union
import warnings
from .kmeans import KMeans
from .base import AbstractCluster, AbstractClustering
from .base import (
    AbstractCluster,
    AbstractClustering,
    AbstractMultilayerClustering,
    AbstractSubspaceClustering,
)
from src.model._thiswork.pca.incremental_pca import IncrementalPCA


class Cluster(AbstractCluster):
    def __init__(self, n_input: int, exp_avg_factor: float = 1.0, **kwargs):
        super(Cluster, self).__init__(
            n_input=n_input, exp_avg_factor=exp_avg_factor, **kwargs
        )
        self.register_buffer("mu", torch.randn(1, n_input))
        self.register_buffer("var", torch.eye(n_input).unsqueeze(0))
        self.register_buffer(
            "pi",
            torch.ones(1),
        )

    def update(self, mu=None, var=None, pi=None, exp_avg_factor=None):
        exp_avg_factor = (
            self.exp_avg_factor if exp_avg_factor is None else exp_avg_factor
        )
        if mu is not None:
            self.mu = exp_avg_factor * mu + (1 - exp_avg_factor) * self.mu
        if var is not None:
            self.var = exp_avg_factor * var + (1 - exp_avg_factor) * self.var
        if pi is not None:
            self.pi = exp_avg_factor * pi + (1 - exp_avg_factor) * self.pi

    def set(self, mu=None, var=None, pi=None):
        if mu is not None:
            self.mu = mu
        if var is not None:
            self.var = var
        if pi is not None:
            self.pi = pi


class GMM(AbstractClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        clusters: Union[List[Cluster], nn.ModuleList] = None,
        exp_avg_factor: float = 1.0,
        init_method: str = "kmeans",
        **kwargs,
    ):
        super(GMM, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            cluster_class=Cluster,
            clusters=clusters,
            exp_avg_factor=exp_avg_factor,
            **kwargs,
        )
        self.init_method = init_method
        self.eps = 1e-6
        self.register_buffer(
            "accum_n_points_in_clusters",
            torch.ones(self.n_clusters, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor):
        device, x = self.to_cpu(x)
        if x.shape[0] < 2:
            return torch.zeros(x.shape[0], device=x.device).long()
        if not self.initialized and self.training:
            self.initialize(x)
        # Ref: sklearn.mixture.BaseMixture.fit_predict
        # Ref: sklearn.mixture.BaseMixture._estimate_weighted_log_porb
        # Calculate the probability that the ith sample belongs to the kth gaussian.
        # Note that the denominator in the bayesian theorem `log_prob_norm` (a constant) is not included.
        weighted_log_prob = (
            self._estimate_log_prob(x, self.mu, self.var) + self._estimate_log_weights()
        )
        x_cluster = torch.argmax(weighted_log_prob, dim=1).squeeze(-1)
        if self.training:
            with torch.no_grad():
                # E step
                # Ref: sklearn.mixture.BaseMixture._estimate_log_prob_resp
                # Calculate the denominator in the bayesian theorem
                log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
                # The log value of the probability that the ith sample belongs to the kth gaussian.
                log_resp = weighted_log_prob - log_prob_norm
                # M step
                # Ref: sklearn.mixture.GaussianMixture._m_step
                weights, means, covariances = self._estimate_parameters(
                    x, torch.exp(log_resp)
                )
                self.update_params(
                    weights=weights,
                    means=means,
                    covariances=covariances,
                    labels=x_cluster,
                )
        x_cluster = self.to_device(x_cluster, device)
        return x_cluster

    def initialize(self, x: torch.Tensor):
        if x.shape[0] < self.n_clusters:
            return None
        if self.init_method == "random":
            resp = torch.rand(x.shape[0], self.n_clusters, device=x.device)
            resp /= resp.sum(dim=1).unsqueeze(-1)
        elif self.init_method == "kmeans":
            kmeans = KMeans(
                n_clusters=self.n_clusters, n_input=self.n_input, on_cpu=self.on_cpu
            ).to(x.device)
            kmeans.fit(x, n_iter=10)
            # Initial parameters are estimated using kmeans results.
            labels = kmeans.predict(x)
            resp = torch.zeros((x.shape[0], self.n_clusters), device=x.device)
            resp[torch.arange(x.shape[0]), labels] = 1
        else:
            raise Exception(
                f"Initialization method {self.init_method} is not implemented."
            )
        self._base_resp = resp
        # Ref: sklearn.mixture.GaussianMixture._initialize
        # Initialization is a special case of maximization step since we assume that labels are already known and
        # the probabilities are ones.
        weights, means, covariances = self._estimate_parameters(x, resp)
        self.set_params(weights=weights, means=means, covariances=covariances)
        self.initialized = True

    def set_params(
        self, weights: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ):
        for i_cluster, cluster in enumerate(self.clusters):
            cluster.set(
                mu=means[i_cluster, :].unsqueeze(0),
                var=covariances[i_cluster, :, :].unsqueeze(0),
                pi=weights[i_cluster],
            )

    def update_params(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        labels=None,
    ):
        if self.adaptive_lr:
            matched_clusters, counts = labels.unique(return_counts=True)
            lr = 1 / self.accum_n_points_in_clusters[:, None] * 0.9 + 0.1
            self.accum_n_points_in_clusters[matched_clusters] += counts
        else:
            lr = self.exp_avg_factor * torch.ones(
                self.n_clusters, device=weights.device
            )
        for i_cluster, cluster in enumerate(self.clusters):
            cluster.update(
                mu=means[i_cluster, :].unsqueeze(0),
                var=covariances[i_cluster, :, :].unsqueeze(0),
                pi=weights[i_cluster],
                exp_avg_factor=lr[i_cluster],
            )

    @property
    def centers(self):
        return torch.concat([cluster.mu for cluster in self.clusters], dim=0)

    @property
    def mu(self):
        return self.centers

    @property
    def var(self):
        return torch.concat([cluster.var for cluster in self.clusters], 0)

    @property
    def pi(self):
        pi = torch.concat(
            [cluster.pi.unsqueeze(-1) for cluster in self.clusters], 0
        ).squeeze(-1)
        return pi / torch.sum(pi)

    def _compute_precision_cholesky(self, covariances: torch.Tensor):
        # Ref: sklearn.mixture._gaussian_mixture._compute_precision_cholesky
        dtype = covariances.dtype
        n_components, n_features = self.n_clusters, self.n_input
        # Note that cholesky_ex seems to be experimental.
        # https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html#torch.linalg.cholesky_ex
        # If using torch.linalg.cholesky, in some extreme cases, the covariances can be non-positive-definite
        # because self.eps is not large enough.
        # It seems to be possible to replace this by LDLT decomposition. But just leave it since it works fine.
        # https://github.com/pytorch/pytorch/issues/71382
        if n_components * n_features > 10:
            # This is the parallel version.
            cov_chol = torch.linalg.cholesky_ex(covariances.to(torch.float64)).L
            precisions_chol = (
                torch.linalg.solve_triangular(
                    cov_chol,
                    torch.eye(n_features, device=cov_chol.device).repeat(
                        n_components, 1, 1
                    ),
                    upper=False,
                )
                .permute(0, 2, 1)
                .to(dtype)
            )
        else:
            precisions_chol = torch.zeros_like(covariances, device=covariances.device)
            for k in range(n_components):
                cov_chol = torch.linalg.cholesky_ex(
                    covariances[k, :, :].to(torch.float64)
                ).L
                precisions_chol[k, :, :] = torch.linalg.solve_triangular(
                    cov_chol, torch.eye(n_features, device=cov_chol.device), upper=False
                ).T.to(dtype)
        return precisions_chol

    def _estimate_parameters(self, x: torch.Tensor, resp: torch.Tensor):
        weights, means, covariances = self._estimate_gaussian_parameters(x, resp)
        weights /= x.shape[0]
        return weights, means, covariances

    def _estimate_gaussian_parameters(self, x: torch.Tensor, resp: torch.Tensor):
        # Ref: sklearn.mixture._gaussian_mixture._estimate_gaussian_parameters
        nk = torch.add(torch.sum(resp, dim=0), 1e-12)
        means = torch.matmul(resp.T, x) / nk.unsqueeze(-1)
        # Ref: sklearn.mixture._gaussian_mixture._estimate_gaussian_covariances_full
        n_components, n_features = self.n_clusters, self.n_input
        covariances = torch.zeros(
            (n_components, n_features, n_features), device=x.device
        )
        for k in range(self.n_clusters):
            diff = x - means[k, :]
            covariances[k] = torch.matmul(resp[:, k] * diff.T, diff) / nk[
                k
            ] + self.eps * torch.eye(self.n_input, device=x.device)
        return nk, means, covariances

    def _estimate_log_prob(
        self, x: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ):
        return self._estimate_log_gaussian_prob(x, means, covariances)

    def _estimate_log_weights(self):
        return torch.log(self.pi)

    def _estimate_log_gaussian_prob(
        self, x: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ):
        # Ref: sklearn.mixture._gaussian_mixture._initialize
        precision_chol = self._compute_precision_cholesky(covariances)
        # Ref: sklearn.mixture._gaussian_mixture._estimate_log_gaussian_prob
        log_det = self._compute_log_det_cholesky(precision_chol)

        log_prob = torch.zeros(x.shape[0], self.n_clusters, device=x.device)
        for k in range(self.n_clusters):
            mu = means[k, :]
            prec_chol = precision_chol[k, :, :]
            y = torch.matmul(x, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        return -0.5 * (self.n_input * np.log(2 * np.pi) + log_prob) + log_det

    def _compute_log_det_cholesky(self, precision_chol: torch.Tensor):
        # Ref: sklearn.mixture._gaussian_mixture._compute_log_det_cholesky
        log_det_chol = torch.sum(
            torch.log(
                precision_chol.reshape(self.n_clusters, -1)[:, :: self.n_input + 1]
            ),
            dim=1,
        )
        return log_det_chol


class PCAGMM(GMM):
    def __init__(self, n_input, n_pca_dim: int = None, on_cpu: bool = True, **kwargs):
        if n_pca_dim is not None:
            if n_input <= n_pca_dim:
                msg = f"Expecting n_pca_dim lower than n_input {n_input}, but got {n_pca_dim}."
                if n_input < n_pca_dim:
                    raise Exception(msg)
                elif n_input == n_pca_dim:
                    print(msg)
                super(PCAGMM, self).__init__(n_input=n_input, on_cpu=on_cpu, **kwargs)
            else:
                self.n_clustering_features = np.min([n_input, n_pca_dim])
                super(PCAGMM, self).__init__(
                    n_input=self.n_clustering_features, on_cpu=on_cpu, **kwargs
                )
                self.pca = IncrementalPCA(
                    n_components=self.n_clustering_features, on_cpu=on_cpu
                )
        else:
            super(PCAGMM, self).__init__(n_input=n_input, on_cpu=on_cpu, **kwargs)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "pca"):
            x = self.pca(x)
        return super(PCAGMM, self).forward(x)


class FirstGMMCluster(Cluster):
    def __init__(
        self,
        n_input_outer: int,
        n_input_inner: int,
        exp_avg_factor: float = 1.0,
        **kwargs,
    ):
        super(FirstGMMCluster, self).__init__(
            n_input=n_input_outer, exp_avg_factor=exp_avg_factor
        )
        self.inner_layer = GMM(
            exp_avg_factor=exp_avg_factor, n_input=n_input_inner, **kwargs
        )


class TwolayerGMM(AbstractMultilayerClustering):
    def __init__(self, **kwargs):
        super(TwolayerGMM, self).__init__(
            algorithm_class=PCAGMM,
            first_layer_cluster_class=FirstGMMCluster,
            **kwargs,
        )


class MultilayerGMM(AbstractSubspaceClustering):
    def __init__(self, n_clusters_ls, **kwargs):
        super(MultilayerGMM, self).__init__(
            algorithm_classes=[PCAGMM] * len(n_clusters_ls),
            n_clusters_ls=n_clusters_ls,
            **kwargs,
        )
