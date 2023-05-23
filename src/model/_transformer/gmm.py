"""
This is a modification of https://github.com/ldeecke/gmm-torch
We improve the initialization and stability.
"""
import torch
from torch import nn
import numpy as np
from typing import List
import warnings
from math import pi
from .kmeans import KMeans


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


class Cluster(nn.Module):
    def __init__(self, n_input: int, momentum: float = 0.8):
        super(Cluster, self).__init__()
        self.register_buffer("mu", torch.randn(1, n_input))
        self.register_buffer("var", torch.randn(1, n_input, n_input))
        self.register_buffer("pi", torch.zeros(1, 1))
        self.momentum = momentum

    def update(self, mu=None, var=None, pi=None, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        if mu is not None:
            self.mu = momentum * mu + (1 - momentum) * self.mu
        if var is not None:
            self.var = momentum * var + (1 - momentum) * self.var
        if pi is not None:
            self.pi = momentum * pi + (1 - momentum) * self.pi

    def set(self, mu=None, var=None, pi=None):
        if mu is not None:
            self.mu = mu
        if var is not None:
            self.var = var
        if pi is not None:
            self.pi = pi


class GMM(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        clusters: List[Cluster] = None,
        momentum: float = 0.8,
        init_method: str = "kmeans++",
    ):
        super(GMM, self).__init__()
        self.n_components = n_clusters
        self.n_features = n_input
        self.clusters = nn.ModuleList(
            [Cluster(n_input=n_input, momentum=momentum) for i in range(n_clusters)]
            if clusters is None
            else clusters
        )
        self.initialized = False
        self.init_method = init_method
        self.eps = 1e-6
        self.register_buffer(
            "accum_n_points_in_clusters",
            torch.ones(self.n_components, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        if not self.initialized and self.training:
            self.initialize(x)
        x = self.modify_size(x)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        x_cluster = torch.argmax(weighted_log_prob, dim=1).squeeze(-1)
        if self.training:
            with torch.no_grad():
                # E step
                log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
                log_resp = weighted_log_prob - log_prob_norm

                # M step
                resp = torch.exp(log_resp)

                pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
                mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

                eps = torch.eye(self.n_features, device=x.device) * self.eps
                var = (
                    torch.sum(
                        (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
                        * resp.unsqueeze(-1),
                        dim=0,
                        keepdim=True,
                    )
                    / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1)
                    + eps
                )
                pi = pi / x.shape[0]

                matched_clusters, counts = x_cluster.unique(return_counts=True)
                lr = 1 / self.accum_n_points_in_clusters[:, None] * 0.9 + 0.1
                self.accum_n_points_in_clusters[matched_clusters] += counts
                for i_cluster, cluster in enumerate(self.clusters):
                    cluster.update(
                        mu=mu[:, i_cluster, :].view(1, -1),
                        var=var[:, i_cluster, :, :],
                        pi=pi[:, i_cluster, :],
                        momentum=lr[i_cluster],
                    )
        return x_cluster

    def initialize(self, x: torch.Tensor):
        if x.shape[0] < self.n_components:
            warnings.warn(
                f"The batch size {x.shape[0]} is smaller than the number of clusters {self.n_components}. Centers "
                f"of clusters are initialized randomly using torch.randn."
            )
        pi = (
            torch.ones((1, 1, self.n_components), requires_grad=False, device=x.device)
            / self.n_components
        )
        var = (
            torch.eye(self.n_features, requires_grad=False, device=x.device)
            .reshape(1, 1, self.n_features, self.n_features)
            .repeat(1, self.n_components, 1, 1)
        )
        if self.init_method == "random":
            centers = x[torch.randperm(x.shape[0])[: self.n_components]]
        elif self.init_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_components, n_input=self.n_features).to(
                x.device
            )
            kmeans.fit(x, n_iter=10)
            centers = kmeans.centers
            # Initial parameters are estimated using kmeans results.
            # Reference: sklearn.mixture.gaussian_mixture._estimate_gaussian_covariances_full
            # Estimate weights.
            labels = kmeans.predict(x)
            _, counts = labels.unique(return_counts=True)
            pi[0, 0, :] = counts / x.shape[0]
            # Estimate variance.
            for k in range(self.n_components):
                resp = torch.zeros((x.shape[0],), device=x.device)
                resp[torch.where(labels == k)[0]] = 1
                diff = x - centers[k, :]
                var[0, k, :, :] = torch.matmul(resp * diff.t(), diff) / counts[
                    k
                ] + self.eps * torch.eye(self.n_features, device=x.device)
        else:
            raise Exception(
                f"Initialization method {self.init_method} is not implemented."
            )

        for i_cluster, cluster in enumerate(self.clusters):
            cluster.set(
                mu=centers[i_cluster, :].view(1, -1),
                var=var[:, i_cluster, :, :],
                pi=pi[:, :, i_cluster],
            )
        self.initialized = True

    @property
    def centers(self):
        return torch.concat([cluster.mu for cluster in self.clusters], dim=0)

    @property
    def mu(self):
        return self.centers.unsqueeze(0)

    @property
    def var(self):
        return torch.concat([cluster.var for cluster in self.clusters], 0).unsqueeze(0)

    @property
    def pi(self):
        return torch.concat([cluster.pi for cluster in self.clusters], 0).unsqueeze(0)

    def modify_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        mu = self.mu
        var = self.var
        x = self.modify_size(x)
        precision = torch.inverse(var)
        d = x.shape[-1]

        log_2pi = d * np.log(2.0 * pi)

        log_det = self._calculate_log_det(precision)

        x_mu_T = (x - mu).unsqueeze(-2)
        x_mu = (x - mu).unsqueeze(-1)

        x_mu_T_precision = calculate_matmul_n_times(
            self.n_components, x_mu_T, precision
        )
        x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

        return -0.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = (
                2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()
            )
        # nan_to_num ignores invalid precision (inverse of variance)
        log_det = torch.nan_to_num_(log_det, 0)
        return log_det.unsqueeze(-1)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def euclidean_pairwise_dist(self, x: torch.Tensor):
        dist = torch.pow((x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)), 2)
        dist = dist.sum(dim=-1)
        return dist

    def k_nearest_neighbors(self, k: int):
        if len(self.clusters) == 1:
            return torch.zeros(self.n_components, 0)
        if k >= self.n_components:
            raise Exception(
                f"The requested number of neighbors {k} is greater than the total number of "
                f"neighbors {self.n_components-1}."
            )
        dist = self.euclidean_pairwise_dist(self.centers)
        sort_dist = torch.argsort(dist, dim=-1)
        if k + 1 == self.n_components:
            return sort_dist[:, 1:]
        else:
            return sort_dist[:, 1 : k + 1]

    def check_size(self, x: torch.Tensor):
        if len(x.shape) != 2 or x.shape[-1] != self.n_input:
            raise Exception(
                f"Invalid input. Required shape: (n,{self.n_input}), got {x.shape} instead."
            )

    def fit(self, x: torch.Tensor, n_iter: int = 100):
        self.check_size(x)
        self.train()
        for i in range(n_iter):
            self(x)
        return self

    def predict(self, x: torch.Tensor):
        self.check_size(x)
        self.eval()
        return self(x)
