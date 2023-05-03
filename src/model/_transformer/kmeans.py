# This script is a pytorch implementation of KMeans clustering. It is based on a simple and original pytorch version
# (https://github.com/subhadarship/kmeans_pytorch) and is enhanced by a version with high efficiency
# (https://github.com/DeMoriarty/fast_pytorch_kmeans/tree/master) with our own modifications.
# These two repositories follow MIT license.


import torch
from torch import nn
import numpy as np
from typing import List
import warnings


class Cluster(nn.Module):
    def __init__(self, n_input: int, momentum: float = 0.8):
        super(Cluster, self).__init__()
        self.register_buffer("center", torch.randn(1, n_input))
        self.momentum = momentum

    def update(self, new_center, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        self.center = momentum * new_center + (1 - momentum) * self.center

    def set(self, new_center):
        self.center = new_center


class KMeans(nn.Module):
    # https://github.com/subhadarship/kmeans_pytorch
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        clusters: List[Cluster] = None,
        momentum: float = 0.8,
    ):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.n_input = n_input
        self.clusters = nn.ModuleList(
            [Cluster(n_input=n_input, momentum=momentum) for i in range(n_clusters)]
            if clusters is None
            else clusters
        )
        self.method = "fast_kmeans"
        self.initialized = False
        self.initialize_method = "kmeans++"
        self.register_buffer(
            "accum_n_points_in_clusters",
            torch.ones(self.n_clusters, dtype=torch.float32),
        )

    def initialize(self, x: torch.Tensor, method="kmeans++"):
        if x.shape[0] < self.n_clusters:
            warnings.warn(
                f"The batch size {x.shape[0]} is smaller than the number of clusters {self.n_clusters}. Centers "
                f"of clusters are initialized randomly using torch.randn."
            )
        if method == "random":
            centers = x[torch.randperm(x.shape[0])[: self.n_clusters]]
        elif method == "kmeans++":
            # Reference:
            # https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/init_methods.py
            # In summary, this method calculates the distance value to the closest centroid for each data point, and
            # data points with higher values are more likely to be selected as the next centroid.
            centers = torch.zeros((self.n_clusters, x.shape[1])).to(x.device)
            r = torch.distributions.uniform.Uniform(0, 1)
            for i in range(self.n_clusters):
                if i == 0:
                    centers[i, :] = x[torch.randint(x.shape[0], [1])]
                else:
                    D2 = torch.cdist(centers[:i, :][None, :], x[None, :], p=2)[0].amin(
                        dim=0
                    )
                    # D2 is the distance to the closest centroid for each data.
                    probs = D2 / torch.sum(D2)
                    # probs is the weight vector for uniform-random sampling.
                    # The following is an implementation of weighted uniform-random sampling using pytorch.
                    cumprobs = torch.cumsum(probs, dim=0)
                    centers[i, :] = x[
                        torch.searchsorted(cumprobs, r.sample([1]).to(x.device))
                    ]
        else:
            raise Exception(f"Initialization method {method} is not implemented.")
        for i_cluster, cluster in enumerate(self.clusters):
            cluster.set(centers[i_cluster, :].view(1, -1))
        self.initialized = True

    def forward(self, x: torch.Tensor):
        x = x.float()
        if not self.initialized and self.training:
            self.initialize(x, method=self.initialize_method)
        dist = self.euclidean_pairwise_dist(x)
        x_cluster = torch.argmin(dist, dim=1)
        if self.training:
            with torch.no_grad():
                if self.method == "kmeans":
                    for i_cluster, cluster in enumerate(self.clusters):
                        x_in_cluster = x[x_cluster == i_cluster, :]
                        new_center = x_in_cluster.mean(dim=0)
                        if x_in_cluster.shape[0] != 0:
                            cluster.update(new_center)
                elif self.method == "fast_kmeans":
                    # Reference:
                    # https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
                    matched_clusters, counts = x_cluster.unique(return_counts=True)
                    mask = torch.zeros(
                        (self.n_clusters, len(x_cluster)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    mask[x_cluster, np.arange(len(x_cluster))] = 1.0
                    c_grad = mask @ x / counts.view(-1, 1).to(x.dtype)
                    c_grad[c_grad != c_grad] = 0  # remove NaNs
                    lr = 1 / self.accum_n_points_in_clusters[:, None] * 0.9 + 0.1
                    self.accum_n_points_in_clusters[matched_clusters] += counts
                    for i_cluster, cluster in enumerate(self.clusters):
                        cluster.update(c_grad[i_cluster, :], momentum=lr[i_cluster])
                else:
                    raise Exception(
                        f"KMeans implementation {self.method} is not implemented."
                    )
        return x_cluster

    @property
    def centers(self):
        return torch.concat([cluster.center for cluster in self.clusters], dim=0)

    def euclidean_pairwise_dist(self, x: torch.Tensor):
        dist = torch.pow((x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)), 2)
        dist = dist.sum(dim=-1)
        return dist

    def k_nearest_neighbors(self, k: int):
        if len(self.clusters) == 1:
            return torch.zeros(self.n_clusters, 0)
        if k >= self.n_clusters:
            raise Exception(
                f"The requested number of neighbors {k} is greater than the total number of "
                f"neighbors {self.n_clusters-1}."
            )
        dist = self.euclidean_pairwise_dist(self.centers)
        sort_dist = torch.argsort(dist, dim=-1)
        if k + 1 == self.n_clusters:
            return sort_dist[:, 1:]
        else:
            return sort_dist[:, 1 : k + 1]
