# This script is a pytorch implementation of KMeans clustering. It is based on a simple and original pytorch version
# (https://github.com/subhadarship/kmeans_pytorch) and is enhanced by a version with high efficiency
# (https://github.com/DeMoriarty/fast_pytorch_kmeans/tree/master) with our own modifications.
# These two repositories follow MIT license.


import torch
import numpy as np
from typing import List
import warnings
from .clustering import AbstractClustering, AbstractCluster


class Cluster(AbstractCluster):
    def __init__(self, n_input: int, momentum: float = 0.8):
        super(Cluster, self).__init__(n_input=n_input, momentum=momentum)
        self.register_buffer("center", torch.zeros(1, n_input))

    def update(self, new_center, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        self.center = momentum * new_center + (1 - momentum) * self.center

    def set(self, new_center):
        self.center = new_center


class KMeans(AbstractClustering):
    # https://github.com/subhadarship/kmeans_pytorch
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        clusters: List[Cluster] = None,
        momentum: float = 0.8,
        method: str = "fast_kmeans",
        init_method: str = "kmeans++",
    ):
        super(KMeans, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            cluster_class=Cluster,
            clusters=clusters,
            momentum=momentum,
        )
        self.method = method
        self.init_method = init_method
        self.register_buffer(
            "accum_n_points_in_clusters",
            torch.ones(self.n_clusters, dtype=torch.float32),
        )

    def initialize(self, x: torch.Tensor):
        if x.shape[0] < self.n_clusters:
            warnings.warn(
                f"The batch size {x.shape[0]} is smaller than the number of clusters {self.n_clusters}. Centers "
                f"of clusters are initialized randomly using torch.randn."
            )
        if self.init_method == "random":
            centers = x[torch.randperm(x.shape[0])[: self.n_clusters]]
        elif self.init_method == "kmeans++":
            # Reference:
            # https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/init_methods.py
            # In summary, this method calculates the distance value to the closest centroid for each data point, and
            # data points with higher values are more likely to be selected as the next centroid.
            x = x[
                torch.randint(
                    0, int(x.shape[0]), [min(100000, x.shape[0])], device=x.device
                )
            ]
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
                    sum_D2 = torch.sum(D2)
                    if sum_D2 == 0:
                        probs = torch.ones_like(D2, device=x.device) / len(D2)
                    else:
                        probs = D2 / sum_D2
                    # probs is the weight vector for uniform-random sampling.
                    # The following is an implementation of weighted uniform-random sampling using pytorch.
                    cumprobs = torch.cumsum(probs, dim=0)
                    centers[i, :] = x[
                        torch.searchsorted(cumprobs, r.sample([1]).to(x.device))
                    ]
        else:
            raise Exception(
                f"Initialization method {self.init_method} is not implemented."
            )
        for i_cluster, cluster in enumerate(self.clusters):
            cluster.set(centers[i_cluster, :].view(1, -1))
        self.initialized = True

    def forward(self, x: torch.Tensor):
        x = x.float()
        if not self.initialized and self.training:
            self.initialize(x)
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
                    c_grad = mask @ x / mask.sum(-1).view(-1, 1)
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
