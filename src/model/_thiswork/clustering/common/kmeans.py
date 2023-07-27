# This script is a pytorch implementation of KMeans clustering. It is based on a simple and original pytorch version
# (https://github.com/subhadarship/kmeans_pytorch) and is enhanced by a version with high efficiency
# (https://github.com/DeMoriarty/fast_pytorch_kmeans/tree/master) with our own modifications.
# These two repositories follow MIT license.


import torch
from torch import nn
import numpy as np
from typing import List, Union
import warnings
from .base import AbstractClustering, AbstractCluster, AbstractMultilayerClustering
from src.model._thiswork.pca.incremental_pca import IncrementalPCA


class Cluster(AbstractCluster):
    def __init__(self, n_input: int, exp_avg_factor: float = 1.0, **kwargs):
        super(Cluster, self).__init__(
            n_input=n_input, exp_avg_factor=exp_avg_factor, **kwargs
        )
        self.register_buffer("center", torch.zeros(1, n_input))

    def update(self, new_center, exp_avg_factor=None):
        exp_avg_factor = (
            self.exp_avg_factor if exp_avg_factor is None else exp_avg_factor
        )
        self.center = exp_avg_factor * new_center + (1 - exp_avg_factor) * self.center

    def set(self, new_center):
        self.center = new_center


class KMeans(AbstractClustering):
    # https://github.com/subhadarship/kmeans_pytorch
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        clusters: Union[List[Cluster], nn.ModuleList] = None,
        exp_avg_factor: float = 1.0,
        method: str = "fast_kmeans",
        init_method: str = "kmeans++",
        n_init: int = 10,
        **kwargs,
    ):
        super(KMeans, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            cluster_class=Cluster,
            clusters=clusters,
            exp_avg_factor=exp_avg_factor,
            **kwargs,
        )
        self.method = method
        self.init_method = init_method
        self.n_init = n_init
        self.register_buffer(
            "accum_n_points_in_clusters",
            torch.ones(self.n_clusters, dtype=torch.float32),
        )

    def initialize(self, x: torch.Tensor):
        """
        This is not what sklearn does. sklearn runs the entire clustering multiple times to find the best results.
        But here only initialization is done instead of the entire fitting.
        """
        inertias = []
        for i_init in range(self.n_init):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(i_init)
            self._initialize_once(x, generator=generator)
            inertia = self._inertia(x)
            inertias.append(inertia.unsqueeze(0))
        best_i_init = torch.argmin(torch.concat(inertias)).squeeze().item()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(best_i_init)
        self._initialize_once(x, generator=generator)
        self.initialized = True

    def _initialize_once(self, x: torch.Tensor, generator):
        if x.shape[0] < self.n_clusters:
            return None
        if self.init_method == "random":
            centers = x[
                torch.randperm(x.shape[0], generator=generator)[: self.n_clusters]
            ]
        elif self.init_method == "kmeans++":
            # Reference:
            # https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/init_methods.py
            # In summary, this method calculates the distance value to the closest centroid for each data point, and
            # data points with higher values are more likely to be selected as the next centroid.
            x = x[
                torch.randint(
                    0, int(x.shape[0]), [min(100000, x.shape[0])], generator=generator
                )
            ]
            centers = torch.zeros((self.n_clusters, x.shape[1]), device=x.device)
            for i in range(self.n_clusters):
                if i == 0:
                    centers[i, :] = x[
                        torch.randint(x.shape[0], [1], generator=generator)
                    ]
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
                        torch.searchsorted(
                            cumprobs,
                            torch.rand([1], generator=generator).to(x.device),
                        )
                    ]
        else:
            raise Exception(
                f"Initialization method {self.init_method} is not implemented."
            )
        for i_cluster, cluster in enumerate(self.clusters):
            cluster.set(centers[i_cluster, :].view(1, -1))

    def _predict(self, x):
        dist = self.euclidean_pairwise_dist(x)
        x_cluster = torch.argmin(dist, dim=1)
        return x_cluster

    def _inertia(self, x):
        x_cluster = self._predict(x)
        inertia = torch.sum(torch.pow(self.centers[x_cluster] - x, 2))
        return inertia

    def forward(self, x: torch.Tensor):
        device, x = self.to_cpu(x)
        if not self.initialized and self.training:
            self.initialize(x)
        x_cluster = self._predict(x)
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
                        cluster.update(
                            c_grad[i_cluster, :], exp_avg_factor=lr[i_cluster]
                        )
                else:
                    raise Exception(
                        f"KMeans implementation {self.method} is not implemented."
                    )
        x_cluster = self.to_device(x_cluster, device)
        return x_cluster

    @property
    def centers(self):
        return torch.concat([cluster.center for cluster in self.clusters], dim=0)


class PCAKMeans(KMeans):
    def __init__(self, n_input, n_pca_dim: int = None, on_cpu: bool = True, **kwargs):
        if n_pca_dim is not None:
            if n_input <= n_pca_dim:
                msg = f"Expecting n_pca_dim lower than n_input {n_input}, but got {n_pca_dim}."
                if n_input < n_pca_dim:
                    raise Exception(msg)
                elif n_input == n_pca_dim:
                    print(msg)
                super(PCAKMeans, self).__init__(
                    n_input=n_input, on_cpu=on_cpu, **kwargs
                )
            else:
                self.n_clustering_features = np.min([n_input, n_pca_dim])
                super(PCAKMeans, self).__init__(
                    n_input=self.n_clustering_features, on_cpu=on_cpu, **kwargs
                )
                self.pca = IncrementalPCA(
                    n_components=self.n_clustering_features, on_cpu=on_cpu
                )
        else:
            super(PCAKMeans, self).__init__(n_input=n_input, on_cpu=on_cpu, **kwargs)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "pca"):
            x = self.pca(x)
        return super(PCAKMeans, self).forward(x)


class FirstKMeansCluster(Cluster):
    def __init__(
        self,
        n_input_outer: int,
        n_input_inner: int,
        exp_avg_factor: float = 1.0,
        **kwargs,
    ):
        super(FirstKMeansCluster, self).__init__(
            n_input=n_input_outer, exp_avg_factor=exp_avg_factor
        )
        self.inner_layer = KMeans(
            exp_avg_factor=exp_avg_factor, n_input=n_input_inner, **kwargs
        )


class TwolayerKMeans(AbstractMultilayerClustering):
    def __init__(self, **kwargs):
        super(TwolayerKMeans, self).__init__(
            algorithm_class=PCAKMeans,
            first_layer_cluster_class=FirstKMeansCluster,
            **kwargs,
        )
