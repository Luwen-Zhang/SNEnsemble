from torch import nn
import torch
from typing import List, Type
import numpy as np


class AbstractCluster(nn.Module):
    def __init__(self, n_input: int, momentum: float = 0.8, **kwargs):
        super(AbstractCluster, self).__init__()
        self.momentum = momentum


class AbstractClustering(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        cluster_class: Type[AbstractCluster] = None,
        clusters: List[AbstractCluster] = None,
        momentum: float = 0.8,
    ):
        super(AbstractClustering, self).__init__()
        self.n_clusters = n_clusters
        self.n_input = n_input
        if clusters is None and cluster_class is None:
            raise Exception(f"Neither `cluster_class` nor `clusters` is provided.")
        self.clusters = nn.ModuleList(
            [
                cluster_class(n_input=n_input, momentum=momentum)
                for i in range(n_clusters)
            ]
            if clusters is None
            else clusters
        )
        self.initialized = False

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

    def check_size(self, x: torch.Tensor):
        if len(x.shape) != 2 or x.shape[-1] != self.n_input:
            raise Exception(
                f"Invalid input. Required shape: (n,{self.n_input}), got {x.shape} instead."
            )

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

    def euclidean_pairwise_dist(self, x: torch.Tensor):
        dist = torch.pow((x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)), 2)
        dist = dist.sum(dim=-1)
        return dist

    @property
    def centers(self):
        raise NotImplementedError


class AbstractMultilayerClustering(AbstractClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        algorithm_class: Type[AbstractClustering],
        second_layer_cluster_class: Type[AbstractCluster],
        clusters: List[AbstractCluster] = None,
        momentum: float = 0.8,
        n_clusters_per_cluster: int = 5,
        **kwargs,
    ):
        if (
            clusters is not None
            and len(clusters) != n_clusters * n_clusters_per_cluster
        ):
            raise Exception(
                f"{n_clusters * n_clusters_per_cluster} clusters is required. Got {len(clusters)} instead."
            )
        second_layer_clusters = [
            second_layer_cluster_class(
                n_input=n_input_2,
                momentum=momentum,
                n_clusters=n_clusters_per_cluster,
                clusters=clusters[
                    i * n_clusters_per_cluster : (i + 1) * n_clusters_per_cluster
                ]
                if clusters is not None
                else None,
                **kwargs,
            )
            for i in range(n_clusters)
        ]
        self.n_total_clusters = n_clusters * n_clusters_per_cluster
        self.n_clusters_per_cluster = n_clusters_per_cluster
        self.input_1_idx = input_1_idx
        self.input_2_idx = input_2_idx
        super().__init__(
            n_clusters=self.n_total_clusters,
            n_input=len(np.union1d(input_1_idx, input_2_idx)),
            momentum=momentum,
            clusters=[],
        )
        self.first_clustering = algorithm_class(
            n_clusters=n_clusters,
            n_input=n_input_1,
            momentum=momentum,
            clusters=second_layer_clusters,
            **kwargs,
        )
        # This is a surrogate instance to gather clusters in second_layer_clusters.
        self.second_clustering = algorithm_class(
            n_clusters=self.n_total_clusters,
            n_input=len(np.union1d(input_1_idx, input_2_idx)),
            momentum=momentum,
            clusters=clusters,
        )

    def forward(self, x: torch.Tensor):
        outer_cluster = self.first_clustering(x[:, self.input_1_idx])
        x_cluster = torch.zeros((x.shape[0],), device=x.device).long()
        for i in range(self.first_clustering.n_clusters):
            where_in_cluster = torch.where(outer_cluster == i)[0]
            inner_cluster = (
                self.first_clustering.clusters[i].inner_layer(
                    x[where_in_cluster, :][:, self.input_2_idx]
                )
                + i * self.n_clusters_per_cluster
            )
            x_cluster[where_in_cluster] = inner_cluster
        return x_cluster
