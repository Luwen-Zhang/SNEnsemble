from torch import nn
import torch
from typing import List, Type, Union
import numpy as np
from copy import deepcopy as cp


class AbstractCluster(nn.Module):
    def __init__(self, n_input: int, exp_avg_factor: float = 1.0, **kwargs):
        super(AbstractCluster, self).__init__()
        self.n_input = n_input
        self.exp_avg_factor = exp_avg_factor


class AbstractClustering(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        cluster_class: Type[AbstractCluster] = None,
        clusters: Union[List[AbstractCluster], nn.ModuleList] = None,
        exp_avg_factor: float = 1.0,
        adaptive_lr: bool = False,
        **kwargs,
    ):
        super(AbstractClustering, self).__init__()
        self.n_total_clusters = n_clusters
        self.n_clusters = n_clusters
        self.n_input = n_input
        if clusters is None and cluster_class is None:
            raise Exception(f"Neither `cluster_class` nor `clusters` is provided.")
        if clusters is not None and len(clusters) > 0:
            if clusters[0].n_input != n_input:
                raise Exception(
                    f"Given clusters have n_input={clusters[0].n_input}, but the n_input of {self.__class__.__name__} "
                    f"is set to {n_input}"
                )
        self.clusters = nn.ModuleList(
            [
                cluster_class(n_input=n_input, exp_avg_factor=exp_avg_factor)
                for i in range(n_clusters)
            ]
            if clusters is None
            else clusters
        )
        self.adaptive_lr = adaptive_lr
        self.exp_avg_factor = exp_avg_factor
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
        algorithm_class: Type[AbstractClustering] = None,
        first_layer_cluster_class: Type[AbstractCluster] = None,
        exp_avg_factor: float = 1.0,
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
        shared_second_layer_clusters: bool = False,
        **kwargs,
    ):
        if algorithm_class is None or first_layer_cluster_class is None:
            raise Exception(
                f"Classes inherited from AbstractMultilayerClustering should provide `algorithm_class` and "
                f"`first_layer_cluster_class`."
            )
        self.shared_second_layer_clusters = shared_second_layer_clusters
        base_first_layer_cluster = first_layer_cluster_class(
            n_input_outer=n_input_1 if n_pca_dim is None else n_pca_dim,
            n_input_inner=n_input_2,
            exp_avg_factor=exp_avg_factor,
            n_clusters=n_clusters_per_cluster,
            **kwargs,
        )
        if shared_second_layer_clusters:

            def _map_inner_clusters(cluster):
                out = cp(cluster)
                out.inner_layer = cluster.inner_layer
                return out

            duplicate_fn = _map_inner_clusters
        else:
            duplicate_fn = cp
        first_layer_clusters = [
            duplicate_fn(base_first_layer_cluster) for i in range(n_clusters)
        ]
        self.n_total_clusters = n_clusters * n_clusters_per_cluster
        self.n_clusters_per_cluster = n_clusters_per_cluster
        self.input_1_idx = input_1_idx
        self.input_2_idx = input_2_idx
        super().__init__(
            n_clusters=self.n_total_clusters,
            n_input=len(np.union1d(input_1_idx, input_2_idx)),
            exp_avg_factor=exp_avg_factor,
            clusters=[],
        )
        self.first_clustering = algorithm_class(
            n_clusters=n_clusters,
            n_input=n_input_1,
            exp_avg_factor=exp_avg_factor,
            clusters=first_layer_clusters,
            n_pca_dim=n_pca_dim,
            **kwargs,
        )
        # This is a surrogate instance to gather clusters in second_layer_clusters.
        if self.shared_second_layer_clusters:
            self.second_clustering = algorithm_class(
                n_clusters=self.n_clusters_per_cluster,
                n_input=n_input_2,
                exp_avg_factor=exp_avg_factor,
                clusters=first_layer_clusters[0].inner_layer.clusters,
                **kwargs,
            )
        else:
            inner_clusters = nn.ModuleList()
            for i in first_layer_clusters:
                inner_clusters += i.inner_layer.clusters
            self.second_clustering = algorithm_class(
                n_clusters=self.n_total_clusters,
                n_input=n_input_2,
                exp_avg_factor=exp_avg_factor,
                clusters=inner_clusters,
                **kwargs,
            )

    def forward(self, x: torch.Tensor):
        outer_cluster = self.first_clustering(x[:, self.input_1_idx])
        if not self.shared_second_layer_clusters:
            x_cluster = torch.zeros((x.shape[0],), device=x.device).long()
            for i in range(self.first_clustering.n_clusters):
                where_in_cluster = torch.where(outer_cluster == i)[0]
                if len(where_in_cluster) > 0:
                    inner_cluster = (
                        self.first_clustering.clusters[i].inner_layer(
                            x[where_in_cluster, :][:, self.input_2_idx]
                        )
                        + i * self.n_clusters_per_cluster
                    )
                    x_cluster[where_in_cluster] = inner_cluster
        else:
            inner_cluster = self.second_clustering(x[:, self.input_2_idx])
            x_cluster = inner_cluster + outer_cluster * self.n_clusters_per_cluster
        return x_cluster
