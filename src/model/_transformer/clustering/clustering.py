from torch import nn
import torch
from typing import List, Type


class AbstractCluster(nn.Module):
    def __init__(self, n_input: int, momentum: float = 0.8):
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
