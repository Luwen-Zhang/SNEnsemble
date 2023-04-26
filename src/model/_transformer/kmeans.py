import torch
from torch import nn
import numpy as np
from typing import List


class Cluster(nn.Module):
    def __init__(self, n_input: int, momentum: float = 0.8):
        super(Cluster, self).__init__()
        self.register_buffer("center", torch.randn(1, n_input))
        self.momentum = momentum

    def update(self, new_center):
        self.center = self.momentum * new_center + (1 - self.momentum) * self.center


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
        self.clusters = nn.ModuleList(
            [Cluster(n_input=n_input, momentum=momentum) for i in range(n_clusters)]
            if clusters is None
            else clusters
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        dist = self.euclidean_pairwise_dist(x)
        x_cluster = torch.argmin(dist, dim=1)
        if self.training:
            with torch.no_grad():
                for i_cluster, cluster in enumerate(self.clusters):
                    x_in_cluster = x[x_cluster == i_cluster, :]
                    if x_in_cluster.shape[0] != 0:
                        cluster.update(x_in_cluster.mean(dim=0))
        return x_cluster

    @property
    def centers(self):
        return torch.concat([cluster.center for cluster in self.clusters], dim=0)

    def euclidean_pairwise_dist(self, x: torch.Tensor):
        dist = torch.pow((x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)), 2)
        dist = dist.sum(dim=-1)
        return dist
