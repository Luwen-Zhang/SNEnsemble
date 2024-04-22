from torch import nn
import torch
from typing import List, Type, Union, Dict
import numpy as np
from copy import deepcopy as cp
from functools import reduce


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
        on_cpu: bool = True,
        **kwargs,
    ):
        super(AbstractClustering, self).__init__()
        self.n_total_clusters = n_clusters
        self.n_clusters = n_clusters
        self.n_input = n_input
        self.on_cpu = on_cpu
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

    def to_cpu(self, x):
        if self.on_cpu:
            self.to("cpu")
            return x.device, x.to("cpu")
        else:
            return x.device, x

    def to_device(self, x, device):
        if self.on_cpu:
            return x.to(device)
        else:
            return x


class AbstractSubspaceClustering(AbstractClustering):
    def __init__(
        self,
        n_clusters_ls: List[int],
        input_idxs: List[List[int]],
        algorithm_classes: List[Type[AbstractClustering]],
        kwargses: List[Dict],
        **kwargs,
    ):
        all_input_idxs = reduce(lambda x, y: np.union1d(x, y), input_idxs)
        super().__init__(
            n_clusters=reduce(lambda x, y: x * y, n_clusters_ls),
            n_input=len(all_input_idxs),
            cluster_class=None,
            clusters=[],
            **kwargs,
        )
        for kws in kwargses:
            kws.update(kwargs)
        self.input_idxs = input_idxs
        self.n_clusters_ls = n_clusters_ls
        self.algorithm_classes = algorithm_classes
        self.clusterings = nn.ModuleList(
            [
                algo(n_clusters=n_clus, n_input=len(idxs), **kws)
                for algo, idxs, n_clus, kws in zip(
                    algorithm_classes,
                    input_idxs,
                    n_clusters_ls,
                    kwargses,
                )
            ]
        )
        self.n_inputs = [len(x) for x in input_idxs]
        self._n_later_clusters = [
            reduce(lambda x, y: x * y, n_clusters_ls[i:])
            for i in range(len(input_idxs))
        ]
        self.res_each_layer = None

    def forward(self, x: torch.Tensor):
        device, x = self.to_cpu(x)
        res = []
        for i, (idxs, clus) in enumerate(zip(self.input_idxs, self.clusterings)):
            res.append(clus(x[:, idxs]))
        x_cluster = reduce(
            lambda x, y: (x[0] + y[0] * x[1], y[1]),
            [
                (res, n_later_clus)
                for res, n_later_clus in zip(res[::-1], self._n_later_clusters[::-1])
            ],
        )[0]
        self.res_each_layer = res
        x_cluster = self.to_device(x_cluster, device)
        return x_cluster
