from .common.gmm import MultilayerGMM
from .common.kmeans import MultilayerKMeans
from .common.bmm import MultilayerBMM
from .base import AbstractPhyClustering
from typing import List


class MultilayerGMMPhy(AbstractPhyClustering):
    def __init__(
        self, n_clusters_ls, input_idxs, kwargses, on_cpu: bool = True, **kwargs
    ):
        clustering = MultilayerGMM(
            n_clusters_ls=n_clusters_ls,
            input_idxs=input_idxs,
            kwargses=kwargses,
            on_cpu=on_cpu,
        )
        super(MultilayerGMMPhy, self).__init__(clustering=clustering, **kwargs)


class MultilayerBMMPhy(AbstractPhyClustering):
    def __init__(
        self, n_clusters_ls, input_idxs, kwargses, on_cpu: bool = True, **kwargs
    ):
        clustering = MultilayerBMM(
            n_clusters_ls=n_clusters_ls,
            input_idxs=input_idxs,
            kwargses=kwargses,
            on_cpu=on_cpu,
        )
        super(MultilayerBMMPhy, self).__init__(clustering=clustering, **kwargs)


class MultilayerKMeansPhy(AbstractPhyClustering):
    def __init__(
        self, n_clusters_ls, input_idxs, kwargses, on_cpu: bool = True, **kwargs
    ):
        clustering = MultilayerKMeans(
            n_clusters_ls=n_clusters_ls,
            input_idxs=input_idxs,
            kwargses=kwargses,
            on_cpu=on_cpu,
        )
        super(MultilayerKMeansPhy, self).__init__(clustering=clustering, **kwargs)
