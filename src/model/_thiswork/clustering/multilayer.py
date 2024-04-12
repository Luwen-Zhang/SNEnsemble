from .common.gmm import TwolayerGMM, MultilayerGMM
from .common.kmeans import TwolayerKMeans, MultilayerKMeans
from .common.bmm import TwolayerBMM, MultilayerBMM
from .base import AbstractPhyClustering
from typing import List


class TwolayerGMMPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = TwolayerGMM(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            shared_second_layer_clusters=True,
            on_cpu=on_cpu,
        )
        super(TwolayerGMMPhy, self).__init__(clustering=clustering, **kwargs)


class TwolayerBMMPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = TwolayerBMM(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            shared_second_layer_clusters=True,
            on_cpu=on_cpu,
        )
        super(TwolayerBMMPhy, self).__init__(clustering=clustering, **kwargs)


class TwolayerKMeansPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = TwolayerKMeans(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            shared_second_layer_clusters=True,
            on_cpu=on_cpu,
        )
        super(TwolayerKMeansPhy, self).__init__(clustering=clustering, **kwargs)


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
