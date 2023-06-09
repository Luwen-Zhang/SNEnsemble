from .common.gmm import TwolayerGMM
from .common.kmeans import TwolayerKMeans
from .common.bmm import TwolayerBMM
from .base import AbstractSNClustering
from typing import List


class TwolayerGMMSN(AbstractSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
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
        )
        super(TwolayerGMMSN, self).__init__(clustering=clustering, **kwargs)


class TwolayerBMMSN(AbstractSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
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
        )
        super(TwolayerBMMSN, self).__init__(clustering=clustering, **kwargs)


class TwolayerKMeansSN(AbstractSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
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
        )
        super(TwolayerKMeansSN, self).__init__(clustering=clustering, **kwargs)
