from .common.gmm import TwolayerGMM
from .common.kmeans import TwolayerKMeans
from .common.bmm import TwolayerBMM
from .base import AbstractMultilayerSNClustering
from typing import List


class TwolayerGMMSN(AbstractMultilayerSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        layers,
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
    ):
        super(TwolayerGMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            algorithm_class=TwolayerGMM,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )


class TwolayerBMMSN(AbstractMultilayerSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        layers,
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
    ):
        super(TwolayerBMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            algorithm_class=TwolayerBMM,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )


class TwolayerKMeansSN(AbstractMultilayerSNClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        layers,
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
    ):
        super(TwolayerKMeansSN, self).__init__(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            algorithm_class=TwolayerKMeans,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )
