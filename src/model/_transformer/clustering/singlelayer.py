from .common.gmm import PCAGMM
from .common.kmeans import PCAKMeans
from .common.bmm import PCABMM
from .base import AbstractSNClustering


class GMMSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers, n_pca_dim: int = None):
        super(GMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=PCAGMM,
            n_pca_dim=n_pca_dim,
        )


class BMMSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers, n_pca_dim: int = None):
        super(BMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=PCABMM,
            n_pca_dim=n_pca_dim,
        )


class KMeansSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers, n_pca_dim: int = None):
        super(KMeansSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=PCAKMeans,
            n_pca_dim=n_pca_dim,
        )
