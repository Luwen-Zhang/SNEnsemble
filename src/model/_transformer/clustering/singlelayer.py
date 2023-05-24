from .common.gmm import GMM
from .common.kmeans import KMeans
from .base import AbstractSNClustering


class GMMSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(GMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=GMM,
        )


class KMeansSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(KMeansSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=KMeans,
        )
