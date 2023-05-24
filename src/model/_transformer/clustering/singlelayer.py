from .common.gmm import GMM
from .common.gmm import Cluster as _GMMCluster
from .common.kmeans import KMeans
from .common.kmeans import Cluster as _KMeansCluster
from .base import SN, AbstractSNClustering


class _GMMSNCluster(_GMMCluster):
    def __init__(self, n_input, layers, momentum, **kwargs):
        super(_GMMSNCluster, self).__init__(n_input, momentum=momentum, **kwargs)
        self.sn = SN(n_input, layers)


class GMMSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(GMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=GMM,
            cluster_class=_GMMSNCluster,
        )


class _KMeansSNCluster(_KMeansCluster):
    def __init__(self, n_input, layers, momentum, **kwargs):
        super(_KMeansSNCluster, self).__init__(n_input, momentum=momentum, **kwargs)
        self.sn = SN(n_input, layers)


class KMeansSN(AbstractSNClustering):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(KMeansSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=KMeans,
            cluster_class=_KMeansSNCluster,
        )
