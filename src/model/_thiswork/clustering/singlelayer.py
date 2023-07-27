from .common.gmm import PCAGMM
from .common.kmeans import PCAKMeans
from .common.bmm import PCABMM
from .base import AbstractPhyClustering


class GMMPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = PCAGMM(
            n_clusters=n_clusters, n_input=n_input, n_pca_dim=n_pca_dim, on_cpu=on_cpu
        )
        super(GMMPhy, self).__init__(clustering=clustering, **kwargs)


class BMMPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = PCABMM(
            n_clusters=n_clusters, n_input=n_input, n_pca_dim=n_pca_dim, on_cpu=on_cpu
        )
        super(BMMPhy, self).__init__(clustering=clustering, **kwargs)


class KMeansPhy(AbstractPhyClustering):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        n_pca_dim: int = None,
        on_cpu: bool = True,
        **kwargs
    ):
        clustering = PCAKMeans(
            n_clusters=n_clusters, n_input=n_input, n_pca_dim=n_pca_dim, on_cpu=on_cpu
        )
        super(KMeansPhy, self).__init__(clustering=clustering, **kwargs)
