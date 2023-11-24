from .common.kmeans import PCAKMeans
from .base import AbstractPhyClustering


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
