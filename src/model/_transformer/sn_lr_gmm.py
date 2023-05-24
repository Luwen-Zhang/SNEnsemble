from .gmm import Cluster, GMM
from .sn_lr_cluster import SN, AbstractSNCluster


class _SNCluster(Cluster):
    def __init__(self, n_input, layers, momentum):
        super(_SNCluster, self).__init__(n_input, momentum=momentum)
        self.sn = SN(n_input, layers)


class GMMSN(AbstractSNCluster):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(GMMSN, self).__init__(
            n_clusters=n_clusters,
            n_input=n_input,
            layers=layers,
            algorithm_class=GMM,
            cluster_class=_SNCluster,
        )
