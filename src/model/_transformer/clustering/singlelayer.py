from .common.gmm import PCAGMM
from .common.kmeans import PCAKMeans
from .common.bmm import PCABMM
from .base import AbstractSNClustering


class GMMSN(AbstractSNClustering):
    def __init__(self, **kwargs):
        super(GMMSN, self).__init__(algorithm_class=PCAGMM, **kwargs)


class BMMSN(AbstractSNClustering):
    def __init__(self, **kwargs):
        super(BMMSN, self).__init__(algorithm_class=PCABMM, **kwargs)


class KMeansSN(AbstractSNClustering):
    def __init__(self, **kwargs):
        super(KMeansSN, self).__init__(algorithm_class=PCAKMeans, **kwargs)
