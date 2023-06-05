from .common.gmm import TwolayerGMM
from .common.kmeans import TwolayerKMeans
from .common.bmm import TwolayerBMM
from .base import AbstractMultilayerSNClustering
from typing import List


class TwolayerGMMSN(AbstractMultilayerSNClustering):
    def __init__(self, **kwargs):
        super(TwolayerGMMSN, self).__init__(algorithm_class=TwolayerGMM, **kwargs)


class TwolayerBMMSN(AbstractMultilayerSNClustering):
    def __init__(self, **kwargs):
        super(TwolayerBMMSN, self).__init__(algorithm_class=TwolayerBMM, **kwargs)


class TwolayerKMeansSN(AbstractMultilayerSNClustering):
    def __init__(self, **kwargs):
        super(TwolayerKMeansSN, self).__init__(algorithm_class=TwolayerKMeans, **kwargs)
