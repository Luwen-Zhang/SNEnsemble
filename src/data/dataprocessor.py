import warnings
from tabensemb.utils import *
from tabensemb.data import (
    AbstractProcessor,
    AbstractFeatureSelector,
    AbstractTransformer,
    AbstractScaler,
    AbstractAugmenter,
)
from tabensemb.data import DataModule
import inspect
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.preprocessing import OrdinalEncoder
from typing import Type
from tabensemb.data.utils import get_corr_sets


class LackDataMaterialRemover(AbstractProcessor):
    """
    Remove materials with fewer data (last 80%).
    """

    def __init__(self):
        super(LackDataMaterialRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        m_codes = data.loc[:, "Material_Code"].copy()
        m_cnts_index = list(m_codes.value_counts(ascending=False).index)
        self.lack_data_mat = m_cnts_index[len(m_cnts_index) // 10 * 8 :]
        for m_code in self.lack_data_mat:
            m_codes = data.loc[:, "Material_Code"].copy()
            where_material = m_codes.index[np.where(m_codes == m_code)[0]]
            data = data.drop(where_material)
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        if datamodule.training:
            for m_code in self.lack_data_mat:
                m_codes = data.loc[:, "Material_Code"].copy()
                where_material = m_codes.index[np.where(m_codes == m_code)[0]]
                data = data.drop(where_material)
        else:
            exist_m_codes = [
                m_code
                for m_code in self.lack_data_mat
                if m_code in data["Material_Code"]
            ]
            if len(exist_m_codes):
                warnings.warn(
                    f"{exist_m_codes} are removed by {self.__class__.__name__} but exist in the input dataset."
                )
        return data


class MaterialSelector(AbstractProcessor):
    """
    Select data with the specified material code. Required arguments:

    m_code: str
        The selected material code.
    """

    def __init__(self):
        super(MaterialSelector, self).__init__()

    def _fit_transform(
        self, data: pd.DataFrame, datamodule: DataModule, m_code=None, **kwargs
    ):
        if m_code is None:
            raise Exception('MaterialSelector requires the argument "m_code".')
        m_codes = datamodule.df.loc[np.array(data.index), "Material_Code"].copy()
        if m_code not in list(m_codes):
            raise Exception(f"m_code {m_code} not available.")
        where_material = m_codes.index[np.where(m_codes == m_code)[0]]
        data = data.loc[where_material, :]
        self.m_code = m_code
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        if datamodule.training:
            m_codes = data.loc[:, "Material_Code"].copy()
            if self.m_code not in list(m_codes):
                raise Exception(f"m_code {self.m_code} not available.")
            where_material = m_codes.index[np.where(m_codes == self.m_code)[0]]
            data = data.loc[where_material, :]
        else:
            if self.m_code not in data["Material_Code"]:
                warnings.warn(
                    f"{self.m_code} selected by {self.__class__.__name__} does not exist in the input dataset."
                )
        return data


mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractProcessor):
        mapping[name] = cls

tabensemb.data.dataprocessor.processor_mapping.update(mapping)
