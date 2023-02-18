from src.core.trainer import Trainer
import pandas as pd
from copy import deepcopy as cp
import numpy as np
import sys, inspect


class AbstractProcessor:
    def __init__(self):
        self.record_features = None

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        res = self._fit_transform(data, trainer, **kwargs)
        self.record_features = cp(trainer.feature_names)
        return res

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        trainer.feature_names = cp(self.record_features)
        data = input_data.copy()
        return self._transform(data, trainer, **kwargs)

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError


class LackDataMaterialRemover(AbstractProcessor):
    def __init__(self):
        super(LackDataMaterialRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        m_codes = data.loc[:, "Material_Code"].copy()
        m_cnts_index = list(m_codes.value_counts(ascending=False).index)
        self.lack_data_mat = m_cnts_index[len(m_cnts_index) // 10 * 8 :]
        for m_code in self.lack_data_mat:
            m_codes = data.loc[:, "Material_Code"].copy()
            where_material = m_codes.index[np.where(m_codes == m_code)[0]]
            data = data.drop(where_material)
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        for m_code in self.lack_data_mat:
            m_codes = data.loc[:, "Material_Code"].copy()
            where_material = m_codes.index[np.where(m_codes == m_code)[0]]
            data = data.drop(where_material)
        return data


class MaterialSelector(AbstractProcessor):
    def __init__(self):
        super(MaterialSelector, self).__init__()

    def _fit_transform(
        self, data: pd.DataFrame, trainer: Trainer, m_code=None, **kwargs
    ):
        if m_code is None:
            raise Exception('MaterialSelector requires the argument "m_code".')
        m_codes = trainer.df.loc[np.array(data.index), "Material_Code"].copy()
        if m_code not in list(m_codes):
            raise Exception(f"m_code {m_code} not available.")
        where_material = m_codes.index[np.where(m_codes == m_code)[0]]
        data = data.loc[where_material, :]
        self.m_code = m_code
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        m_codes = data.loc[:, "Material_Code"].copy()
        if self.m_code not in list(m_codes):
            raise Exception(f"m_code {self.m_code} not available.")
        where_material = m_codes.index[np.where(m_codes == self.m_code)[0]]
        data = data.loc[where_material, :]
        return data


class FeatureValueSelector(AbstractProcessor):
    def __init__(self):
        super(FeatureValueSelector, self).__init__()

    def _fit_transform(
        self,
        data: pd.DataFrame,
        trainer: Trainer,
        feature=None,
        value=None,
        **kwargs,
    ):
        if feature is None or value is None:
            raise Exception(
                'FeatureValueSelector requires arguments "feature" and "value".'
            )
        if value not in list(data[feature]):
            raise Exception(
                f"Value {value} not available for feature {feature}. Select from {data[feature].unique()}"
            )
        where_value = data.index[np.where(data[feature] == value)[0]]
        data = data.loc[where_value, :]
        self.feature, self.value = feature, value
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        if self.value not in list(data[self.feature]):
            raise Exception(
                f"Value {self.value} not available for feature {self.feature}. Select from {data[self.feature].unique()}"
            )
        where_value = data.index[np.where(data[self.feature] == self.value)[0]]
        data = data.loc[where_value, :]
        return data


class IQRRemover(AbstractProcessor):
    def __init__(self):
        super(IQRRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        print(f"Removing outliers by IQR. Original size: {len(data)}, ", end="")
        for feature in list(trainer.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            Q1 = np.percentile(
                data[feature].dropna(axis=0), 25, interpolation="midpoint"
            )
            Q3 = np.percentile(
                data[feature].dropna(axis=0), 75, interpolation="midpoint"
            )
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = data.index[np.where(data[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = data.index[np.where(data[feature] <= (Q1 - 1.5 * IQR))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data


class StdRemover(AbstractProcessor):
    def __init__(self):
        super(StdRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        print(f"Removing outliers by std. Original size: {len(data)}, ", end="")
        for feature in list(trainer.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            m = np.mean(data[feature].dropna(axis=0))
            std = np.std(data[feature].dropna(axis=0))
            if std == 0:
                continue
            upper = data.index[np.where(data[feature] >= (m + 3 * std))[0]]
            lower = data.index[np.where(data[feature] <= (m - 3 * std))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data


class AbstractFeatureSelector(AbstractProcessor):
    def __init__(self):
        super(AbstractFeatureSelector, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        retain_features = list(self._get_feature_names_out(data, trainer))
        removed_features = list(np.setdiff1d(trainer.feature_names, retain_features))
        if len(removed_features) > 0:
            trainer.feature_names = retain_features
            print(
                f"{len(removed_features)} features removed: {removed_features}. {len(retain_features)} features retained: {retain_features}."
            )
        return data[retain_features + trainer.label_name]

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data

    def _get_feature_names_out(self, input_data, trainer):
        raise NotImplementedError


class SingleValueFeatureRemover(AbstractFeatureSelector):
    def __init__(self):
        super(SingleValueFeatureRemover, self).__init__()

    def _get_feature_names_out(self, data, trainer):
        retain_features = []
        for feature in trainer.feature_names:
            if len(np.unique(data[feature])) != 1:
                retain_features.append(feature)
        return retain_features


class NaNFeatureRemover(AbstractFeatureSelector):
    def __init__(self):
        super(NaNFeatureRemover, self).__init__()

    def _get_feature_names_out(self, data, trainer):
        retain_features = []
        all_missing_idx = np.where(
            np.isnan(data[trainer.feature_names].values).all(axis=0)
        )[0]
        for idx, feature in enumerate(trainer.feature_names):
            if idx not in all_missing_idx:
                retain_features.append(feature)
        return retain_features


class RFEFeatureSelector(AbstractFeatureSelector):
    def __init__(self):
        super(RFEFeatureSelector, self).__init__()

    def _get_feature_names_out(self, data, trainer):
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import KFold
        from sklearn.ensemble import RandomForestRegressor

        min_features_to_select = 1  # Minimum number of features to consider
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
        cv = KFold(5)

        rfecv = RFECV(
            estimator=rf,
            step=1,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=min_features_to_select,
            n_jobs=-1,
            verbose=0,
        )
        rfecv.fit(
            data[trainer.feature_names],
            data[trainer.label_name].values.flatten(),
        )
        retain_features = list(rfecv.get_feature_names_out())
        return retain_features


class VarianceFeatureSelector(AbstractFeatureSelector):
    def __init__(self):
        super(VarianceFeatureSelector, self).__init__()

    def _get_feature_names_out(self, data, trainer):
        from sklearn.feature_selection import VarianceThreshold

        sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
        sel.fit(
            data[trainer.feature_names],
            data[trainer.label_name].values.flatten(),
        )
        retain_features = list(sel.get_feature_names_out())
        return retain_features


class AbstractTransformer(AbstractProcessor):
    def __init__(self):
        super(AbstractTransformer, self).__init__()
        self.transformer = None

    def zero_slip(self, feature_name, x):
        trans_res = self.transformer.transform(
            pd.DataFrame(
                data=np.array(
                    [
                        0 if feature_name != record_feature else x
                        for record_feature in self.record_features
                    ]
                ).reshape(1, -1),
                columns=self.record_features,
            )
        )
        return trans_res[0, self.record_features.index(feature_name)]


class UnscaledDataRecorder(AbstractTransformer):
    def __init__(self):
        super(UnscaledDataRecorder, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        feature_data, label_data = trainer._divide_from_tabular_dataset(data)

        trainer.unscaled_feature_data = feature_data
        trainer.unscaled_label_data = label_data
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        feature_data, label_data = trainer._divide_from_tabular_dataset(data)
        trainer.unscaled_feature_data = feature_data
        trainer.unscaled_label_data = label_data
        return data


class StandardScaler(AbstractTransformer):
    def __init__(self):
        super(StandardScaler, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        from sklearn.preprocessing import StandardScaler as ss

        scaler = ss()
        # Indeed, scalers should fit on training dataset and transform the whole dataset. But due to the incorporation
        # of complex data processing pipline, we just fit and transform on the whole dataset. We do the same thing in
        # data-imputers.
        data.loc[:, trainer.feature_names] = scaler.fit_transform(
            data.loc[:, trainer.feature_names]
        ).astype(np.float32)

        self.transformer = scaler
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        data.loc[:, trainer.feature_names] = self.transformer.transform(
            data.loc[:, trainer.feature_names]
        ).astype(np.float32)
        return data


processor_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractProcessor):
        processor_mapping[name] = cls


def get_data_processor(name: str):
    if name not in processor_mapping.keys():
        raise Exception(f"Data processor {name} not implemented.")
    elif not issubclass(processor_mapping[name], AbstractProcessor):
        raise Exception(f"{name} is not the subclass of AbstractProcessor.")
    else:
        return processor_mapping[name]
