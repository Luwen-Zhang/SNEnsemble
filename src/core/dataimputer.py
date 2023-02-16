from src.core.trainer import Trainer
import pandas as pd
from copy import deepcopy as cp
import numpy as np
import sys, inspect


class AbstractImputer:
    def __init__(self):
        self.record_features = None
        self.record_imputed_features = None

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def _get_impute_features(self, feature_names, data):
        all_missing_idx = np.where(np.isnan(data[feature_names].values).all(axis=0))[0]
        impute_features = [
            x for idx, x in enumerate(feature_names) if idx not in all_missing_idx
        ]
        self.record_imputed_features = impute_features
        return impute_features


class MiceImputer(AbstractImputer):
    def __init__(self):
        super(MiceImputer, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        import miceforest as mf

        impute_features = self._get_impute_features(trainer.feature_names, data)
        imputer = mf.ImputationKernel(data.loc[:, impute_features], random_state=0)
        imputer.mice(iterations=2, n_estimators=1)
        data.loc[:, impute_features] = imputer.complete_data().values.astype(np.float32)
        imputer.compile_candidate_preds()
        self.transformer = imputer
        self.record_features = cp(trainer.feature_names)
        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        trainer.feature_names = cp(self.record_features)
        data = input_data.copy()
        data.loc[:, self.record_imputed_features] = (
            self.transformer.impute_new_data(
                new_data=data.loc[:, self.record_imputed_features]
            )
            .complete_data()
            .values.astype(np.float32)
        )
        return data


class MeanImputer(AbstractImputer):
    def __init__(self):
        super(MeanImputer, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        impute_features = self._get_impute_features(trainer.feature_names, data)
        imputer = self._new_imputer()
        # https://github.com/scikit-learn/scikit-learn/issues/16426
        # SimpleImputer reduces the number of features without giving messages. The issue is fixed in
        # scikit-learn==1.2.0 by an argument "keep_empty_features"; however, autogluon==0.6.1 requires
        # scikit-learn<1.2.0.
        data.loc[:, impute_features] = imputer.fit_transform(
            data.loc[:, impute_features]
        ).astype(np.float32)

        self.transformer = imputer
        self.record_features = cp(trainer.feature_names)
        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        trainer.feature_names = cp(self.record_features)
        data = input_data.copy()
        data.loc[:, self.record_imputed_features] = self.transformer.transform(
            data.loc[:, self.record_imputed_features]
        ).astype(np.float32)
        return data

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="mean")


class MedianImputer(MeanImputer):
    def __init__(self):
        super(MedianImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="median")


class ModeImputer(MeanImputer):
    def __init__(self):
        super(ModeImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="most_frequent")


class NaNImputer(AbstractImputer):
    def __init__(self):
        super(NaNImputer, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        self.record_features = cp(trainer.feature_names)
        impute_features = self._get_impute_features(trainer.feature_names, data)
        return data.dropna(axis=0, subset=impute_features)

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        trainer.feature_names = cp(self.record_features)
        data = input_data.copy()
        return data.dropna(axis=0, subset=self.record_imputed_features)


imputer_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractImputer):
        imputer_mapping[name] = cls


def get_data_imputer(name: str):
    if name not in imputer_mapping.keys():
        raise Exception(f"Data imputer {name} not implemented.")
    elif not issubclass(imputer_mapping[name], AbstractImputer):
        raise Exception(f"{name} is not the subclass of AbstractImputer.")
    else:
        return imputer_mapping[name]
