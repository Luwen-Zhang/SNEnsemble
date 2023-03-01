import sklearn.exceptions
from src.trainer.trainer import Trainer
import pandas as pd
from copy import deepcopy as cp
import numpy as np
import sys, inspect


class AbstractImputer:
    def __init__(self):
        self.record_cont_features = None
        self.record_imputed_features = None

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        self.record_cont_features = cp(trainer.cont_feature_names)
        self.record_cat_features = cp(trainer.cat_feature_names)
        data.loc[:, self.record_cat_features] = data[self.record_cat_features].fillna(
            "UNK"
        )
        return self._fit_transform(data, trainer, **kwargs)

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        trainer.cont_feature_names = cp(self.record_cont_features)
        trainer.cat_feature_names = cp(self.record_cat_features)
        data.loc[:, self.record_cat_features] = data[self.record_cat_features].fillna(
            "UNK"
        )
        return self._transform(data, trainer, **kwargs)

    def _fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def _transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def _get_impute_features(self, cont_feature_names, data):
        all_missing_idx = np.where(
            np.isnan(data[cont_feature_names].values).all(axis=0)
        )[0]
        impute_features = [
            x for idx, x in enumerate(cont_feature_names) if idx not in all_missing_idx
        ]
        self.record_imputed_features = impute_features
        return impute_features


class NaNImputer(AbstractImputer):
    def __init__(self):
        super(NaNImputer, self).__init__()

    def _fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        impute_features = self._get_impute_features(
            trainer.cont_feature_names, input_data
        )
        return input_data.dropna(axis=0, subset=impute_features)

    def _transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        return input_data.dropna(axis=0, subset=self.record_imputed_features)


class MiceLightgbmImputer(AbstractImputer):
    def __init__(self):
        super(MiceLightgbmImputer, self).__init__()

    def _fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        import miceforest as mf

        impute_features = self._get_impute_features(
            trainer.cont_feature_names, input_data
        )
        imputer = mf.ImputationKernel(
            input_data.loc[:, impute_features], random_state=0
        )
        imputer.mice(iterations=2, n_estimators=1)
        input_data.loc[:, impute_features] = imputer.complete_data().values.astype(
            np.float32
        )
        imputer.compile_candidate_preds()
        self.transformer = imputer
        return input_data

    def _transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        input_data.loc[:, self.record_imputed_features] = (
            self.transformer.impute_new_data(
                new_data=input_data.loc[:, self.record_imputed_features]
            )
            .complete_data()
            .values.astype(np.float32)
        )
        return input_data


class AbstractSklearnImputer(AbstractImputer):
    def __init__(self):
        super(AbstractSklearnImputer, self).__init__()

    def _fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        impute_features = self._get_impute_features(
            trainer.cont_feature_names, input_data
        )
        imputer = self._new_imputer()
        # https://github.com/scikit-learn/scikit-learn/issues/16426
        # SimpleImputer reduces the number of features without giving messages. The issue is fixed in
        # scikit-learn==1.2.0 by an argument "keep_empty_features"; however, autogluon==0.6.1 requires
        # scikit-learn<1.2.0.
        res = imputer.fit_transform(input_data.loc[:, impute_features]).astype(
            np.float32
        )
        if type(res) == pd.DataFrame:
            res = res.values
        input_data.loc[:, impute_features] = res

        self.transformer = imputer
        return input_data

    def _transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        res = self.transformer.transform(
            input_data.loc[:, self.record_imputed_features]
        ).astype(np.float32)
        if type(res) == pd.DataFrame:
            res = res.values
        input_data.loc[:, self.record_imputed_features] = res
        return input_data

    def _new_imputer(self):
        raise NotImplementedError


class MiceImputer(AbstractSklearnImputer):
    def __init__(self):
        super(MiceImputer, self).__init__()

    def _new_imputer(self):
        # https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_sklearn_ice.py
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import warnings

        warnings.simplefilter(
            action="ignore", category=sklearn.exceptions.ConvergenceWarning
        )
        return IterativeImputer(
            random_state=0,
            max_iter=1000,
            tol=1e-3,
            sample_posterior=False,
        )


class MissForestImputer(AbstractSklearnImputer):
    def __init__(self):
        super(MissForestImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import warnings

        warnings.simplefilter(
            action="ignore", category=sklearn.exceptions.ConvergenceWarning
        )
        estimator_rf = RandomForestRegressor(
            n_estimators=1,
            max_depth=3,
            random_state=0,
            bootstrap=True,
            n_jobs=-1,
        )
        return IterativeImputer(estimator=estimator_rf, random_state=0, max_iter=10)


class GainImputer(AbstractSklearnImputer):
    def __init__(self):
        super(GainImputer, self).__init__()

    def _new_imputer(self):
        from src.utils.imputers.gain import GainImputation

        return GainImputation()


class MeanImputer(AbstractSklearnImputer):
    def __init__(self):
        super(MeanImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="mean")


class MedianImputer(AbstractSklearnImputer):
    def __init__(self):
        super(MedianImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="median")


class ModeImputer(AbstractSklearnImputer):
    def __init__(self):
        super(ModeImputer, self).__init__()

    def _new_imputer(self):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy="most_frequent")


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
