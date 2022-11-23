from src.core.trainer import Trainer
import pandas as pd
from copy import deepcopy as cp
import numpy as np


class AbstractProcessor:
    def __init__(self):
        pass

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        raise NotImplementedError

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        raise NotImplementedError


class IQRRemover(AbstractProcessor):
    def __init__(self):
        super(IQRRemover, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        print(f'Removing outliers by IQR. Original size: {len(input_data)}')
        data = input_data.copy()
        for feature in trainer.feature_names:
            Q1 = np.percentile(data[feature].dropna(axis=0), 25, interpolation='midpoint')
            Q3 = np.percentile(data[feature].dropna(axis=0), 75, interpolation='midpoint')
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = data.index[np.where(data[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = data.index[np.where(data[feature] <= (Q1 - 1.5 * IQR))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        return input_data.copy()


class StdRemover(AbstractProcessor):
    def __init__(self):
        super(StdRemover, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        print(f'Removing outliers by std. Original size: {len(input_data)}')
        data = input_data.copy()
        for feature in trainer.feature_names:
            m = np.mean(data[feature].dropna(axis=0))
            std = np.std(data[feature].dropna(axis=0))
            if std == 0:
                continue
            upper = data.index[np.where(data[feature] >= (m + 3 * std))[0]]
            lower = data.index[np.where(data[feature] <= (m - 3 * std))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        return input_data.copy()


class SingleValueFeatureRemover(AbstractProcessor):
    def __init__(self):
        super(SingleValueFeatureRemover, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        data = input_data.copy()
        retain_features = []
        removed_features = []
        for feature in trainer.feature_names:
            if len(np.unique(data[feature])) == 1:
                removed_features.append(feature)
            else:
                retain_features.append(feature)

        if len(removed_features) > 0:
            trainer.feature_names = retain_features
            print(f'{len(removed_features)} features removed: {removed_features}. {len(retain_features)} features retained: {retain_features}.')

        return data[retain_features + trainer.label_name]


class UnscaledDataRecorder(AbstractProcessor):
    def __init__(self):
        super(UnscaledDataRecorder, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        feature_data, label_data = trainer._divide_from_tabular_dataset(input_data)

        trainer.unscaled_feature_data = feature_data
        trainer.unscaled_label_data = label_data

        return input_data.copy()

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        pass


class AbstractTransformer(AbstractProcessor):
    def __init__(self):
        super(AbstractTransformer, self).__init__()
        self.transformer = None


class MeanImputer(AbstractTransformer):
    def __init__(self):
        super(MeanImputer, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        data = input_data.copy()
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        fit_indices = np.intersect1d(np.array(list(trainer.train_indices) + list(trainer.val_indices)),
                                     np.array(data.index))
        trans_indices = np.setdiff1d(np.array(data.index), fit_indices)
        data.loc[fit_indices, trainer.feature_names] = imputer.fit_transform(
            data.loc[fit_indices, trainer.feature_names]).astype(np.float32)
        data.loc[trans_indices, trainer.feature_names] = imputer.transform(
            data.loc[trans_indices, trainer.feature_names]).astype(np.float32)

        self.transformer = imputer

        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        return pd.DataFrame(data=self.transformer.transform(input_data[trainer.feature_names]),
                            columns=trainer.feature_names).astype(np.float32)


class NaNImputer(AbstractTransformer):
    def __init__(self):
        super(NaNImputer, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        data = input_data.copy()
        return data.dropna(axis=0, subset=trainer.feature_names)

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        data = input_data.copy()
        return data.dropna(axis=0, subset=trainer.feature_names)


class StandardScaler(AbstractTransformer):
    def __init__(self):
        super(StandardScaler, self).__init__()

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer):
        data = input_data.copy()
        from sklearn.preprocessing import StandardScaler as ss
        scaler = ss()
        fit_indices = np.intersect1d(np.array(list(trainer.train_indices) + list(trainer.val_indices)),
                                     np.array(data.index))
        trans_indices = np.setdiff1d(np.array(data.index), fit_indices)
        data.loc[fit_indices, trainer.feature_names] = scaler.fit_transform(
            data.loc[fit_indices, trainer.feature_names]).astype(np.float32)
        data.loc[trans_indices, trainer.feature_names] = scaler.transform(
            data.loc[trans_indices, trainer.feature_names]).astype(np.float32)

        self.transformer = scaler

        return data

    def transform(self, input_data: pd.DataFrame, trainer: Trainer):
        return pd.DataFrame(data=self.transformer.transform(input_data[trainer.feature_names]),
                            columns=trainer.feature_names).astype(np.float32)


processor_mapping = {
    'IQRRemover': IQRRemover(),
    'StdRemover': StdRemover(),
    'UnscaledDataRecorder': UnscaledDataRecorder(),
    'MeanImputer': MeanImputer(),
    'NaNImputer': NaNImputer(),
    'StandardScaler': StandardScaler(),
    'SingleValueFeatureRemover': SingleValueFeatureRemover(),
}


def get_data_processor(name: str):
    if name not in processor_mapping.keys():
        raise Exception(f'Data processor {name} not implemented or added to dataprocessor.processor_mapping.')
    elif not issubclass(type(processor_mapping[name]), AbstractProcessor):
        raise Exception(f'{name} is not the subclass of AbstractProcessor.')
    else:
        return processor_mapping[name]
