from src.utils import *
from src.trainer import Trainer
from copy import deepcopy as cp


class AbstractDeriver:
    def __init__(self):
        pass

    def derive(
        self,
        df,
        trainer,
        derived_name,
        **kwargs,
    ):
        kwargs = self.make_defaults(**kwargs)
        for arg_name in self._required_cols(**kwargs):
            self._check_arg(arg_name, **kwargs)
            self._check_exist(df, arg_name, **kwargs)
        for arg_name in self._required_params(**kwargs) + ["stacked", "intermediate"]:
            self._check_arg(arg_name, **kwargs)
        values = self._derive(df, trainer, **kwargs)
        self._check_values(values)
        names = (
            self._generate_col_names(derived_name, values.shape[-1], **kwargs)
            if "col_names" not in kwargs
            else kwargs["col_names"]
        )
        return values, derived_name, names

    def make_defaults(self, **kwargs):
        for key, value in self._defaults().items():
            if key not in kwargs.keys():
                kwargs[key] = value
        return kwargs

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        raise NotImplementedError

    def _defaults(self):
        return {}

    def _derived_names(self, **kwargs):
        raise NotImplementedError

    def _generate_col_names(self, derived_name, length, **kwargs):
        try:
            names = self._derived_names(**kwargs)
        except:
            names = (
                [f"{derived_name}-{idx}" for idx in range(length)]
                if length > 1
                else [derived_name]
            )
        return names

    def _required_cols(self, **kwargs):
        raise NotImplementedError

    def _required_params(self, **kwargs):
        raise NotImplementedError

    def _check_arg(self, name, **kwargs):
        if name not in kwargs.keys():
            raise Exception(
                f"Derivation: {name} should be specified for deriver {self.__class__.__name__}"
            )

    def _check_exist(self, df, name, **kwargs):
        if kwargs[name] not in df.columns:
            raise Exception(
                f"Derivation: {name} is not a valid column in df for deriver {self.__class__.__name__}."
            )

    def _check_values(self, values):
        if len(values.shape) == 1:
            raise Exception(
                f"Derivation: {self.__class__.__name__} returns a one dimensional numpy.ndarray. Use reshape(-1, 1) to "
                f"transform into 2D."
            )


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


class AbstractProcessor:
    def __init__(self):
        self.record_cont_features = None
        self.record_cat_features = None

    def fit_transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        data = input_data.copy()
        res = self._fit_transform(data, trainer, **kwargs)
        self.record_cont_features = cp(trainer.cont_feature_names)
        self.record_cat_features = cp(trainer.cat_feature_names)
        return res

    def transform(self, input_data: pd.DataFrame, trainer: Trainer, **kwargs):
        trainer.cont_feature_names = cp(self.record_cont_features)
        trainer.cat_feature_names = cp(self.record_cat_features)
        data = input_data.copy()
        return self._transform(data, trainer, **kwargs)

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        raise NotImplementedError


class AbstractFeatureSelector(AbstractProcessor):
    def __init__(self):
        super(AbstractFeatureSelector, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        retain_features = list(self._get_feature_names_out(data, trainer, **kwargs))
        removed_features = list(
            np.setdiff1d(trainer.all_feature_names, retain_features)
        )
        if len(removed_features) > 0:
            trainer.cont_feature_names = [
                x for x in trainer.cont_feature_names if x in retain_features
            ]
            trainer.cat_feature_names = [
                x for x in trainer.cat_feature_names if x in retain_features
            ]
            print(
                f"{len(removed_features)} features removed: {removed_features}. {len(retain_features)} features retained: {retain_features}."
            )
        return data[
            trainer.cont_feature_names + trainer.cat_feature_names + trainer.label_name
        ]

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data[
            trainer.cont_feature_names + trainer.cat_feature_names + trainer.label_name
        ]

    def _get_feature_names_out(self, input_data, trainer, **kwargs):
        raise NotImplementedError


class AbstractTransformer(AbstractProcessor):
    def __init__(self):
        super(AbstractTransformer, self).__init__()
        self.transformer = None

    def zero_slip(self, feature_name, x):
        zero_data = pd.DataFrame(
            data=np.array(
                [
                    0 if feature_name != record_feature else x
                    for record_feature in self.record_cont_features
                ]
            ).reshape(1, -1),
            columns=self.record_cont_features,
        )
        try:
            trans_res = self.transformer.transform(zero_data)
        except:
            trans_res = zero_data.values
        return trans_res[0, self.record_cont_features.index(feature_name)]


class AbstractSplitter:
    def __init__(self, train_val_test=None):
        self.train_val_test = (
            np.array([0.6, 0.2, 0.2])
            if train_val_test is None
            else np.array(train_val_test)
        )
        self.train_val_test /= np.sum(self.train_val_test)

    def split(self, df, cont_feature_names, cat_feature_names, label_name):
        train_indices, val_indices, test_indices = self._split(
            df, cont_feature_names, cat_feature_names, label_name
        )
        self._check_split(train_indices, val_indices, test_indices)
        return train_indices, val_indices, test_indices

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        raise NotImplementedError

    def _check_split(self, train_indices, val_indices, test_indices):
        def individual_check(indices, name):
            if not issubclass(type(indices), np.ndarray):
                raise Exception(
                    f"The class of {name}_indices {type(indices)} is not the subclass of numpy.ndarray."
                )
            if len(indices.shape) != 1:
                raise Exception(
                    f"{name}_indices is not one dimensional. Use numpy.ndarray.flatten() to convert."
                )

        def intersect_check(a_indices, b_indices, a_name, b_name):
            if len(np.intersect1d(a_indices, b_indices)) != 0:
                raise Exception(
                    f"There exists intersection {np.intersect1d(a_indices, b_indices)} between {a_name}_indices "
                    f"and {b_name}_indices."
                )

        individual_check(train_indices, "train")
        individual_check(val_indices, "val")
        individual_check(test_indices, "test")

        intersect_check(train_indices, val_indices, "train", "val")
        intersect_check(train_indices, test_indices, "train", "test")
        intersect_check(val_indices, test_indices, "val", "test")

    def _check_exist(self, df, arg, name):
        if arg not in df.columns:
            raise Exception(
                f"Splitter: {name} is not a valid column in df for splitter {self.__class__.__name__}."
            )
