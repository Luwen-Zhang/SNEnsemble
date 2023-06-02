import os.path
import src
from src.utils import *
from src.config import UserConfig
from copy import deepcopy as cp
from typing import *
import torch.cuda
import torch.utils.data as Data
from torch.utils.data import Subset
from sklearn.decomposition import PCA
import sklearn.pipeline
import sklearn.ensemble


class DataModule:
    def __init__(
        self,
        config: Union[UserConfig, Dict],
        verbose: bool = True,
    ):
        self.args = config
        self.set_data_splitter(self.args["data_splitter"], verbose=verbose)
        self.set_data_imputer(name=self.args["data_imputer"], verbose=verbose)
        self.set_data_processors(self.args["data_processors"], verbose=verbose)
        self.set_data_derivers(self.args["data_derivers"], verbose=verbose)
        self.training = False
        self.data_path = None

    def set_status(self, training: bool):
        """
        Set the status of the datamodule. If a datamodule is not training, some data derivers will use learned characteristics
        from training data to derive for new data.

        Parameters
        ----------
        training
            The training status of the datamodule.
        """
        self.training = training

    def set_data_splitter(self, config: Union[str, Tuple[str, Dict]], verbose=True):
        """
        Set the data splitter. The specified splitter should be implemented in ``data/datasplitter.py``. Also, data
        splitter can be set directly using ``datamodule.datasplitter = YourSplitter()``

        Parameters
        ----------
        config
            The name of a data splitter implemented in ``data/datasplitter.py`` or a tuple providing the name and kwargs
            of the data splitter
        verbose
            Ignored.
        """
        from src.data.datasplitter import get_data_splitter

        if type(config) in [tuple, list]:
            self.datasplitter = get_data_splitter(config[0])(**dict(config[1]))
        else:
            self.datasplitter = get_data_splitter(config)()

    def set_data_imputer(self, name, verbose=True):
        """
        Set the data imputer. The specified splitter should be implemented in ``data/dataimputer.py``. Also, data
        imputer can be set directly using ``datamodule.dataimputer = YourImputer()``

        Parameters
        ----------
        name
            The name of a data imputer implemented in ``data/dataimputer.py``.
        verbose
            Ignored.
        """
        from src.data.dataimputer import get_data_imputer

        self.dataimputer = get_data_imputer(name)()

    def set_data_processors(self, config: List[Tuple[str, Dict]], verbose=True):
        """
        Set a list of data processors with the name and arguments for each data processors. The processor should be
        implemented in ``data/dataprocessor.py``. Also, data processors can be set directly using
        ``datamodule.dataprocessors = [(YourProcessor(), A Dict of kwargs) for EACH DATA PROCESSOR]``

        Parameters
        ----------
        config
            A list of tuple. The tuple includes the name of the processor and a dict of kwargs for the processor.
        verbose
            Ignored.

        Notes
        ----------
        The `UnscaledDataRecorder` should be set before any scaling processor. If not found in the input, it will be
        appended at the end.
        """
        from src.data.dataprocessor import get_data_processor, AbstractScaler

        self.dataprocessors = [
            (get_data_processor(name)(), kwargs) for name, kwargs in config
        ]
        is_scaler = np.array(
            [
                int(isinstance(x, AbstractScaler))
                for x in [processor for processor, _ in self.dataprocessors]
            ]
        )
        if np.sum(is_scaler) > 1:
            raise Exception(f"More than one AbstractScaler.")
        if is_scaler[-1] != 1:
            raise Exception(f"The last dataprocessor should be an AbstractScaler.")

    def set_data_derivers(self, config: List[Tuple[str, Dict]], verbose=True):
        """
        Set a list of data derivers with the name and arguments for each data derivers. The deriver should be
        implemented in data/dataderiver.py. Also, data derivers can be set directly using
        ``datamodule.dataderivers = [(YourDeriver(), A Dict of kwargs) for EACH DATA DERIVER]``

        Parameters
        ----------
        config
            A list of tuple. The tuple includes the name of the deriver and a dict of kwargs for the deriver.
        verbose
            Ignored.
        """
        from src.data.dataderiver import get_data_deriver

        self.dataderivers = [
            (get_data_deriver(name)(), kwargs) for name, kwargs in config
        ]

    def load_data(
        self,
        data_path: str = None,
        save_path: str = None,
        **kwargs,
    ) -> None:
        """
        Load tabular data. Either .csv or .xlsx is supported.

        Parameters
        ----------
        data_path
            Path to the tabular data. By default, the file ``data/{database}.csv/.xlsx`` is loaded.
        save_path
            Path to save the loaded data.
        **kwargs
            Arguments for pd.read_excel or pd.read_csv.
        """
        if data_path is None and self.data_path is None:
            data_path = os.path.join(
                src.setting["default_data_path"], f"{self.args['database']}"
            )
        elif self.data_path is not None:
            print(f"Using previously used data path {self.data_path}")
            data_path = self.data_path
        file_type = os.path.splitext(data_path)[-1]
        if file_type == "":
            is_csv = os.path.isfile(data_path + ".csv")
            is_xlsx = os.path.isfile(data_path + ".xlsx")
            if is_csv and is_xlsx:
                raise Exception(
                    f"Both {data_path}.csv and {data_path}.xlsx exist. Provide the postfix in data_path."
                )
            if not is_csv and not is_xlsx:
                raise Exception(f"{data_path}.csv and .xlsx do not exist.")
            file_type = ".csv" if is_csv else ".xlsx"
            data_path = data_path + file_type
        if file_type == ".xlsx":
            self.df = pd.read_excel(data_path, engine="openpyxl", **kwargs)
        else:
            self.df = pd.read_csv(data_path, **kwargs)
        self.data_path = data_path

        cont_feature_names = self.extract_cont_feature_names(
            self.args["feature_names_type"].keys()
        )
        cat_feature_names = self.extract_cat_feature_names(
            self.args["feature_names_type"].keys()
        )
        label_name = self.args["label_name"]

        self.set_data(self.df, cont_feature_names, cat_feature_names, label_name)
        print(
            "Dataset size:",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        if save_path is not None:
            self.save_data(save_path)

    def set_data(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        derived_data: Dict[str, np.ndarray] = None,
        warm_start: bool = False,
        verbose: bool = True,
        all_training: bool = False,
        force_features: bool = False,
        train_indices: np.ndarray = None,
        val_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
    ):
        """
        Set up the datamodule with a dataframe. Data splitting, imputation, derivation, and processing will be performed.

        Parameters
        ----------
        df
            The tabular dataset.
        cont_feature_names
            A list of continuous features in the tabular dataset.
        cat_feature_names
            A list of categorical features in the tabular dataset.
        label_name
            A list of targets. Currently, only one target is supported.
        derived_data
            The derived data calculated using data derivers with the argument "stacked" to False, i.e. unstacked data.
        warm_start
            Whether to use fitted data imputers and processors to process the data.
        verbose
            Verbosity.
        all_training
            Whether all samples are used for training.
        force_features
            Ignore derived features and do not remove features.
        train_indices
            Manually specify the training set by indices.
        val_indices
            Manually specify the validation set by indices.
        test_indices
            Manually specify the testing set by indices.
        """
        if len(label_name) > 1:
            raise Exception(f"Only one target is supported.")

        self.set_status(training=True)
        self._force_features = force_features
        self.cont_feature_names = cont_feature_names
        self.cat_feature_names = cat_feature_names
        self.cat_feature_mapping = {}
        for feature in self.cat_feature_mapping:
            self.cat_feature_mapping[feature] = []
        self.label_name = label_name
        self.df = df.copy()
        if np.isnan(df[self.label_name].values).any():
            raise Exception("Label missing in the input dataframe.")

        if all_training:
            self.train_indices = self.test_indices = self.val_indices = np.arange(
                len(self.df)
            )
        elif train_indices is None or val_indices is None or test_indices is None:
            (
                self.train_indices,
                self.val_indices,
                self.test_indices,
            ) = self.datasplitter.split(
                self.df, cont_feature_names, cat_feature_names, label_name
            )
        else:
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.test_indices = test_indices

        self._cont_imputed_mask = pd.DataFrame(
            columns=self.cont_feature_names,
            data=np.isnan(self.unscaled_feature_data.values).astype(int),
            index=np.arange(len(self.df)),
        )
        self._cat_imputed_mask = pd.DataFrame(
            columns=self.cat_feature_names,
            data=pd.isna(self.df[self.cat_feature_names]).values.astype(int),
            index=np.arange(len(self.df)),
        )

        def make_imputation():
            train_val_indices = list(self.train_indices) + list(self.val_indices)
            self.df.loc[train_val_indices, :] = self.dataimputer.fit_transform(
                self.df.loc[train_val_indices, :], datamodule=self
            )
            self.df.loc[self.test_indices, :] = self.dataimputer.transform(
                self.df.loc[self.test_indices, :], datamodule=self
            )

        make_imputation()
        (
            self.df,
            cont_feature_names,
        ) = self.derive_stacked(self.df)
        if not self._force_features:
            self.cont_feature_names = cont_feature_names
        # There may exist nan in stacked features.
        self._cont_imputed_mask = pd.concat(
            [
                self._cont_imputed_mask,
                pd.DataFrame(
                    columns=self.derived_stacked_features,
                    data=np.isnan(self.df[self.derived_stacked_features].values).astype(
                        int
                    ),
                    index=np.arange(len(self.df)),
                ),
            ],
            axis=1,
        )[self.cont_feature_names]
        make_imputation()

        self._data_process(
            warm_start=warm_start,
            verbose=verbose,
        )

        self._cont_imputed_mask = (
            self._cont_imputed_mask.loc[self.retained_indices, self.cont_feature_names]
            .copy()
            .reset_index(drop=True)
        )
        self._cat_imputed_mask = (
            self._cat_imputed_mask.loc[self.retained_indices, self.cat_feature_names]
            .copy()
            .reset_index(drop=True)
        )

        def update_indices(indices):
            return np.array(
                [
                    x - np.count_nonzero(self.dropped_indices < x)
                    for x in indices
                    if x in self.retained_indices
                ]
            )

        self.train_indices = update_indices(self.train_indices)
        self.test_indices = update_indices(self.test_indices)
        self.val_indices = update_indices(self.val_indices)

        if len(self.augmented_indices) > 0:
            augmented_indices = self.augmented_indices - len(self.dropped_indices)
            np.random.shuffle(augmented_indices)
            self.train_indices = np.array(
                list(self.train_indices) + list(augmented_indices)
            )

        if (
            len(self.train_indices) == 0
            or len(self.val_indices) == 0
            or len(self.test_indices) == 0
        ):
            raise Exception(
                "No sufficient data after preprocessing. This is caused by arguments train/val/test_"
                "indices or warm_start of set_data()."
            )

        self.derived_data = (
            self.derive_unstacked(self.df)
            if derived_data is None
            else self.sort_derived_data(derived_data)
        )
        self._update_dataset_auto()
        self._material_code = (
            pd.DataFrame(self.df["Material_Code"])
            if "Material_Code" in self.df.columns
            else None
        )
        self.set_status(training=False)

    def prepare_new_data(
        self, df: pd.DataFrame, derived_data: dict = None, ignore_absence=False
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Prepare the new tabular dataset for predictions from AbstractModel._predict. Stacked and unstacked features
        are derived; missing values are imputed; AbstractProcessor.transform are called. Users usually do not need to
        call this because AbstractModel.predict does it.

        Parameters
        ----------
        df:
            A new tabular dataset.
        derived_data:
            Data derived from :func:``DataModule.derive_unstacked``. If not None, unstacked data will be re-derived.
        ignore_absence:
            Whether to ignore absent keys in derived_data. Use True only when the model does not use derived_data.

        Returns
        -------
        df
            The dataset after derivation, imputation, and processing. It has the same structure as self.X_train
        derived_data:
            Data derived from :func:``DataModule.derive_unstacked``. It has the same structure as self.D_train

        Notes
        -------
        The returned df is not scaled for the sake of further treatments. To scale the df,
        run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        self.set_status(training=False)
        absent_features = [
            x
            for x in np.setdiff1d(self.all_feature_names, self.derived_stacked_features)
            if x not in df.columns
        ]
        absent_derived_features = [
            x for x in self.derived_stacked_features if x not in df.columns
        ]
        if len(absent_features) > 0:
            raise Exception(f"Feature {absent_features} not in the input dataframe.")

        if derived_data is None or len(absent_derived_features) > 0:
            df, _, derived_data = self.derive(df)
        else:
            absent_keys = [
                key
                for key in self.derived_data.keys()
                if key not in derived_data.keys()
            ]
            if len(absent_keys) > 0 and not ignore_absence:
                raise Exception(
                    f"Additional feature {absent_keys} not in the input derived_data."
                )
        df = self.dataimputer.transform(df.copy(), self)
        df = self.data_transform(df, skip_scaler=True)
        derived_data = self.sort_derived_data(
            derived_data, ignore_absence=ignore_absence
        )
        return df, derived_data

    @property
    def cont_imputed_mask(self) -> pd.DataFrame:
        """
        A byte mask for continuous data, where 1 means the data is imputed, and 0 means the data originally exists.

        Returns
        -------
        mask
            A byte mask dataframe.
        """
        return self._cont_imputed_mask[self.cont_feature_names]

    @property
    def cat_imputed_mask(self) -> pd.DataFrame:
        """
        A byte mask for categorical data, where 1 means the data is imputed, and 0 means the data originally exists.

        Returns
        -------
        mask
            A byte mask dataframe.
        """
        return self._cat_imputed_mask[self.cat_feature_names]

    @property
    def all_feature_names(self) -> List[str]:
        """
        Get continuous feature names and categorical feature names after ``load_data()``.

        Returns
        -------
        names:
            A list of continuous features and categorical features.
        """
        return self.cont_feature_names + self.cat_feature_names

    @property
    def derived_stacked_features(self) -> List[str]:
        """
        Find derived features in ``all_feature_names`` derived by data derivers with the argument "stacked" to True,
        i.e. stacked data.

        Returns
        -------
        names:
            A list of feature names.
        """
        return self.extract_derived_stacked_feature_names(self.all_feature_names)

    def get_feature_names_by_type(self, typ: str) -> List[str]:
        """
        Find features with the type given by ``feature_names_type`` and ``feature_types`` in the configuration.

        Parameters
        ----------
        typ
            One type of features in ``feature_types`` in the configuration.

        Returns
        -------
        names
            A list of found features.

        See Also
        -------
        ``Trainer.get_feature_idx_by_type``
        """
        if typ not in self.args["feature_types"]:
            raise Exception(
                f"Feature type {typ} is invalid (among {self.args['feature_types']})"
            )
        return [
            name
            for name, type_idx in self.args["feature_names_type"].items()
            if type_idx == self.args["feature_types"].index(typ)
            and name in self.all_feature_names
        ]

    def get_feature_idx_by_type(self, typ: str) -> np.ndarray:
        """
        Find features (by their index) with the type given by ``feature_names_type`` and ``feature_types`` in the
        configuration.

        Parameters
        ----------
        typ
            One type of features in ``feature_types`` in the configuration.

        Returns
        -------
        indices
            A list of indices of found features.

        See Also
        -------
        ``Trainer.get_feature_names_by_type``
        """
        names = self.get_feature_names_by_type(typ=typ)
        if typ == "Categorical":
            return np.array([self.cat_feature_names.index(name) for name in names])
        else:
            return np.array([self.cont_feature_names.index(name) for name in names])

    def extract_cont_feature_names(self, all_feature_names: List[str]) -> List[str]:
        """
        Get original continuous features specified in the config file.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some original features in the config file.

        Returns
        -------
        name
            Names of continuous original features both in the config file and the input list.
        """
        return [
            x
            for x in all_feature_names
            if x in self.args["feature_names_type"].keys()
            and x not in self.args["categorical_feature_names"]
        ]

    def extract_cat_feature_names(self, all_feature_names):
        """
        Get original categorical features specified in the config file.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some original features in the config file.

        Returns
        -------
        name
            Names of categorical original features that are both in the config file and the input list.
        """
        return [
            str(x)
            for x in np.intersect1d(
                list(all_feature_names),
                self.args["categorical_feature_names"],
            )
        ]

    def extract_derived_stacked_feature_names(
        self, all_feature_names: List[str]
    ) -> List[str]:
        """
        Find derived features in the input list derived by data derivers with the argument "stacked" to True,
        i.e. stacked data.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some stacked features.

        Returns
        -------
        names
            Names of stacked features in the input list.
        """
        return [
            str(x)
            for x in np.setdiff1d(
                all_feature_names,
                np.append(
                    self.extract_cont_feature_names(all_feature_names),
                    self.extract_cat_feature_names(all_feature_names),
                ),
            )
        ]

    def set_feature_names(self, all_feature_names: List[str]):
        """
        Set feature names to a subset of current features (i.e. ``all_feature_names``) and reload the data.

        Parameters
        ----------
        all_feature_names
            A subset of current features.
        """
        self.set_status(training=True)
        cont_feature_names = self.extract_cont_feature_names(all_feature_names)
        cat_feature_names = self.extract_cat_feature_names(all_feature_names)
        derived_stacked_features = self.extract_derived_stacked_feature_names(
            all_feature_names
        )
        if len(cont_feature_names) == 0:
            raise Exception(f"At least one continuous feature should be provided.")
        has_indices = hasattr(self, "train_indices")
        self.set_data(
            self.df,
            cont_feature_names,
            cat_feature_names,
            self.label_name,
            verbose=False,
            force_features=True,
            train_indices=self.train_indices if has_indices else None,
            val_indices=self.val_indices if has_indices else None,
            test_indices=self.test_indices if has_indices else None,
        )
        # Some derived stacked features are not in the specified list.
        self.cont_feature_names = [
            x
            for x in self.cont_feature_names
            if not (
                x in self.derived_stacked_features and x not in derived_stacked_features
            )
        ]
        self._update_dataset_auto()
        self._force_features = True
        self.set_status(training=False)

    def sort_derived_data(
        self, derived_data: Dict[str, np.ndarray], ignore_absence: bool = False
    ) -> Union[Dict[str, np.ndarray], None]:
        """
        Sort the dict of derived unstacked data according to the order of derivation.

        Parameters
        ----------
        derived_data
            A dict of derived unstacked data calculated by ``Trainer.derive_unstacked()``
        ignore_absence
            Whether to ignore the absence of any derived unstacked data that does not exist in the input.

        Returns
        -------
        derived_data
            The sorted derived unstacked data.
        """
        if derived_data is None:
            return None
        else:
            tmp_derived_data = {}
            for key in self.derived_data.keys():
                if ignore_absence:
                    if key in derived_data.keys():
                        tmp_derived_data[key] = derived_data[key]
                else:
                    tmp_derived_data[key] = derived_data[key]
            return tmp_derived_data

    def categories_inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation of `data.dataprocessor.OrdinalEncoder` of categorical features (If there is one in
        `self.dataprocessors`).

        Parameters
        ----------
        X
            The data to be inverse-transformed.

        Returns
        -------
        data
            The inverse-transformed data.
        """
        from src.data.dataprocessor import CategoricalOrdinalEncoder

        for processor, _ in self.dataprocessors:
            if isinstance(processor, CategoricalOrdinalEncoder):
                encoder = processor.transformer
                cat_features = processor.record_cat_features
                if len(cat_features) == 0:
                    return X.copy()
                break
        else:
            return X.copy()
        X_copy = X.copy()
        try:
            X_copy.loc[:, cat_features] = encoder.inverse_transform(
                X[cat_features].copy()
            )
        except:
            try:
                encoder.transform(X[cat_features].copy())
                return X_copy
            except:
                raise Exception(
                    f"Categorical features are not compatible with the fitted OrdinalEncoder."
                )
        return X_copy

    def save_data(self, path: str):
        """
        Save the tabular data processed by ``set_data()``. Two files will be saved: ``data.csv`` contains all
        information from the input dataframe, and ``tabular_data.csv`` contains merely used features.

        Parameters
        ----------
        path
            The path to save the data.
        """
        self.categories_inverse_transform(self.df).to_csv(
            os.path.join(path, "data.csv"), encoding="utf-8", index=False
        )
        tabular_data, _, cat_feature_names, _ = self.get_tabular_dataset()
        tabular_data_inv = (
            self.categories_inverse_transform(tabular_data)
            if len(cat_feature_names) > 0
            else tabular_data
        )
        tabular_data_inv.to_csv(
            os.path.join(path, "tabular_data.csv"), encoding="utf-8", index=False
        )

        print(f"Data saved to {path} (data.csv and tabular_data.csv).")

    def derive(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, np.ndarray]]:
        """
        Derive both stacked and unstacked features using the input dataframe.

        Parameters
        ----------
        df
            The tabular dataset.

        Returns
        -------
        df_tmp
            The tabular dataset with derived stacked features.
        cont_feature_names
            Continuous feature names with derived stacked features.
        derived_data
            The derived unstacked data.
        """
        df_tmp, cont_feature_names = self.derive_stacked(df)
        derived_data = self.derive_unstacked(df_tmp)

        return df_tmp, cont_feature_names, derived_data

    def derive_stacked(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Derive stacked features using the input dataframe.

        Parameters
        ----------
        df
            The tabular dataset.

        Returns
        -------
        df_tmp
            The tabular dataset with derived stacked features.
        cont_feature_names
            Continuous feature names with derived stacked features.
        """
        df_tmp = df.copy()
        cont_feature_names = cp(self.cont_feature_names)
        for deriver, kwargs in self.dataderivers:
            kwargs = deriver.make_defaults(**kwargs)
            if kwargs["stacked"]:
                value, name, col_names = deriver.derive(
                    df_tmp, datamodule=self, **kwargs
                )
                if not kwargs["intermediate"]:
                    for col_name in col_names:
                        if col_name not in cont_feature_names:
                            cont_feature_names.append(col_name)
                df_tmp[col_names] = value
        return df_tmp, cont_feature_names

    def derive_unstacked(
        self, df: pd.DataFrame, categorical_only=False
    ) -> Dict[str, np.ndarray]:
        """
        Derive unstacked features using the input dataframe.

        Parameters
        ----------
        df
            The tabular dataset.
        categorical_only
            Whether to get categorical features only in the returned dict.

        Returns
        -------
        derived_data
            The derived unstacked data.
        """
        derived_data = {}
        if not categorical_only:
            for deriver, kwargs in self.dataderivers:
                kwargs = deriver.make_defaults(**kwargs)
                if not kwargs["stacked"]:
                    value, name, _ = deriver.derive(df, datamodule=self, **kwargs)
                    derived_data[name] = value
        if len(self.cat_feature_names) > 0:
            derived_data["categorical"] = df[self.cat_feature_names].values
        if len(self.augmented_indices) > 0:
            augmented = np.zeros((len(df), 1))
            if self.training:
                augmented[self.augmented_indices - len(self.dropped_indices), 0] = 1
            derived_data["augmented"] = augmented
        return derived_data

    def _data_process(
        self,
        warm_start: bool = False,
        verbose: bool = True,
    ):
        """
        Main procedure to process data after splitting and imputation. Both scaled and unscaled data will be recorded.
        Note that processors will fit on training datasets and transform on validation and testing dataset.

        Parameters
        ----------
        warm_start
            Whether to use fitted data imputers and processors to process the data.
        verbose
            Verbosity.
        """
        self.df.reset_index(drop=True, inplace=True)
        self.scaled_df = self.df.copy()
        original_length = len(self.df)

        with HiddenPrints(disable_std=not verbose):
            unscaled_training_data = self._data_preprocess(
                self.df.loc[list(self.train_indices) + list(self.val_indices), :],
                warm_start=warm_start,
                skip_scaler=True,
            )
            training_data = self._data_preprocess(
                unscaled_training_data, warm_start=warm_start, scaler_only=True
            )
            unscaled_testing_data = self.data_transform(
                self.df.loc[self.test_indices, :], skip_scaler=True
            )
            testing_data = self.data_transform(unscaled_testing_data, scaler_only=True)

        all_indices = np.unique(
            np.sort(np.array(list(training_data.index) + list(testing_data.index)))
        )
        self.retained_indices = np.intersect1d(all_indices, np.arange(original_length))
        self.dropped_indices = np.setdiff1d(
            np.arange(original_length), self.retained_indices
        )
        self.augmented_indices = np.setdiff1d(
            training_data.index, np.arange(original_length)
        )
        if len(np.setdiff1d(testing_data.index, np.arange(original_length))) > 0:
            raise Exception(f"Testing data should not be augmented.")

        def process_df(df, training, testing):
            inplace_training_index = np.intersect1d(
                training.index, self.retained_indices
            )
            df.loc[inplace_training_index, training.columns] = training.loc[
                inplace_training_index, :
            ].values
            df.loc[testing.index, testing.columns] = testing.values
            df = df.loc[self.retained_indices, :].copy().reset_index(drop=True)
            if len(self.augmented_indices) > 0:
                df = pd.concat(
                    [df, training.loc[self.augmented_indices, :]], axis=0
                ).reset_index(drop=True)
            df.loc[:, self.cat_feature_names] = df.loc[
                :, self.cat_feature_names
            ].astype(np.int16)
            return df

        self.df = process_df(self.df, unscaled_training_data, unscaled_testing_data)
        self.scaled_df = process_df(self.scaled_df, training_data, testing_data)

    @property
    def unscaled_feature_data(self) -> pd.DataFrame:
        """
        Get unscaled feature data.

        Returns
        -------
        df
            The unscaled feature data.
        """
        return self.df[self.cont_feature_names].copy()

    @property
    def unscaled_label_data(self) -> pd.DataFrame:
        """
        Get unscaled label data.

        Returns
        -------
        df
            The unscaled label data.
        """
        return self.df[self.label_name].copy()

    @property
    def categorical_data(self) -> pd.DataFrame:
        """
        Get categorical data.

        Returns
        -------
        df
            The categorical data.
        """
        return self.df[self.cat_feature_names].copy()

    @property
    def feature_data(self) -> pd.DataFrame:
        """
        Get scaled feature data.

        Returns
        -------
        df
            The scaled feature data.
        """
        return self.scaled_df[self.cont_feature_names].copy()

    @property
    def label_data(self) -> pd.DataFrame:
        """
        Get scaled label data.

        Returns
        -------
        df
            The scaled label data.
        """
        return self.scaled_df[self.label_name].copy()

    def dataset_dict(self):
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "D_train": self.D_train,
            "D_val": self.D_val,
            "D_test": self.D_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
        }

    def __getitem__(self, item):
        return self.dataset_dict()[item]

    @property
    def X_train(self):
        return self.df.loc[self.train_indices, :].copy()

    @property
    def X_val(self):
        return self.df.loc[self.val_indices, :].copy()

    @property
    def X_test(self):
        return self.df.loc[self.test_indices, :].copy()

    @property
    def y_train(self):
        return self.df.loc[self.train_indices, self.label_name].values

    @property
    def y_val(self):
        return self.df.loc[self.val_indices, self.label_name].values

    @property
    def y_test(self):
        return self.df.loc[self.test_indices, self.label_name].values

    @property
    def D_train(self):
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.train_indices
        )

    @property
    def D_val(self):
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.val_indices
        )

    @property
    def D_test(self):
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.test_indices
        )

    def _data_preprocess(
        self,
        input_data: pd.DataFrame,
        warm_start: bool = False,
        skip_scaler: bool = False,
        scaler_only: bool = False,
    ) -> pd.DataFrame:
        """
        Call data processors to fit and/or transform the input tabular dataset.

        Parameters
        ----------
        input_data
            The tabular dataset.
        warm_start
            False to fit and transform data processors, and True to transform only.
        skip_scaler
            True to skip scaling (the last processor).
        scaler_only
            True to only perform scaling (the last processor).

        Returns
        -------
        df
            The processed data.
        """
        from .base import AbstractScaler

        if skip_scaler and scaler_only:
            raise Exception(f"Both skip_scaler and scaler_only are True.")
        data = input_data.copy()
        for processor, kwargs in self.dataprocessors:
            if skip_scaler and isinstance(processor, AbstractScaler):
                continue
            if scaler_only and not isinstance(processor, AbstractScaler):
                continue
            if warm_start:
                data = processor.transform(data, self, **kwargs)
            else:
                data = processor.fit_transform(data, self, **kwargs)
        return data

    def data_transform(
        self,
        input_data: pd.DataFrame,
        skip_scaler: bool = False,
        scaler_only: bool = False,
    ):
        """
        Transform the input tabular dataset using fitted data processors. Only AbstractTransformers take effects.

        Parameters
        ----------
        input_data
            The tabular dataset.
        skip_scaler
            True to skip scaling (the last processor).
        scaler_only
            True to only perform scaling (the last processor).

        Returns
        -------
        df
            Transformed tabular dataset.
        """
        return self._data_preprocess(
            input_data.copy(),
            warm_start=True,
            skip_scaler=skip_scaler,
            scaler_only=scaler_only,
        )

    def _update_dataset_auto(self):
        """
        Update PyTorch tensors and datasets for the datamodule. This is called after features change.
        """
        X = torch.tensor(
            self.feature_data.values.astype(np.float32), dtype=torch.float32
        )
        y = torch.tensor(self.label_data.values.astype(np.float32), dtype=torch.float32)

        D = [
            torch.tensor(value.astype(np.float32))
            for value in self.derived_data.values()
        ]
        dataset = Data.TensorDataset(X, *D, y)

        self.train_dataset = Subset(dataset, self.train_indices)
        self.val_dataset = Subset(dataset, self.val_indices)
        self.test_dataset = Subset(dataset, self.test_indices)
        self.tensors = (X, *D, y) if len(D) > 0 else (X, None, y)

    def get_derived_data_slice(
        self, derived_data: Dict[str, np.ndarray], indices: Iterable
    ) -> Dict[str, np.ndarray]:
        """
        Get slices of the derived unstacked data.

        Parameters
        ----------
        derived_data
            Derived unstacked data.
        indices
            The indices to make slice.

        Returns
        -------
        derived_data
            The sliced derived stacked data.
        """
        tmp_derived_data = {}
        for key, value in derived_data.items():
            tmp_derived_data[key] = value[indices, :]
        return tmp_derived_data

    def get_zero_slip(self, feature_name: str) -> float:
        """
        See how data processors act on a feature if its value is zero.
        It is a wrapper for ``Trainer.get_var_change``.

        Parameters
        ----------
        feature_name
            The investigated feature.

        Returns
        -------
        value
            The transformed value for the feature using data processors.
        """
        return self.get_var_change(feature_name=feature_name, value=0)

    def get_var_change(self, feature_name: str, value: float) -> float:
        """
        See how data processors act on a feature if its value is ``value``.

        Parameters
        ----------
        feature_name
            The investigated feature.
        value
            The investigated value.

        Returns
        -------
        value
            The transformed value for the feature using data processors.
        """
        from .dataprocessor import AbstractTransformer

        if not hasattr(self, "dataprocessors"):
            raise Exception(f"Run load_config first.")
        elif len(self.dataprocessors) == 0 and feature_name in self.cont_feature_names:
            return 0
        if feature_name not in self.dataprocessors[-1][0].record_cont_features:
            raise Exception(f"Feature {feature_name} not available.")

        x = value
        for processor, _ in self.dataprocessors:
            if isinstance(processor, AbstractTransformer) and hasattr(
                processor, "transformer"
            ):
                x = processor.var_slip(feature_name, x)
        return x

    def describe(self, transformed=False) -> pd.DataFrame:
        """
        Describe the dataset using pd.DataFrame.describe, skewness, gini index, mode values, etc.

        Parameters
        ----------
        transformed
            Whether to describe the transformed (scaled) data.

        Returns
        -------
        desc
            The descriptions of the dataset.
        """
        tabular = self.get_tabular_dataset(transformed=transformed)[0]
        desc = tabular.describe()

        skew = tabular.skew()
        desc = pd.concat(
            [
                desc,
                pd.DataFrame(
                    data=skew.values.reshape(len(skew), 1).T,
                    columns=skew.index,
                    index=["Skewness"],
                ),
            ],
            axis=0,
        )

        g = self._get_gini(tabular)
        desc = pd.concat([desc, g], axis=0)

        mode, cnt_mode, mode_percent = self._get_mode(tabular)
        desc = pd.concat([desc, mode, cnt_mode, mode_percent], axis=0)

        return desc

    def get_derived_data_sizes(self) -> List[Tuple]:
        """
        Get dimensions of derived unstacked features.

        Returns
        -------
        sizes
            A list of np.ndarray.shape representing dimensions of each derived unstacked features.
        """
        return [x.shape for x in self.derived_data.values()]

    @staticmethod
    def _get_gini(tabular: pd.DataFrame) -> pd.DataFrame:
        """
        Get the gini index for each feature in the tabular dataset.

        Parameters
        ----------
        tabular
            The tabular dataset.

        Returns
        -------
        gini
            The gini index for each feature in the dataset.
        """
        return pd.DataFrame(
            data=np.array([[gini(tabular[x]) for x in tabular.columns]]),
            columns=tabular.columns,
            index=["Gini Index"],
        )

    @staticmethod
    def _get_mode(
        tabular: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the mode value for each feature in the tabular dataset.

        Parameters
        ----------
        tabular
            The tabular dataset.

        Returns
        -------
        mode
            The mode value for each feature in the dataset, and the number and percentage of the mode value.
        """
        mode = tabular.mode().loc[0, :]
        cnt_mode = pd.DataFrame(
            data=np.array(
                [
                    [
                        tabular[mode.index[x]].value_counts()[mode.values[x]]
                        for x in range(len(mode))
                    ]
                ]
            ),
            columns=tabular.columns,
            index=["Mode counts"],
        )
        mode_percent = cnt_mode / tabular.count()
        mode_percent.index = ["Mode percentage"]

        mode = pd.DataFrame(
            data=mode.values.reshape(len(mode), 1).T, columns=mode.index, index=["Mode"]
        )
        return mode, cnt_mode, mode_percent

    def pca(
        self, feature_names: List[str] = None, feature_idx: List[int] = None, **kwargs
    ) -> PCA:
        """
        Perform sklearn.decomposition.PCA

        Parameters
        -------
        feature_names
            A list of names of continuous features.
        **kwargs
            Arguments of sklearn.decomposition.PCA.

        Returns
        -------
        pca
            A sklearn.decomposition.PCA instance.
        """
        pca = PCA(**kwargs)
        if feature_names is not None:
            pca.fit(self.feature_data.loc[self.train_indices, feature_names])
        elif feature_idx is not None:
            pca.fit(
                self.feature_data.loc[
                    self.train_indices, np.array(self.cont_feature_names)[feature_idx]
                ]
            )
        else:
            pca.fit(self.feature_data.loc[self.train_indices, :])
        return pca

    def divide_from_tabular_dataset(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get continuous feature data, categorical feature data, and label data respectively.

        Parameters
        ----------
        data
            The tabular dataset.

        Returns
        -------
        feature_data
            The continuous feature data.
        categorical_data
            The categorical feature data.
        label_data
            The label data.
        """
        feature_data = data[self.cont_feature_names]
        categorical_data = data[self.cat_feature_names]
        label_data = data[self.label_name]

        return feature_data, categorical_data, label_data

    def get_tabular_dataset(
        self, transformed: bool = False
    ) -> Tuple[pd.DataFrame, List, List, List]:
        """
        Get the tabular dataset loaded in the datamodule.

        Parameters
        ----------
        transformed
            Whether to return the scaled data or not.

        Returns
        -------
        tabular_dataset
            The tabular dataset.
        cont_feature_names
            The continuous feature names in the dataset.
        cat_feature_names
            The categorical feature names in the dataset.
        label_names
            The target names (only one target is currently supported).
        """
        if transformed:
            feature_data = self.feature_data
        else:
            feature_data = self.unscaled_feature_data

        cont_feature_names = cp(self.cont_feature_names)
        cat_feature_names = cp(self.cat_feature_names)
        label_name = cp(self.label_name)

        tabular_dataset = pd.concat(
            [feature_data, self.categorical_data, self.label_data], axis=1
        )

        return tabular_dataset, cont_feature_names, cat_feature_names, label_name

    def cal_corr(
        self, imputed: bool = False, features_only: bool = False
    ) -> pd.DataFrame:
        """
        Calculate Pearson correlation among continuous features.

        Parameters
        ----------
        imputed
            Whether the imputed dataset should be considered. If False, some NaN values may exit for features with
            missing value.
        features_only
            If False, the target is also considered.

        Returns
        -------
        corr
            The correlation dataframe.
        """
        subset = (
            self.cont_feature_names
            if features_only
            else self.cont_feature_names + self.label_name
        )
        if not imputed:
            not_imputed_df = self.get_not_imputed_df()
            return not_imputed_df[subset].corr()
        else:
            return self.df[subset].corr()

    def get_not_imputed_df(self) -> pd.DataFrame:
        """
        Get the tabular data without imputation.

        Returns
        -------
        df
            The tabular dataset without imputation.
        """
        tmp_cont_df = self.df.copy().loc[:, self.cont_feature_names]
        if np.sum(np.abs(self.cont_imputed_mask.values)) != 0:
            tmp_cont_df.values[
                np.where(self.cont_imputed_mask[self.cont_feature_names].values)
            ] = np.nan
        tmp_cat_df = self.categories_inverse_transform(self.df).loc[
            :, self.cat_feature_names
        ]
        if np.sum(np.abs(self.cat_imputed_mask.values)) != 0:
            tmp_cat_df.values[
                np.where(self.cat_imputed_mask[self.cat_feature_names].values)
            ] = np.nan
        not_imputed_df = self.df.copy()
        not_imputed_df.loc[:, self.all_feature_names] = pd.concat(
            [tmp_cont_df, tmp_cat_df], axis=1
        )
        return not_imputed_df

    def _get_indices(self, partition: str = "train") -> np.ndarray:
        """
        Get training/validation/testing indices.

        Parameters
        ----------
        partition
            "train", "val", "test", or "all"
        Returns
        -------
        indices
            The indices of the selected partition.
        """
        indices_map = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
            "all": np.array(self.feature_data.index),
        }

        if partition not in indices_map.keys():
            raise Exception(
                f"Partition {partition} not available. Select among {list(indices_map.keys())}"
            )

        return indices_map[partition]

    def get_material_code(
        self, unique: bool = False, partition: str = "all"
    ) -> pd.DataFrame:
        """
        Get Material_Code of the dataset.

        Parameters
        ----------
        unique
            If True, values in the Material_Code column will be counted.
        partition
            "train", "val", "test", or "all". See ``Trainer._get_indices``.
        Returns
        -------
        m_code
            If unique is True, the returned dataframe contains counts for each material code in the selected partition.
            Otherwise, the original material codes in the selected partition are returned.
        """
        if self._material_code is None:
            raise Exception(
                f"The column Material_Code is not available in the dataset."
            )
        indices = self._get_indices(partition=partition)
        if unique:
            unique_list = list(
                sorted(set(self._material_code.loc[indices, "Material_Code"]))
            )
            val_cnt = self._material_code.loc[indices, :].value_counts()
            return pd.DataFrame(
                {
                    "Material_Code": unique_list,
                    "Count": [val_cnt[x] for x in unique_list],
                }
            )
        else:
            return self._material_code.loc[indices, :]

    def select_by_material_code(self, m_code: str, partition: str = "all"):
        """
        Select samples with the specified material code.

        Parameters
        ----------
        m_code
            The selected material code.
        partition
            "train", "val", "test", or "all". See ``Trainer._get_indices``.
        Returns
        -------
        indices
            The pandas index where the material code exists.
        """
        code_df = self.get_material_code(unique=False, partition=partition)
        return code_df.index[np.where(code_df["Material_Code"] == m_code)[0]]

    def get_additional_tensors_slice(self, indices) -> Union[Tuple[Any], Tuple]:
        """
        Get slices of the derived unstacked tensors.

        Parameters
        ----------
        indices
            The indices to make slice.

        Returns
        -------
        res
            Sliced derived unstacked tensors.
        """
        res = []
        for tensor in self.tensors[1 : len(self.tensors) - 1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)

    def get_first_tensor_slice(self, indices) -> torch.Tensor:
        """
        Get slices of the continuous tensor.

        Parameters
        ----------
        indices
            The indices to make slice.

        Returns
        -------
        res
            The sliced continuous tensor.
        """
        return self.tensors[0][indices, :]

    def get_base_predictor(
        self,
        categorical: bool = True,
        **kwargs,
    ) -> Union[sklearn.pipeline.Pipeline, sklearn.ensemble.RandomForestRegressor]:
        """
        Get a sklearn ``RandomForestRegressor`` for fundamental usages like pre-processing.

        Parameters
        ----------
        categorical
            Whether to include OneHotEncoder for categorical features.
        kwargs
            Arguments for ``sklearn.ensemble.RandomForestRegressor``

        Returns
        -------
        model
            A Pipeline if categorical is True, or a RandomForestRegressor if categorical is False.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(**kwargs)

        if len(self.cat_feature_names) > 0 and categorical:
            categorical_encoder = OneHotEncoder()
            numerical_pipe = SimpleImputer(strategy="mean")
            preprocessing = ColumnTransformer(
                [
                    (
                        "cat",
                        categorical_encoder,
                        lambda x: [y for y in self.cat_feature_names if y in x.columns],
                    ),
                    (
                        "num",
                        numerical_pipe,
                        lambda x: [
                            y for y in self.cont_feature_names if y in x.columns
                        ],
                    ),
                ],
                verbose_feature_names_out=False,
            )

            pip = Pipeline(
                [
                    ("preprocess", preprocessing),
                    ("classifier", rf),
                ]
            )
            return pip
        else:
            return rf
