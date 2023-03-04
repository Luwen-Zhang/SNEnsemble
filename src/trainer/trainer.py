"""
The basic class for the project. It includes configuration, data processing, plotting,
and comparing baseline models.
"""
from src.utils import *
from copy import deepcopy as cp
from importlib import import_module, reload
from skopt.space import Real, Integer, Categorical
import json
import time
from typing import *
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Subset
import itertools
import scipy.stats as st
from captum.attr import FeaturePermutation
from sklearn.utils import resample as skresample

set_random_seed(0)
sys.path.append("configs/")


class Trainer:
    def __init__(self, device="cpu", project=None):
        self.device = device
        self.project = project
        self.modelbases = []
        self.modelbases_names = []

    def add_modelbases(self, models: list):
        self.modelbases += models
        self.modelbases_names = [x.program for x in self.modelbases]
        if len(self.modelbases_names) != len(list(set(self.modelbases_names))):
            raise Exception(f"Conflicted modelbase names: {self.modelbases_names}")

    def get_modelbase(self, program: str):
        if program not in self.modelbases_names:
            raise Exception(f"Program {program} not added to the trainer.")
        return self.modelbases[self.modelbases_names.index(program)]

    def load_config(
        self,
        default_configfile: str = None,
        verbose: bool = True,
        manual_config: dict = None,
        project_root_subfolder: str = None,
    ) -> None:
        """
        Load a configfile.
        :param default_configfile: The path to a configfile. If in notebook environment, this parameter is required.
        If the parameter is assigned, the script will not take any input argument from command line. Default to None.
        :param verbose: Whether to output the loaded configs. Default to True.
        :return: None
        """
        base_config = import_module("base_config").BaseConfig().data

        # The base config is loaded using the --base argument
        if is_notebook() and default_configfile is None:
            raise Exception("A config file must be assigned in notebook environment.")
        elif is_notebook() or default_configfile is not None:
            parse_res = {"base": default_configfile}
        else:  # not notebook and configfile is None
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("--base", required=True)
            for key in base_config.keys():
                if type(base_config[key]) in [str, int, float]:
                    parser.add_argument(
                        f"--{key}", type=type(base_config[key]), required=False
                    )
                elif type(base_config[key]) == list:
                    parser.add_argument(f"--{key}", nargs="+", required=False)
                elif type(base_config[key]) == bool:
                    parser.add_argument(f"--{key}", dest=key, action="store_true")
                    parser.add_argument(f"--no-{key}", dest=key, action="store_false")
                    parser.set_defaults(**{key: base_config[key]})
            parse_res = parser.parse_args().__dict__

        self.configfile = parse_res["base"]

        if self.configfile not in sys.modules:
            arg_loaded = import_module(self.configfile).config().data
        else:
            arg_loaded = reload(sys.modules.get(self.configfile)).config().data

        # Then, several args can be modified using other arguments like --lr, --weight_decay
        # only when a config file is not given so that configs depend on input arguments.
        if not is_notebook() and default_configfile is None:
            for key, value in parse_res.items():
                if value is not None:
                    arg_loaded[key] = value

        if manual_config is not None:
            for key, value in manual_config.items():
                if key in arg_loaded.keys():
                    arg_loaded[key] = value
                else:
                    raise Exception(
                        f"Manual configuration argument {key} not available."
                    )

        # Preprocess configs
        tmp_static_params = {}
        for key in arg_loaded["static_params"]:
            tmp_static_params[key] = arg_loaded[key]
        arg_loaded["static_params"] = tmp_static_params

        tmp_chosen_params = {}
        for key in arg_loaded["chosen_params"]:
            tmp_chosen_params[key] = arg_loaded[key]
        arg_loaded["chosen_params"] = tmp_chosen_params

        key_chosen = list(arg_loaded["chosen_params"].keys())
        key_space = list(arg_loaded["SPACEs"].keys())
        for a, b in zip(key_chosen, key_space):
            if a != b:
                raise Exception(
                    "Variables in 'chosen_params' and 'SPACEs' should be in the same order."
                )

        if verbose:
            print(pretty(arg_loaded))

        self.args = arg_loaded

        self.loss = self.args["loss"]
        self.bayes_opt = self.args["bayes_opt"]

        self.database = self.args["database"]

        self.static_params = self.args["static_params"]
        self.chosen_params = self.args["chosen_params"]
        self.layers = self.args["layers"]
        self.n_calls = self.args["n_calls"]

        SPACE = []
        for var in key_space:
            setting = arg_loaded["SPACEs"][var]
            ty = setting["type"]
            setting.pop("type")
            if ty == "Real":
                SPACE.append(Real(name=var, **setting))
            elif ty == "Categorical":
                SPACE.append(Categorical(name=var, **setting))
            elif ty == "Integer":
                SPACE.append(Integer(name=var, **setting))
            else:
                raise Exception("Invalid type of skopt space.")
        self.SPACE = SPACE

        if self.loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss == "r2":
            self.loss_fn = r2_loss
        elif self.loss == "mae":
            self.loss_fn = nn.L1Loss()
        else:
            raise Exception(f"Loss function {self.loss} not implemented.")

        self.bayes_epoch = self.args["bayes_epoch"]

        self.set_data_splitter(name=self.args["data_splitter"], verbose=verbose)
        self.set_data_imputer(name=self.args["data_imputer"], verbose=verbose)
        self.set_data_processors(self.args["data_processors"], verbose=verbose)
        self.set_data_derivers(self.args["data_derivers"], verbose=verbose)

        self.project = self.database if self.project is None else self.project
        self.create_dir(project_root_subfolder=project_root_subfolder)

    def create_dir(self, verbose=True, project_root_subfolder=None):
        if project_root_subfolder is not None:
            if not os.path.exists(f"output/{project_root_subfolder}"):
                os.mkdir(f"output/{project_root_subfolder}")
        subfolder = (
            self.project
            if project_root_subfolder is None
            else f"{project_root_subfolder}/{self.project}"
        )
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        folder_name = t + "-0" + "_" + self.configfile
        self.project_root = f"output/{subfolder}/{folder_name}/"
        if not os.path.exists(f"output/{subfolder}"):
            os.mkdir(f"output/{subfolder}")
        postfix_iter = itertools.count()
        while os.path.exists(self.project_root):
            tmp_folder_name = (
                t + "-" + str(postfix_iter.__next__()) + "_" + self.configfile
            )
            self.project_root = f"output/{subfolder}/{tmp_folder_name}/"
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)

        json.dump(self.args, open(self.project_root + "args.json", "w"), indent=4)
        if verbose:
            print(f"Project will be saved to {self.project_root}")

    def set_data_splitter(self, name, verbose=True):
        from src.data.datasplitter import get_data_splitter

        self.datasplitter = get_data_splitter(name)()

    def set_data_imputer(self, name, verbose=True):
        from src.data.dataimputer import get_data_imputer

        self.dataimputer = get_data_imputer(name)()

    def set_data_processors(self, config: List[Tuple], verbose=True):
        from src.data.dataprocessor import get_data_processor

        if "UnscaledDataRecorder" not in [name for name, _ in config]:
            if verbose:
                print(
                    "UnscaledDataRecorder not in the data_processors pipeline. Only scaled data will be recorded."
                )
            self.args["data_processors"].append(("UnscaledDataRecorder", {}))
        self.dataprocessors = [
            (get_data_processor(name)(), kwargs) for name, kwargs in config
        ]

    def set_data_derivers(self, config: List[Tuple], verbose=True):
        from src.data.dataderiver import get_data_deriver

        self.dataderivers = [
            (get_data_deriver(name)(), kwargs) for name, kwargs in config
        ]

    def load_data(self, data_path: str = None, file_type: str = "csv") -> None:
        """
        Load the data file in ../data directory specified by the 'project' argument in configfile. Data will be splitted
         into training, validation, and testing datasets.
        :param data_path: specify a data file different from 'project'. Default to None.
        :return: None
        """
        if data_path is None:
            data_path = f"data/{self.database}.{file_type}"
        if file_type == "xlsx":
            self.df = pd.read_excel(data_path, engine="openpyxl")
        else:
            self.df = pd.read_csv(data_path)
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

        self.save_data()

    @property
    def all_feature_names(self):
        return self.cont_feature_names + self.cat_feature_names

    @property
    def derived_stacked_features(self):
        return self.extract_derived_stacked_feature_names(self.all_feature_names)

    def set_data(
        self,
        df,
        cont_feature_names,
        cat_feature_names,
        label_name,
        derived_data=None,
        warm_start=False,
        verbose=True,
        all_training=False,
        train_indices=None,
        val_indices=None,
        test_indices=None,
    ):
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
                self.df.loc[train_val_indices, :], trainer=self
            )
            self.df.loc[self.test_indices, :] = self.dataimputer.transform(
                self.df.loc[self.test_indices, :], trainer=self
            )

        make_imputation()
        (
            self.df,
            self.cont_feature_names,
        ) = self.derive_stacked(self.df)
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

    @property
    def cont_imputed_mask(self):
        return self._cont_imputed_mask[self.cont_feature_names]

    @property
    def cat_imputed_mask(self):
        return self._cat_imputed_mask[self.cat_feature_names]

    def extract_cont_feature_names(self, all_feature_names):
        return [
            x
            for x in all_feature_names
            if x in self.args["feature_names_type"].keys()
            and x not in self.args["categorical_feature_names"]
        ]

    def extract_cat_feature_names(self, all_feature_names):
        return [
            str(x)
            for x in np.intersect1d(
                list(all_feature_names),
                self.args["categorical_feature_names"],
            )
        ]

    def extract_derived_stacked_feature_names(self, all_feature_names):
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

    def set_feature_names(self, all_feature_names):
        cont_feature_names = self.extract_cont_feature_names(all_feature_names)
        cat_feature_names = self.extract_cat_feature_names(all_feature_names)
        derived_stacked_features = self.extract_derived_stacked_feature_names(
            all_feature_names
        )
        self.set_data(
            self.df,
            cont_feature_names,
            cat_feature_names,
            self.label_name,
            verbose=False,
            train_indices=self.train_indices,
            val_indices=self.val_indices,
            test_indices=self.test_indices,
        )
        self.cont_feature_names = [
            x
            for x in self.cont_feature_names
            if not (
                x in self.derived_stacked_features and x not in derived_stacked_features
            )
        ]
        self._update_dataset_auto()

    def sort_derived_data(self, derived_data):
        if derived_data is None:
            return None
        else:
            tmp_derived_data = {}
            for key in self.derived_data.keys():
                tmp_derived_data[key] = derived_data[key]
            return tmp_derived_data

    def categories_inverse_transform(self, X: pd.DataFrame):
        from src.data.dataprocessor import CategoricalOrdinalEncoder

        for processor, _ in self.dataprocessors:
            if type(processor) == CategoricalOrdinalEncoder:
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

    def save_data(self, path: str = None):
        if path is None:
            path = self.project_root

        self.categories_inverse_transform(self.df).to_csv(
            os.path.join(path, "data.csv"), encoding="utf-8", index=False
        )
        tabular_data, _, _, _ = self.get_tabular_dataset()
        self.categories_inverse_transform(tabular_data).to_csv(
            os.path.join(path, "tabular_data.csv"), encoding="utf-8", index=False
        )

        print(f"Data saved to {path} (data.csv and tabular_data.csv).")

    def derive(self, df):
        df_tmp, cont_feature_names = self.derive_stacked(df)
        derived_data = self.derive_unstacked(df_tmp)

        return (df_tmp, cont_feature_names, derived_data)

    def derive_stacked(self, df):
        df_tmp = df.copy()
        cont_feature_names = cp(self.cont_feature_names)
        for deriver, kwargs in self.dataderivers:
            kwargs = deriver.make_defaults(**kwargs)
            if kwargs["stacked"]:
                value, name, col_names = deriver.derive(df_tmp, trainer=self, **kwargs)
                if not kwargs["intermediate"]:
                    for col_name in col_names:
                        if col_name not in cont_feature_names:
                            cont_feature_names.append(col_name)
                df_tmp[col_names] = value
        return df_tmp, cont_feature_names

    def derive_unstacked(self, df):
        derived_data = {}
        for deriver, kwargs in self.dataderivers:
            kwargs = deriver.make_defaults(**kwargs)
            if not kwargs["stacked"]:
                value, name, _ = deriver.derive(df, trainer=self, **kwargs)
                derived_data[name] = value
        if len(self.cat_feature_names) > 0:
            derived_data["categorical"] = df[self.cat_feature_names].values
        return derived_data

    def _data_process(
        self,
        warm_start=False,
        verbose=True,
    ):
        self._unscaled_feature_data = pd.DataFrame()
        self._unscaled_label_data = pd.DataFrame()
        self._categorical_data = pd.DataFrame()
        self.df.reset_index(drop=True, inplace=True)
        self.scaled_df = self.df.copy()
        original_length = len(self.df)

        with HiddenPrints(disable_std=not verbose):
            training_data = self._data_preprocess(
                self.df.loc[list(self.train_indices) + list(self.val_indices), :],
                warm_start=warm_start,
            )
            unscaled_training_data = pd.concat(
                [
                    self._unscaled_feature_data,
                    self._categorical_data,
                    self._unscaled_label_data,
                ],
                axis=1,
            )
            testing_data = self.data_transform(self.df.loc[self.test_indices, :])
            unscaled_testing_data = pd.concat(
                [
                    self._unscaled_feature_data,
                    self._categorical_data,
                    self._unscaled_label_data,
                ],
                axis=1,
            )

        self.retained_indices = np.unique(
            np.sort(np.array(list(training_data.index) + list(testing_data.index)))
        )
        self.dropped_indices = np.setdiff1d(
            np.arange(original_length), self.retained_indices
        )

        def process_df(df, training, testing):
            df.loc[training.index, training.columns] = training.values
            df.loc[testing.index, testing.columns] = testing.values
            df = pd.DataFrame(df.loc[self.retained_indices, :]).reset_index(drop=True)
            df.loc[:, self.cat_feature_names] = df.loc[
                :, self.cat_feature_names
            ].astype(np.int16)
            return df

        self.df = process_df(self.df, unscaled_training_data, unscaled_testing_data)
        self.scaled_df = process_df(self.scaled_df, training_data, testing_data)

    @property
    def unscaled_feature_data(self):
        return self.df[self.cont_feature_names].copy()

    @property
    def unscaled_label_data(self):
        return self.df[self.label_name].copy()

    @property
    def categorical_data(self):
        return self.df[self.cat_feature_names].copy()

    @property
    def feature_data(self):
        return self.scaled_df[self.cont_feature_names].copy()

    @property
    def label_data(self):
        return self.scaled_df[self.label_name].copy()

    def _data_preprocess(self, input_data: pd.DataFrame, warm_start=False):
        data = input_data.copy()
        cont_feature_names = cp(self.cont_feature_names)
        cat_feature_names = cp(self.cat_feature_names)
        for processor, kwargs in self.dataprocessors:
            if warm_start:
                data = processor.transform(data, self, **kwargs)
            else:
                data = processor.fit_transform(data, self, **kwargs)
        data = data[self.all_feature_names + self.label_name]
        if warm_start:
            # If set_feature is called, and if some derived features are removed, recorded features in processors will
            # be restored when predicting new data.
            self.cont_feature_names = cont_feature_names
            self.cat_feature_names = cat_feature_names
        return data

    def data_transform(self, input_data: pd.DataFrame):
        return self._data_preprocess(input_data.copy(), warm_start=True)

    def _update_dataset_auto(self):
        X = torch.tensor(
            self.feature_data.values.astype(np.float32), dtype=torch.float32
        ).to(self.device)
        y = torch.tensor(
            self.label_data.values.astype(np.float32), dtype=torch.float32
        ).to(self.device)

        D = [
            torch.tensor(value, dtype=torch.float32).to(self.device)
            for value in self.derived_data.values()
        ]
        dataset = Data.TensorDataset(X, *D, y)

        self.train_dataset = Subset(dataset, self.train_indices)
        self.val_dataset = Subset(dataset, self.val_indices)
        self.test_dataset = Subset(dataset, self.test_indices)
        self.tensors = (X, *D, y) if len(D) > 0 else (X, None, y)

    def get_derived_data_slice(self, derived_data, indices):
        tmp_derived_data = {}
        for key, value in derived_data.items():
            tmp_derived_data[key] = value[indices, :]
        return tmp_derived_data

    def get_zero_slip(self, feature_name):
        if not hasattr(self, "dataprocessors"):
            raise Exception(f"Run load_config first.")
        elif len(self.dataprocessors) == 0 and feature_name in self.cont_feature_names:
            return 0
        if feature_name not in self.dataprocessors[-1][0].record_cont_features:
            raise Exception(f"Feature {feature_name} not available.")

        x = 0
        for processor, _ in self.dataprocessors:
            if hasattr(processor, "transformer"):
                x = processor.zero_slip(feature_name, x)
        return x

    def describe(self, transformed=False, save=True):
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

        if save:
            desc.to_csv(self.project_root + "describe.csv")
        return desc

    def train(
        self, programs: list = None, verbose: bool = False, debug_mode: bool = False
    ):
        if programs is None:
            modelbases_to_train = self.modelbases
        else:
            modelbases_to_train = [self.get_modelbase(x) for x in programs]

        for modelbase in modelbases_to_train:
            modelbase.train(verbose=verbose, debug_mode=debug_mode)

    def get_derived_data_sizes(self):
        return [x.shape for x in self.derived_data.values()]

    @staticmethod
    def _get_gini(tabular):
        return pd.DataFrame(
            data=np.array([[gini(tabular[x]) for x in tabular.columns]]),
            columns=tabular.columns,
            index=["Gini Index"],
        )

    @staticmethod
    def _get_mode(tabular):
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

    def divide_from_tabular_dataset(self, data: pd.DataFrame):
        feature_data = data[self.cont_feature_names]
        categorical_data = data[self.cat_feature_names]
        label_data = data[self.label_name]

        return feature_data, categorical_data, label_data

    def get_tabular_dataset(
        self, transformed=False
    ) -> Tuple[pd.DataFrame, list, list, list]:
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

    def cross_validation(
        self, programs, n_random, verbose, test_data_only, type="random"
    ):
        programs_predictions = {}
        for program in programs:
            programs_predictions[program] = {}
        if type == "random":
            set_data_handler = self.load_data
        else:
            raise Exception(f"{type} cross validation not implemented.")
        for i in range(n_random):
            if verbose:
                print(
                    f"----------------------------{i + 1}/{n_random} {type} cross validation----------------------------"
                )
            with HiddenPrints(disable_std=not verbose):
                if i != 0:
                    set_data_handler()
            for program in programs:
                modelbase = self.get_modelbase(program)
                modelbase.train(dump_trainer=True, verbose=verbose)
                predictions = modelbase._predict_all(
                    verbose=verbose, test_data_only=test_data_only
                )
                for model_name, value in predictions.items():
                    if model_name in programs_predictions[program].keys():
                        current_predictions = programs_predictions[program][model_name]

                        def append_once(key):
                            current_predictions[key] = (
                                np.append(
                                    current_predictions[key][0],
                                    value[key][0],
                                ),
                                np.append(
                                    current_predictions[key][1],
                                    value[key][1],
                                ),
                            )

                        append_once("Testing")
                        if not test_data_only:
                            append_once("Training")
                            append_once("Validation")
                    else:
                        programs_predictions[program][model_name] = value
            if verbose:
                print(
                    f"--------------------------End {i + 1}/{n_random} {type} cross validation--------------------------"
                )
        return programs_predictions

    def get_leaderboard(
        self,
        test_data_only: bool = False,
        dump_trainer=True,
        cross_validation=0,
        verbose=False,
    ) -> pd.DataFrame:
        """
        Run all baseline models and the model in this work for a leaderboard.
        :param test_data_only: False to get metrics on training and validation datasets. Default to True.
        :return: The leaderboard dataframe.
        """
        if cross_validation != 0:
            programs_predictions = self.cross_validation(
                programs=self.modelbases_names,
                n_random=cross_validation,
                verbose=verbose,
                test_data_only=test_data_only,
            )
        else:
            programs_predictions = {}
            for modelbase in self.modelbases:
                print(f"{modelbase.program} metrics")
                programs_predictions[modelbase.program] = modelbase._predict_all(
                    verbose=False, test_data_only=test_data_only
                )

        df_leaderboard = self._cal_leaderboard(
            programs_predictions, test_data_only=test_data_only
        )
        if dump_trainer:
            save_trainer(self)
        return df_leaderboard

    def _cal_leaderboard(
        self,
        programs_predictions,
        metrics=["rmse", "mse", "mae", "mape", "r2"],
        test_data_only=False,
    ):
        dfs = []
        for modelbase_name in self.modelbases_names:
            df = Trainer._metrics(
                programs_predictions[modelbase_name],
                metrics,
                test_data_only=test_data_only,
            )
            df["Program"] = modelbase_name
            dfs.append(df)

        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        df_leaderboard.sort_values(
            "Testing RMSE" if not test_data_only else "RMSE", inplace=True
        )
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard = df_leaderboard[["Program"] + list(df_leaderboard.columns)[:-1]]
        df_leaderboard.to_csv(self.project_root + "leaderboard.csv")
        self.leaderboard = df_leaderboard
        return df_leaderboard

    def plot_loss(self, train_ls, val_ls):
        """
        Plot loss-epoch while training.
        :return: None
        """
        plt.figure()
        plt.rcParams["font.size"] = 20
        ax = plt.subplot(111)
        ax.plot(
            np.arange(len(train_ls)),
            train_ls,
            label="Training loss",
            linewidth=2,
            color=clr[0],
        )
        ax.plot(
            np.arange(len(val_ls)),
            val_ls,
            label="Validation loss",
            linewidth=2,
            color=clr[1],
        )
        # minposs = val_ls.index(min(val_ls))+1
        # ax.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax.legend()
        ax.set_ylabel("MSE Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{self.loss.upper()} Loss")
        plt.savefig(self.project_root + "loss_epoch.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred(
        self, program: str = "ThisWork", log_trans: bool = True, upper_lim=9
    ):
        """
        Comparing ground truth and prediction for different models.
        :param program: Choose a program from 'autogluon', 'pytorch_tabular', 'TabNet'.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param upper_lim: The upper boundary of the plot. Default to 9.
        :return: None
        """
        modelbase = self.get_modelbase(program)
        model_names = modelbase.get_model_names()
        predictions = modelbase._predict_all()

        for idx, model_name in enumerate(model_names):
            print(model_name, f"{idx + 1}/{len(model_names)}")
            plt.figure()
            plt.rcParams["font.size"] = 14
            ax = plt.subplot(111)

            self._plot_truth_pred(
                predictions, ax, model_name, "Training", clr[0], log_trans=log_trans
            )
            if "Validation" in predictions[model_name].keys():
                self._plot_truth_pred(
                    predictions,
                    ax,
                    model_name,
                    "Validation",
                    clr[2],
                    log_trans=log_trans,
                )
            self._plot_truth_pred(
                predictions, ax, model_name, "Testing", clr[1], log_trans=log_trans
            )

            set_truth_pred(ax, log_trans, upper_lim=upper_lim)

            plt.legend(
                loc="upper left", markerscale=1.5, handlelength=0.2, handleheight=0.9
            )

            s = model_name.replace("/", "_")

            plt.savefig(self.project_root + f"{program}/{s}_truth_pred.pdf")
            if is_notebook():
                plt.show()

            plt.close()

    def cal_feature_importance(self, program, model_name, method="permutation"):
        from src.model.model import TorchModel

        modelbase = self.get_modelbase(program)

        if issubclass(type(modelbase), TorchModel):
            if method == "permutation":

                def forward_func(X, *D):
                    ground_truth = self.label_data.loc[
                        self.test_indices, :
                    ].values.flatten()
                    y = self.tensors[-1][self.test_indices, :]
                    loader = Data.DataLoader(
                        Data.TensorDataset(X, *D, y),
                        batch_size=len(y),
                        shuffle=False,
                    )
                    prediction, _, _ = modelbase._test_step(
                        modelbase.model[model_name], loader, self.loss_fn
                    )
                    loss = float(
                        self._metric_sklearn(ground_truth, prediction, self.loss)
                    )
                    return loss

                feature_perm = FeaturePermutation(forward_func)
                attr = [
                    x.cpu().numpy().flatten()
                    for x in feature_perm.attribute(
                        tuple(
                            [
                                self._get_first_tensor_slice(self.test_indices),
                                *self._get_additional_tensors_slice(self.test_indices),
                            ]
                        )
                    )
                ]
                attr = np.abs(np.append(attr[0], attr[1:]))
            elif method == "shap":
                shap_values, data = self.cal_shap(
                    program=program, model_name=model_name
                )
                attr = (
                    np.append(
                        np.mean(np.abs(shap_values[0]), axis=0),
                        [np.mean(np.abs(x), axis=0) for x in shap_values[1:]],
                    )
                    if type(shap_values) == list and len(shap_values) > 1
                    else np.mean(np.abs(shap_values[0]), axis=0)
                )
            else:
                raise NotImplementedError
            dims = self.get_derived_data_sizes()
            importance_names = cp(self.cont_feature_names)
            for key_idx, key in enumerate(self.derived_data.keys()):
                importance_names += (
                    [
                        f"{key} (dim {i})" if dims[key_idx][-1] > 1 else key
                        for i in range(dims[key_idx][-1])
                    ]
                    if key != "categorical"
                    else self.cat_feature_names
                )
        else:
            if method == "permutation":
                attr = np.zeros((len(self.all_feature_names),))
                test_data = self.df.loc[self.test_indices, :].copy()
                base_pred = modelbase.predict(
                    test_data,
                    derived_data=self.derive_unstacked(test_data),
                    model_name=model_name,
                )
                base_metric = self._metric_sklearn(
                    test_data[self.label_name].values, base_pred, metric="rmse"
                )
                for idx, feature in enumerate(self.all_feature_names):
                    df = test_data.copy()
                    df.loc[:, feature] = np.random.shuffle(df.loc[:, feature].values)
                    perm_pred = modelbase.predict(
                        df,
                        derived_data=self.derive_unstacked(df),
                        model_name=model_name,
                    )
                    attr[idx] = np.abs(
                        self._metric_sklearn(
                            df[self.label_name].values, perm_pred, metric="rmse"
                        )
                        - base_metric
                    )
                attr /= np.sum(attr)
            elif method == "shap":
                shap_values, data = self.cal_shap(
                    program=program, model_name=model_name
                )
                attr = (
                    np.append(
                        np.mean(np.abs(shap_values[0]), axis=0),
                        [np.mean(np.abs(x), axis=0) for x in shap_values[1:]],
                    )
                    if type(shap_values) == list and len(shap_values) > 1
                    else np.mean(np.abs(shap_values), axis=0)
                )
            else:
                raise NotImplementedError
            importance_names = cp(self.all_feature_names)

        return attr, importance_names

    def cal_shap(self, program, model_name):
        from src.model.model import TorchModel
        import shap

        modelbase = self.get_modelbase(program)

        if issubclass(type(modelbase), TorchModel):
            bk_indices = np.random.choice(self.train_indices, size=100, replace=False)
            X_train_bk = self._get_first_tensor_slice(bk_indices)
            D_train_bk = self._get_additional_tensors_slice(bk_indices)
            background_data = [X_train_bk, *D_train_bk]

            X_test = self._get_first_tensor_slice(self.test_indices)
            D_test = self._get_additional_tensors_slice(self.test_indices)
            test_data = [X_test, *D_test]
            explainer = shap.DeepExplainer(modelbase.model[model_name], background_data)

            shap_values = explainer.shap_values(test_data)
        else:
            background_data = shap.kmeans(
                self.df.loc[self.train_indices, self.all_feature_names], 10
            )

            def func(data):
                df = pd.DataFrame(columns=self.all_feature_names, data=data)
                return modelbase.predict(
                    df, model_name=model_name, derived_data=self.derive_unstacked(df)
                ).flatten()

            test_indices = np.random.choice(self.test_indices, size=10, replace=False)
            test_data = self.df.loc[test_indices, self.all_feature_names].copy()
            shap_values = shap.KernelExplainer(func, background_data).shap_values(
                test_data
            )
        return shap_values, test_data

    def plot_feature_importance(
        self, program, model_name, fig_size=(7, 4), method="permutation"
    ):
        """
        Calculate and plot permutation feature importance.
        :return: None
        """
        attr, names = self.cal_feature_importance(
            program=program, model_name=model_name, method=method
        )

        clr = sns.color_palette("deep")

        # if feature type is not assigned in config files, the feature is from dataderiver.
        pal = [
            clr[self.args["feature_names_type"][x]]
            if x in self.args["feature_names_type"].keys()
            else clr[self.args["feature_types"].index("Derived")]
            for x in self.cont_feature_names
        ]

        dims = self.get_derived_data_sizes()
        for key_idx, key in enumerate(self.derived_data.keys()):
            if key == "categorical":
                pal += [clr[self.args["feature_types"].index("Categorical")]] * dims[
                    key_idx
                ][-1]
            else:
                pal += [clr[self.args["feature_types"].index("Derived")]] * dims[
                    key_idx
                ][-1]

        clr_map = dict()
        for idx, feature_type in enumerate(self.args["feature_types"]):
            clr_map[feature_type] = clr[idx]

        plt.figure(figsize=fig_size)
        ax = plt.subplot(111)
        plot_importance(
            ax,
            names,
            attr,
            pal=pal,
            clr_map=clr_map,
            linewidth=1,
            edgecolor="k",
            orient="h",
        )
        if method == "permutation":
            ax.set_xlabel("Permutation feature importance")
        elif method == "shap":
            ax.set_xlabel("SHAP feature importance")
        else:
            ax.set_xlabel("Feature importance")
        plt.tight_layout()

        boxes = []
        import matplotlib

        for x in ax.get_children():
            if isinstance(x, matplotlib.patches.PathPatch):
                boxes.append(x)

        for patch, color in zip(boxes, pal):
            patch.set_facecolor(color)

        plt.savefig(
            self.project_root
            + f"feature_importance_{program}_{model_name}_{method}.png",
            dpi=600,
        )
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_dependence(
        self,
        program,
        model_name,
        refit=True,
        log_trans: bool = True,
        lower_lim=2,
        upper_lim=7,
        n_bootstrap=1,
        grid_size=30,
        CI=0.95,
        verbose=True,
    ):
        """
        Calculate and plot partial dependence plots.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param lower_lim: Lower limit of y-axis when plotting.
        :param upper_lim: Upper limit of y-axis when plotting.
        :return: None
        """
        modelbase = self.get_modelbase(program)

        (
            x_values_list,
            mean_pdp_list,
            ci_left_list,
            ci_right_list,
        ) = self.cal_partial_dependence(
            feature_subset=self.all_feature_names,
            program=program,
            model_name=model_name,
            df=self.df.loc[self.train_indices, :],
            derived_data=self.get_derived_data_slice(
                self.derived_data, self.train_indices
            ),
            n_bootstrap=n_bootstrap,
            refit=refit,
            grid_size=grid_size,
            verbose=verbose,
            rederive=True,
            percentile=80,
            CI=CI,
            average=True,
        )

        fig = plot_pdp(
            self.all_feature_names,
            self.cat_feature_names,
            self.cat_feature_mapping,
            x_values_list,
            mean_pdp_list,
            ci_left_list,
            ci_right_list,
            self.unscaled_feature_data,
            log_trans=log_trans,
            lower_lim=lower_lim,
            upper_lim=upper_lim,
        )

        plt.savefig(
            self.project_root + f"partial_dependence_{program}_{model_name}.pdf"
        )
        if is_notebook():
            plt.show()
        plt.close()

    def cal_partial_dependence(self, feature_subset=None, **kwargs):
        x_values_list = []
        mean_pdp_list = []
        ci_left_list = []
        ci_right_list = []

        for feature_idx, feature_name in enumerate(
            self.cont_feature_names if feature_subset is None else feature_subset
        ):
            if kwargs["verbose"]:
                print("Calculate PDP: ", feature_name)

            x_value, model_predictions, ci_left, ci_right = self._bootstrap(
                focus_feature=feature_name, **kwargs
            )

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)
            ci_left_list.append(ci_left)
            ci_right_list.append(ci_right)

        return x_values_list, mean_pdp_list, ci_left_list, ci_right_list

    def plot_partial_err(self, program, model_name, thres=0.8):
        """
        Calculate and plot partial error dependency for each feature.
        :param thres: Points with loss higher than thres will be marked.
        :return: None
        """
        modelbase = self.get_modelbase(program)

        ground_truth = self.label_data.loc[self.test_indices, :].values.flatten()
        prediction = modelbase.predict(
            df=self.df.loc[self.test_indices, :],
            derived_data=self.derive_unstacked(self.df.loc[self.test_indices, :]),
            model_name=model_name,
        ).flatten()
        plot_partial_err(
            self.df.loc[
                np.array(self.test_indices), self.all_feature_names
            ].reset_index(drop=True),
            cat_feature_names=self.cat_feature_names,
            cat_feature_mapping=self.cat_feature_mapping,
            truth=ground_truth,
            pred=prediction,
            thres=thres,
        )

        plt.savefig(self.project_root + f"partial_err_{program}_{model_name}.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def cal_corr(self, imputed=False, features_only=False):
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

    def get_not_imputed_df(self):
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

    def plot_corr(self, fontsize=10, cmap="bwr", imputed=False):
        """
        Plot Pearson correlation among features and the target.
        :return: None
        """
        cont_feature_names = self.cont_feature_names + self.label_name
        # sns.reset_defaults()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        plt.box(on=True)
        corr = self.cal_corr(imputed=imputed).values
        im = ax.imshow(corr, cmap=cmap)
        ax.set_xticks(np.arange(len(cont_feature_names)))
        ax.set_yticks(np.arange(len(cont_feature_names)))

        ax.set_xticklabels(cont_feature_names, fontsize=fontsize)
        ax.set_yticklabels(cont_feature_names, fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        norm_corr = corr - (np.max(corr) + np.min(corr)) / 2
        norm_corr /= np.max(norm_corr)

        for i in range(len(cont_feature_names)):
            for j in range(len(cont_feature_names)):
                text = ax.text(
                    j,
                    i,
                    round(corr[i, j], 2),
                    ha="center",
                    va="center",
                    color="w" if np.abs(norm_corr[i, j]) > 0.3 else "k",
                    fontsize=fontsize,
                )

        plt.tight_layout()
        plt.savefig(self.project_root + "corr.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def plot_pairplot(self, **kwargs):
        df_all = pd.concat(
            [self.unscaled_feature_data, self.unscaled_label_data], axis=1
        )
        sns.pairplot(df_all, corner=True, diag_kind="kde", **kwargs)
        plt.tight_layout()
        plt.savefig(self.project_root + "pair.jpg")
        if is_notebook():
            plt.show()
        plt.close()

    def plot_feature_box(self, imputed=False):
        # sns.reset_defaults()
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        bp = sns.boxplot(
            data=self.feature_data
            if imputed
            else self.get_not_imputed_df()[self.cont_feature_names],
            orient="h",
            linewidth=1,
            fliersize=4,
            flierprops={"marker": "o"},
        )

        boxes = []

        for x in ax.get_children():
            if isinstance(x, matplotlib.patches.PathPatch):
                boxes.append(x)

        color = "#639FFF"

        for patch in boxes:
            patch.set_facecolor(color)

        plt.grid(linewidth=0.4, axis="x")
        ax.set_axisbelow(True)
        plt.ylabel("Values (Standard Scaled)")
        # ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()
        plt.savefig(self.project_root + "feature_box.pdf")
        plt.show()
        plt.close()

    def _get_indices(self, partition="train"):
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

    def get_material_code(self, unique=False, partition="all"):
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

    def _select_by_material_code(self, m_code: str, partition="all"):
        code_df = self.get_material_code(unique=False, partition=partition)
        return code_df.index[np.where(code_df["Material_Code"] == m_code)[0]]

    def plot_data_split(self, bins=30, percentile="all"):
        from matplotlib.gridspec import GridSpec

        train_m_code = list(
            self.get_material_code(unique=True, partition="train")["Material_Code"]
        )
        val_m_code = list(
            self.get_material_code(unique=True, partition="val")["Material_Code"]
        )
        test_m_code = list(
            self.get_material_code(unique=True, partition="test")["Material_Code"]
        )
        all_m_code = list(self.get_material_code(unique=True)["Material_Code"])
        no_train_m_code = np.setdiff1d(all_m_code, train_m_code)
        val_only_m_code = np.setdiff1d(no_train_m_code, test_m_code)
        rest_m_code = np.setdiff1d(no_train_m_code, val_only_m_code)

        all_m_code = train_m_code + list(val_only_m_code) + list(rest_m_code)

        all_cycle = self.df[self.label_name].values.flatten()
        length = len(all_m_code)
        train_heat = np.zeros((length, bins))
        val_heat = np.zeros((length, bins))
        test_heat = np.zeros((length, bins))
        np.seterr(invalid="ignore")
        for idx, material in enumerate(all_m_code):
            cycle = all_cycle[
                self._select_by_material_code(m_code=material, partition="all")
            ]
            if percentile == "all":
                hist_range = (np.min(all_cycle), np.max(all_cycle))
            else:
                hist_range = (np.min(cycle), np.max(cycle))
            all_hist = np.histogram(cycle, bins=bins, range=hist_range)[0]

            def get_heat(partition):
                cycles = all_cycle[
                    self._select_by_material_code(m_code=material, partition=partition)
                ]
                return np.histogram(cycles, range=hist_range, bins=bins)[0] / all_hist

            train_heat[idx, :] = get_heat(partition="train")
            val_heat[idx, :] = get_heat(partition="val")
            test_heat[idx, :] = get_heat(partition="test")

        train_heat[np.isnan(train_heat)] = 0
        val_heat[np.isnan(val_heat)] = 0
        test_heat[np.isnan(test_heat)] = 0
        fig = plt.figure(figsize=(8, 2.5))
        gs = GridSpec(100, 100, figure=fig)

        def plot_im(heat, name, pos, hide_y_ticks=False):
            ax = fig.add_subplot(pos)
            im = ax.imshow(heat, aspect=bins / length, cmap="Oranges")
            if hide_y_ticks:
                ax.set_yticks([])
            ax.set_title(name)
            ax.set_xlim([-0.5, bins - 0.5])
            ax.set_xticks([0 - 0.5, (bins - 1) / 2, bins - 0.5])
            if percentile == "all":
                ax.set_xticklabels(
                    [
                        f"{x:.1f}"
                        for x in [hist_range[0], np.mean(hist_range), hist_range[1]]
                    ]
                )
            else:
                ax.set_xticklabels([0, 50, 100])
            return im

        plot_im(train_heat, "Training set", gs[:, 0:30], hide_y_ticks=False)
        plot_im(val_heat, "Validation set", gs[:, 33:63], hide_y_ticks=True)
        im = plot_im(test_heat, "Testing set", gs[:, 66:96], hide_y_ticks=True)
        # plt.colorbar(mappable=im)
        cax = fig.add_subplot(gs[50:98, 98:])
        cbar = plt.colorbar(cax=cax, mappable=im)
        cax.set_ylabel("Density")

        ax = fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        if percentile == "all":
            ax.set_xlabel(self.label_name[0])
        else:
            ax.set_xlabel(f"Percentile of {self.label_name[0]} for each material")
        ax.set_ylabel(f"ID of Material")
        plt.savefig(
            self.project_root
            + f"{self.datasplitter.__class__.__name__}_{percentile}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def plot_multiple_S_N(self, m_codes, hide_plt_show=True, **kwargs):
        for m_code in m_codes:
            print(m_code)
            if hide_plt_show:
                with HiddenPltShow():
                    self.plot_S_N(m_code=m_code, **kwargs)
            else:
                self.plot_S_N(m_code=m_code, **kwargs)

    def plot_S_N(
        self,
        s_col,
        n_col,
        r_col,
        m_code,
        r_value,
        load_dir="tension",
        ax=None,
        grid_size=30,
        n_bootstrap=1,
        CI=0.95,
        method="statistical",
        verbose=True,
        program="ThisWork",
        model_name="ThisWork",
        refit=True,
    ):
        # Check whether columns exist.
        if s_col not in self.df.columns:
            raise Exception(f"{s_col} not in features.")
        if n_col not in self.label_name:
            raise Exception(f"{n_col} is not the target.")

        # Find the selected material.
        m_train_indices = self._select_by_material_code(m_code, partition="train")
        m_test_indices = self._select_by_material_code(m_code, partition="test")
        m_val_indices = self._select_by_material_code(m_code, partition="val")
        original_indices = self._select_by_material_code(m_code, partition="all")

        if len(original_indices) == 0:
            raise Exception(f"Material_Code {m_code} not available.")

        sgn = 1 if load_dir == "tension" else -1
        m_train_indices = m_train_indices[
            (self.df.loc[m_train_indices, s_col] * sgn > 0)
            & ((self.df.loc[m_train_indices, r_col] - r_value).__abs__() < 1e-3)
        ]
        m_test_indices = m_test_indices[
            (self.df.loc[m_test_indices, s_col] * sgn > 0)
            & ((self.df.loc[m_test_indices, r_col] - r_value).__abs__() < 1e-3)
        ]
        m_val_indices = m_val_indices[
            (self.df.loc[m_val_indices, s_col] * sgn > 0)
            & ((self.df.loc[m_val_indices, r_col] - r_value).__abs__() < 1e-3)
        ]

        # If other parameters are not consistent, raise Warning.
        stress_unrelated_cols = [
            name for name in self.cont_feature_names if "Stress" not in name
        ]
        other_params = self.df.loc[
            np.concatenate(
                [m_train_indices.values, m_test_indices.values, m_val_indices.values]
            ),
            stress_unrelated_cols,
        ].copy()
        not_unique_cols = [
            (col, list(other_params[col].value_counts().index))
            for col in stress_unrelated_cols
            if len(other_params[col].value_counts()) > 1
        ]
        if len(not_unique_cols) != 0:
            message = (
                f"More than one values of each stress unrelated column are found {not_unique_cols}. Bootstrapped "
                f"prediction of SN curves may be incorrect."
            )
            if is_notebook():
                print(message)
            else:
                warnings.warn(message, UserWarning)

        # If no training or validation points available, raise an exception.
        if (
            len(m_train_indices) == 0
            and len(m_val_indices) == 0
            and len(m_test_indices) == 0
        ):
            unique_r = np.unique(self.df.loc[original_indices, r_col])
            available_r = []
            for r in unique_r:
                if (
                    (self.df.loc[original_indices, s_col] * sgn > 0)
                    & ((self.df.loc[original_indices, r_col] - r).__abs__() < 1e-3)
                ).any():
                    available_r.append(r)
            raise Exception(
                f"R-value {r_value} not available. Choose among {available_r}."
            )

        # Extract S and N.
        s_train = self.df.loc[m_train_indices, s_col]
        n_train = self.df.loc[m_train_indices, n_col]
        s_val = self.df.loc[m_val_indices, s_col]
        n_val = self.df.loc[m_val_indices, n_col]
        s_test = self.df.loc[m_test_indices, s_col]
        n_test = self.df.loc[m_test_indices, n_col]

        all_s = np.vstack(
            [
                s_train.values.reshape(-1, 1),
                s_val.values.reshape(-1, 1),
                s_test.values.reshape(-1, 1),
            ]
        )

        # Determine the prediction range.
        s_min = np.min(all_s) - np.abs(np.max(all_s) - np.min(all_s)) * 0.5
        s_max = np.max(all_s) + np.abs(np.max(all_s) - np.min(all_s)) * 0.5
        s_min = np.max([s_min, 1e-5]) if sgn > 0 else s_min
        s_max = s_max if sgn > 0 else np.min([s_max, -1e-5])

        # Get bootstrap predictions and confidence intervals from program-model_name
        chosen_indices = (
            m_train_indices
            if len(m_train_indices) != 0
            else np.append(m_val_indices, m_test_indices)
        )
        x_value, mean_pred, ci_left, ci_right = self._bootstrap(
            program=program,
            df=self.df.loc[chosen_indices, :],
            derived_data=self.get_derived_data_slice(self.derived_data, chosen_indices),
            focus_feature=s_col,
            n_bootstrap=n_bootstrap,
            grid_size=grid_size,
            x_min=s_min,
            x_max=s_max,
            CI=CI,
            average=False,
            verbose=verbose,
            model_name=model_name,
            refit=refit if len(m_train_indices) != 0 else False,
        )

        # Defining a series of utilities.
        def get_interval_psn(s, n, xvals, n_pred_vals=None):
            # Calculate predictions, intervals, and psn from lin-log or log-log S and N.
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            lr.fit(s.reshape(-1, 1), n.reshape(-1, 1))
            n_pred_interp = (
                lr.predict(xvals.reshape(-1, 1)).flatten()
                if n_pred_vals is None
                else n_pred_vals
            )
            n_pred = (
                lr.predict(s.reshape(-1, 1))
                if n_pred_vals is None
                else np.interp(s, xvals, n_pred_vals).reshape(-1, 1)
            )
            CL, CR = self._sn_interval(
                method=method,
                y=n,
                y_pred=n_pred,
                x=s,
                xvals=xvals,
                CI=CI,
            )
            ci_left, ci_right = n_pred_interp - CL, n_pred_interp + CR
            psn_CL = self._psn(
                method="iso",
                y=n,
                y_pred=n_pred,
                x=s,
                xvals=xvals,
                CI=CI,
                p=0.95,
            )
            psn_pred = n_pred_interp - psn_CL
            return n_pred_interp, ci_left, ci_right, psn_pred

        def scatter_plot_func(x, y, color, name):
            # Plot training, validation, and testing sets.
            ax.scatter(
                x,
                y,
                s=20,
                color=color,
                marker="o",
                label=f"{name} dataset",
                linewidth=0.4,
                edgecolors="k",
                zorder=20,
            )

        def in_fill_between(x_arr, y_arr, xvals, cl, cr):
            # Calculate the number of points that are in the interval.
            def point_in_fill_between(x, y):
                which_x = np.where(np.abs(x - xvals) == np.min(np.abs(x - xvals)))[0][0]
                cl_x = cl[which_x]
                cr_x = cr[which_x]
                return True if cl_x <= y <= cr_x else False

            res = []
            for x, y in zip(x_arr, y_arr):
                res.append(point_in_fill_between(x, y))
            return np.count_nonzero(np.array(res))

        def report(interv_left, interv_right):
            # Report the number of points that are in the interval for three sets.
            print(
                f"Training {in_fill_between(s_train, n_train, x_value, interv_left, interv_right)}/{len(s_train)}"
            )
            print(
                f"Validation {in_fill_between(s_val, n_val, x_value, interv_left, interv_right)}/{len(s_val)}"
            )
            print(
                f"Testing {in_fill_between(s_test, n_test, x_value, interv_left, interv_right)}/{len(s_test)}"
            )

        def interval_plot_func(pred, interv_left, interv_right, color, name):
            # Plot predictions and intervals.
            ax.plot(pred, x_value, color=color, zorder=10)
            ax.fill_betweenx(
                x_value,
                interv_left,
                interv_right,
                alpha=0.4,
                color=color,
                edgecolor=None,
                label=name,
                zorder=0,
            )
            print(name)
            report(interv_left, interv_right)

        def psn_plot_func(pred, color, name):
            # Plot psn
            ax.plot(pred, x_value, "--", color=color, label=name)

        if ax is None:
            new_ax = True
            plt.figure()
            plt.rcParams["font.size"] = 14
            ax = plt.subplot(111)
        else:
            new_ax = False

        # Plot datasets.
        scatter_plot_func(n_train, s_train, clr[0], "Training")
        scatter_plot_func(n_val, s_val, clr[1], "Validation")
        scatter_plot_func(n_test, s_test, clr[2], "Testing")

        # Plot predictions and intervals.
        if len(m_train_indices) != 0:
            _, ci_left, ci_right, psn_pred = get_interval_psn(
                s_train.values,
                n_train.values,
                x_value,
                n_pred_vals=mean_pred,
            )

            interval_plot_func(mean_pred, ci_left, ci_right, clr[1], f"{model_name} CI")
            psn_plot_func(psn_pred, color=clr[1], name=f"{model_name} 5\% PoF")
        else:
            if not (np.isnan(ci_left).any() or np.isnan(ci_right).any()):
                interval_plot_func(
                    mean_pred,
                    ci_left,
                    ci_right,
                    clr[1],
                    f"Bootstrap {model_name} CI {CI*100:.1f}\%",
                )
            else:
                ax.plot(mean_pred, x_value, color=clr[1], zorder=10)

        # Get predictions, intervals and psn for lin-log and log-log SN.
        # lin_pred, lin_ci_left, lin_ci_right, lin_psn_pred = get_interval_psn(
        #     s_train.values, n_train.values, x_value
        # )
        # log_pred, log_ci_left, log_ci_right, log_psn_pred = get_interval_psn(
        #     np.log10(s_train.values), n_train.values, np.log10(x_value)
        # )

        # Plot predictions, intervals and psn.
        # interval_plot_func(lin_pred, lin_ci_left, lin_ci_right, clr[0], f"Lin-log CI")

        # interval_plot_func(log_pred, log_ci_left, log_ci_right, clr[2], f"Log-log CI")

        # psn_plot_func(lin_psn_pred, color=clr[0], name=f"Lin-log 5\% PoF")
        # psn_plot_func(log_psn_pred, color=clr[2], name=f"Log-log 5\% PoF")

        ax.legend(loc="upper right", markerscale=1.5, handlelength=1, handleheight=0.9)
        ax.set_xlabel(n_col)
        ax.set_ylabel(s_col)
        ax.set_xlim([0, 10])
        ax.set_title(f"{m_code} R={r_value} CI={CI * 100:.1f}\%")

        path = f"{self.project_root}SN_curves_{program}_{model_name}"
        if not os.path.exists(path):
            os.mkdir(path=path)
        fig_name = m_code.replace("/", "_") + f"_r_{r_value}.pdf"
        plt.savefig(path + "/" + fig_name)

        if is_notebook() and new_ax:
            plt.show()
        if new_ax:
            plt.close()

    @staticmethod
    def _sn_interval(method, y, y_pred, x, xvals, CI):
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "statistical":
            # Schneider, C. R. A., and S. J. Maddox. "Best practice guide on statistical analysis of fatigue data."
            # Weld Inst Stat Rep (2003).
            # The two-sided prediction limits are symmetrical, so we calculate one-sided limit instead; therefore, in
            # st.t.ppf or st.f.ppf, the first probability argument is (CI+1)/2 instead of CI for one-sided prediction limit.
            # Because, for example for two-sided CI=95%, the lower limit is equivalent to one-sided 97.5% limit.
            tinv = st.t.ppf((CI + 1) / 2, n - 2)
            CL = tinv * STEYX * (1 + 1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            return CL.flatten(), CL.flatten()
        elif method == "astm":
            # According to ASTM E739-10(2015). x can be stress, log(stress), strain, log(strain), etc.
            # It is valid when y and x follow the linear assumption.
            # Barbosa, Joelton Fonseca, et al. "Probabilistic SN fields based on statistical distributions applied to
            # metallic and composite materials: State of the art." Advances in Mechanical Engineering 11.8 (2019):
            # 1687814019870395.
            # The first parameter is CI instead of (CI+1)/2 according to the ASTM standard. We verified have verified
            # this point by reproducing its given example in Section 8.3 using the following code:
            # from src.core.trainer import Trainer
            # import numpy as np
            # import scipy.stats as st
            # from sklearn.linear_model import LinearRegression
            # x = np.array([-1.78622, -1.79344, -2.17070, -2.16622, -2.74715, -2.79588, -2.78252, -3.27252, -3.26761])
            # y = np.array([2.22531, 2.30103, 3., 3.07188, 3.67486, 3.90499, 3.72049, 4.45662, 4.51388])
            # lr = LinearRegression()
            # lr.fit(x.reshape(-1, 1), y.reshape(-1,1))
            # y_pred = lr.predict(x.reshape(-1,1))
            # xvals = np.linspace(np.min(x), np.max(x), 100)
            # cl, cr = Trainer._sn_interval("ASTM", y, y_pred, x, xvals, 0.95)
            # print(xvals[-15]) #-1.9964038383838383
            # print(cl[-15]) #0.15220609531569082, comparable with given 0.15215. Differences might come from the
            # # regression coefficients.
            tinv = st.f.ppf(CI, 2, n - 2)
            CL = (
                np.sqrt(2 * tinv)
                * STEYX
                * (1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            )
            return CL.flatten(), CL.flatten()
        else:
            raise Exception(f"S-N interval type {method} not implemented.")

    @staticmethod
    def _psn(method, y, y_pred, x, xvals, CI, p):
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "iso":
            # ISO 12107
            def oneside_normal(p, CI, sample_size, n_random=100000, ddof=1):
                # The one-sided tolerance limits of normal distribution in ISO are given in a table. We find that the
                # analytical calculation is difficult to implement (https://statpages.info/tolintvl.html gives a
                # interactive implementation and a .xls file). We use Monte Carlo simulation to get a more precise
                # value. Since the value is calculated once per plot, the cost is affordable.
                # Refs:
                # https://stackoverflow.com/questions/63698305/how-to-calculate-one-sided-tolerance-interval-with-scipy
                # (or https://jekel.me/tolerance_interval_py/oneside/oneside.html)
                from scipy.stats import norm, nct

                p = 1 - p if p < 0.5 else p
                x_tmp = np.random.randn(n_random, sample_size)
                sigma_est = x_tmp.std(axis=1, ddof=ddof)
                zp = norm.ppf(p)
                t = nct.ppf(CI, df=sample_size - ddof, nc=np.sqrt(sample_size) * zp)
                k = t / np.sqrt(sample_size)
                return np.mean(k * sigma_est)

            k = oneside_normal(p=p, CI=CI, sample_size=n - 2)
            CL = k * STEYX * (1 + 1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            return CL.flatten()
        else:
            raise Exception(f"P-S-N type {method} not implemented.")

    def _bootstrap(
        self,
        program,
        df,
        derived_data,
        focus_feature,
        n_bootstrap,
        grid_size,
        verbose=True,
        rederive=True,
        refit=True,
        resample=True,
        percentile=100,
        x_min=None,
        x_max=None,
        CI=0.95,
        average=True,
        model_name="ThisWork",
    ):
        # Cook, Thomas R., et al. Explaining Machine Learning by Bootstrapping Partial Dependence Functions and Shapley
        # Values. No. RWP 21-12. 2021.
        modelbase = self.get_modelbase(program)
        derived_data = self.sort_derived_data(derived_data)
        if focus_feature in self.cont_feature_names:
            x_value = np.linspace(
                np.nanpercentile(df[focus_feature].values, (100 - percentile) / 2)
                if x_min is None
                else x_min,
                np.nanpercentile(df[focus_feature].values, 100 - (100 - percentile) / 2)
                if x_max is None
                else x_max,
                grid_size,
            )
        elif focus_feature in self.cat_feature_names:
            x_value = np.unique(df[focus_feature].values)
        else:
            raise Exception(f"{focus_feature} not available.")
        df = df.reset_index(drop=True)
        expected_value_bootstrap_replications = []
        for i_bootstrap in range(n_bootstrap):
            if resample:
                df_bootstrap = skresample(df)
            else:
                df_bootstrap = df
            tmp_derived_data = self.get_derived_data_slice(
                derived_data, list(df_bootstrap.index)
            )
            df_bootstrap = df_bootstrap.reset_index(drop=True)
            bootstrap_model = cp(modelbase)
            if refit:
                bootstrap_model.fit(
                    df_bootstrap,
                    model_subset=[model_name],
                    cont_feature_names=self.dataprocessors[0][0].record_cont_features,
                    cat_feature_names=self.dataprocessors[0][0].record_cat_features,
                    label_name=self.label_name,
                    verbose=False,
                    warm_start=True,
                )
            bootstrap_model_predictions = []
            for value in x_value:
                df_perm = df_bootstrap.copy()
                df_perm[focus_feature] = value
                bootstrap_model_predictions.append(
                    bootstrap_model.predict(
                        df_perm,
                        model_name=model_name,
                        derived_data=tmp_derived_data
                        if focus_feature in self.derived_stacked_features
                        else None,  # To avoid rederiving stacked data
                    )
                )
            if average:
                expected_value_bootstrap_replications.append(
                    np.mean(np.hstack(bootstrap_model_predictions), axis=0)
                )
            else:
                expected_value_bootstrap_replications.append(
                    np.hstack(bootstrap_model_predictions)
                )

        expected_value_bootstrap_replications = np.vstack(
            expected_value_bootstrap_replications
        )
        ci_left = []
        ci_right = []
        mean_pred = []
        for col_idx in range(expected_value_bootstrap_replications.shape[1]):
            y_pred = expected_value_bootstrap_replications[:, col_idx]
            if len(y_pred) != 1 and len(np.unique(y_pred)) != 1:
                ci_int = st.norm.interval(
                    alpha=CI, loc=np.mean(y_pred), scale=np.std(y_pred)
                )
            else:
                ci_int = (np.nan, np.nan)
            ci_left.append(ci_int[0])
            ci_right.append(ci_int[1])
            mean_pred.append(np.mean(y_pred))

        return x_value, np.array(mean_pred), np.array(ci_left), np.array(ci_right)

    def _get_best_model(self):
        if not hasattr(self, "leaderboard"):
            self.get_leaderboard(test_data_only=True, dump_trainer=False)
        return (
            self.leaderboard["Program"].values[0],
            self.leaderboard["Model"].values[0],
        )

    @staticmethod
    def _metrics(predictions, metrics, test_data_only):
        df_metrics = pd.DataFrame()
        for model_name, model_predictions in predictions.items():
            df = pd.DataFrame(index=[0])
            df["Model"] = model_name
            for tvt, (y_pred, y_true) in model_predictions.items():
                if test_data_only and tvt != "Testing":
                    continue
                for metric in metrics:
                    metric_value = Trainer._metric_sklearn(y_true, y_pred, metric)
                    df[
                        tvt + " " + metric.upper()
                        if not test_data_only
                        else metric.upper()
                    ] = metric_value
            df_metrics = pd.concat([df_metrics, df], axis=0, ignore_index=True)

        return df_metrics

    @staticmethod
    def _metric_sklearn(y_true, y_pred, metric):
        if metric == "mse":
            from sklearn.metrics import mean_squared_error

            return mean_squared_error(y_true, y_pred)
        elif metric == "rmse":
            from sklearn.metrics import mean_squared_error

            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            from sklearn.metrics import mean_absolute_error

            return mean_absolute_error(y_true, y_pred)
        elif metric == "mape":
            from sklearn.metrics import mean_absolute_percentage_error

            return mean_absolute_percentage_error(y_true, y_pred)
        elif metric == "r2":
            from sklearn.metrics import r2_score

            return r2_score(y_true, y_pred)
        else:
            raise Exception(f"Metric {metric} not implemented.")

    def _plot_truth_pred(
        self,
        predictions,
        ax,
        model_name,
        name,
        color,
        marker="o",
        log_trans=True,
        verbose=True,
    ):
        pred_y, y = predictions[model_name][name]
        r2 = Trainer._metric_sklearn(y, pred_y, "r2")
        loss = self.loss_fn(torch.Tensor(y), torch.Tensor(pred_y))
        if verbose:
            print(f"{name} Loss: {loss:.4f}, R2: {r2:.4f}")
        ax.scatter(
            10**y if log_trans else y,
            10**pred_y if log_trans else pred_y,
            s=20,
            color=color,
            marker=marker,
            label=f"{name} dataset ($R^2$={r2:.3f})",
            linewidth=0.4,
            edgecolors="k",
        )
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")

    def _get_additional_tensors_slice(self, indices):
        res = []
        for tensor in self.tensors[1 : len(self.tensors) - 1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)

    def _get_first_tensor_slice(self, indices):
        return self.tensors[0][indices, :]

    def get_base_predictor(
        self,
        categorical=True,
        **kwargs,
    ):
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


def save_trainer(trainer, path=None, verbose=True):
    import pickle

    path = trainer.project_root + "trainer.pkl" if path is None else path
    if verbose:
        print(
            f"Trainer saved. To load the trainer, run trainer = load_trainer(path='{path}')"
        )
    with open(path, "wb") as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)


def load_trainer(path=None):
    import pickle

    with open(path, "rb") as inp:
        trainer = pickle.load(inp)
    return trainer
