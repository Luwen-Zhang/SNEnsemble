"""
The basic class for the project. It includes configuration, data processing, plotting,
and comparing baseline models.
"""
from typing import *
import os.path
from ..utils.utils import *
import torch
from torch import nn
from torch.utils.data import Subset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from captum.attr import FeaturePermutation
import sys
from importlib import import_module, reload
from skopt.space import Real, Integer, Categorical
import torch.utils.data as Data
import time
import json
from copy import deepcopy as cp
from sklearn.utils import resample as skresample
import scipy.stats as st
import itertools

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
        from src.core.datasplitter import get_data_splitter

        self.datasplitter = get_data_splitter(name)()

    def set_data_processors(self, config: List[Tuple], verbose=True):
        from src.core.dataprocessor import get_data_processor

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
        from src.core.dataderiver import get_data_deriver

        self.dataderivers = [
            (get_data_deriver(name)(), kwargs) for name, kwargs in config
        ]

    def load_data(self, data_path: str = None) -> None:
        """
        Load the data file in ../data directory specified by the 'project' argument in configfile. Data will be splitted
         into training, validation, and testing datasets.
        :param data_path: specify a data file different from 'project'. Default to None.
        :return: None
        """
        if data_path is None:
            data_path = f"data/{self.database}.xlsx"
            self.df = pd.read_excel(data_path, engine="openpyxl")
        else:
            self.df = pd.read_excel(data_path, engine="openpyxl")
        self.data_path = data_path

        feature_names = list(self.args["feature_names_type"].keys())
        label_name = self.args["label_name"]

        self.set_data(self.df, feature_names, label_name)
        print(
            "Dataset size:",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        self.save_data()

    def set_data(
        self,
        df,
        feature_names,
        label_name,
        derived_data=None,
        warm_start=False,
        verbose=True,
        all_training=False,
        train_indices=None,
        val_indices=None,
        test_indices=None,
    ):
        self.feature_names = feature_names
        self.label_name = label_name
        if derived_data is None:
            (
                df,
                derived_data,
                self.feature_names,
                self.derived_data_col_names,
                self.derivation_related_cols,
                self.stacked_derivation_related_cols,
            ) = self.derive(df)
        self.df = df
        self.derived_data = derived_data
        self.df = self.df.copy().dropna(
            axis=0, subset=self.label_name + self.derivation_related_cols
        )
        if all_training:
            indices = np.arange(len(df))
            self._data_process(
                train_indices=indices,
                val_indices=indices,
                test_indices=indices,
                warm_start=warm_start,
                verbose=verbose,
            )
        else:
            self._data_process(
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                warm_start=warm_start,
                verbose=verbose,
            )
        self._rederive_unstacked()
        self._update_dataset_auto()
        self._material_code = (
            pd.DataFrame(self.df["Material_Code"])
            if "Material_Code" in self.df.columns
            else None
        )

    def save_data(self, path: str = None):
        if path is None:
            path = self.project_root

        self.df.to_csv(os.path.join(path, "data.csv"), encoding="utf-8", index=False)
        tabular_data, _, _ = self._get_tabular_dataset()
        tabular_data.to_csv(
            os.path.join(path, "tabular_data.csv"), encoding="utf-8", index=False
        )

        print(f"Data saved to {path} (data.csv and tabular_data.csv).")

    def _rederive_unstacked(self):
        for deriver, kwargs in self.dataderivers:
            if kwargs["derived_name"] in self.derived_data.keys():
                value, name, col_names, _, _, _ = deriver.derive(self.df, **kwargs)
                self.derived_data[name] = value
                self.derived_data_col_names[name] = col_names
                if len(self.derived_data[name]) == 0:
                    self.derived_data_col_names.pop(name, None)
                    self.derived_data.pop(name, None)

    def _data_process(
        self,
        train_indices=None,
        val_indices=None,
        test_indices=None,
        warm_start=False,
        verbose=True,
    ):
        self.feature_data = self.df[self.feature_names]
        self.label_data = self.df[self.label_name]
        self.unscaled_feature_data = cp(self.feature_data)
        self.unscaled_label_data = cp(self.label_data)
        _, feature_names, label_name = self._get_tabular_dataset()
        self.df.reset_index(drop=True, inplace=True)
        original_length = len(self.df)

        if train_indices is None or val_indices is None or test_indices is None:
            (
                self.train_indices,
                self.val_indices,
                self.test_indices,
            ) = self.datasplitter.split(self.df, feature_names, label_name)
        else:
            self.train_indices, self.val_indices, self.test_indices = (
                train_indices,
                val_indices,
                test_indices,
            )

        if verbose:
            data = self._data_preprocess(self.df, warm_start=warm_start)
        else:
            with HiddenPrints():
                data = self._data_preprocess(self.df, warm_start=warm_start)

        # Reset indices
        self.retained_indices = np.array(data.index)
        self.dropped_indices = np.setdiff1d(
            np.arange(original_length), self.retained_indices
        )

        def update_indices(indices):
            return np.array(
                [
                    x - np.count_nonzero(self.dropped_indices < x)
                    for x in indices
                    if x in data.index
                ]
            )

        self.train_indices = update_indices(self.train_indices)
        self.test_indices = update_indices(self.test_indices)
        self.val_indices = update_indices(self.val_indices)

        self.df = pd.DataFrame(self.df.loc[self.retained_indices, :]).reset_index(
            drop=True
        )

        if (
            len(self.train_indices) == 0
            or len(self.val_indices) == 0
            or len(self.test_indices) == 0
        ):
            print(
                f"No sufficient data after preprocessing. Splitting datasets again which might make data transformation"
                f" not reasonable."
            )
            (
                self.train_indices,
                self.val_indices,
                self.test_indices,
            ) = self.datasplitter.split(self.df, feature_names, label_name)

        # feature_data and label_data does not contain derived data.
        self.feature_data, self.label_data = self._divide_from_tabular_dataset(data)

    def _data_preprocess(self, input_data: pd.DataFrame, warm_start=False):
        data = input_data.copy()
        for processor, kwargs in self.dataprocessors:
            if warm_start:
                data = processor.transform(data, self, **kwargs)
            else:
                data = processor.fit_transform(data, self, **kwargs)
        data = data[self.feature_names + self.label_name]
        return data

    def _data_transform(self, input_data: pd.DataFrame):
        data = input_data.copy()
        from src.core.dataprocessor import AbstractTransformer

        for processor, kwargs in self.dataprocessors:
            if issubclass(type(processor), AbstractTransformer):
                data = processor.transform(data, self, **kwargs)
        data = data[self.feature_names + self.label_name]
        return data

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

    def get_zero_slip(self, feature_name):
        if not hasattr(self, "dataprocessors"):
            raise Exception(f"Run load_config first.")
        elif len(self.dataprocessors) == 0 and feature_name in self.feature_names:
            return 0
        if feature_name not in self.dataprocessors[-1][0].record_features:
            raise Exception(f"Feature {feature_name} not available.")

        x = 0
        for processor, _ in self.dataprocessors:
            if hasattr(processor, "transformer"):
                x = processor.zero_slip(feature_name, x)
        return x

    def describe(self, transformed=False, save=True):
        tabular = self._get_tabular_dataset(transformed=transformed)[0]
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

    def _get_derived_data_sizes(self):
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

    def _divide_from_tabular_dataset(self, data: pd.DataFrame):
        feature_data = data[self.feature_names].reset_index(drop=True)
        label_data = data[self.label_name].reset_index(drop=True)

        return feature_data, label_data

    def _get_tabular_dataset(
        self, transformed=False
    ) -> Tuple[pd.DataFrame, list, list]:
        if transformed:
            feature_data = self.feature_data
        else:
            feature_data = self.unscaled_feature_data

        feature_names = cp(self.feature_names)
        label_name = cp(self.label_name)

        tabular_dataset = pd.concat([feature_data, self.label_data], axis=1)

        return tabular_dataset, feature_names, label_name

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
        model_names = modelbase._get_model_names()
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

    def plot_feature_importance(self, modelbase, fig_size=(7, 4)):
        """
        Calculate and plot permutation feature importance.
        :return: None
        """
        from src.core.model import TorchModel

        if not issubclass(type(modelbase), TorchModel):
            raise Exception("A TorchModel should be passed.")

        def forward_func(data):
            prediction, ground_truth, loss = test_tensor(
                data,
                self._get_additional_tensors_slice(self.test_dataset.indices),
                self.tensors[-1][self.test_dataset.indices, :],
                modelbase.model,
                self.loss_fn,
            )
            return loss

        feature_perm = FeaturePermutation(forward_func)
        attr = (
            feature_perm.attribute(self.tensors[0][self.test_dataset.indices, :])
            .cpu()
            .numpy()[0]
        )

        clr = sns.color_palette("deep")

        # if feature type is not assigned in config files, the feature is from dataderiver.
        pal = [
            clr[self.args["feature_names_type"][x]]
            if x in self.args["feature_names_type"].keys()
            else clr[len(self.args["feature_types"]) - 1]
            for x in self.feature_names
        ]

        clr_map = dict()
        for idx, feature_type in enumerate(self.args["feature_types"]):
            clr_map[feature_type] = clr[idx]

        plt.figure(figsize=fig_size)
        ax = plt.subplot(111)
        plot_importance(
            ax,
            self.feature_names,
            attr,
            pal=pal,
            clr_map=clr_map,
            linewidth=1,
            edgecolor="k",
            orient="h",
        )
        plt.tight_layout()

        boxes = []
        import matplotlib

        for x in ax.get_children():
            if isinstance(x, matplotlib.patches.PathPatch):
                boxes.append(x)

        for patch, color in zip(boxes, pal):
            patch.set_facecolor(color)

        plt.savefig(self.project_root + "feature_importance.png", dpi=600)
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_dependence(
        self,
        modelbase,
        model_name,
        log_trans: bool = True,
        lower_lim=2,
        upper_lim=7,
        n_bootstrap=1,
        grid_size=30,
        CI=0.95,
    ):
        """
        Calculate and plot partial dependence plots.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param lower_lim: Lower limit of y-axis when plotting.
        :param upper_lim: Upper limit of y-axis when plotting.
        :return: None
        """
        from src.core.model import TorchModel

        if not issubclass(type(modelbase), TorchModel):
            raise Exception("A TorchModel should be passed.")

        x_values_list = []
        mean_pdp_list = []
        ci_left_list = []
        ci_right_list = []

        for feature_idx, feature_name in enumerate(self.feature_names):
            print("Calculate PDP: ", feature_name)

            x_value, model_predictions, ci_left, ci_right = self._bootstrap(
                model=modelbase,
                model_name=model_name,
                df=self.df.loc[self.train_indices, :],
                focus_feature=feature_name,
                n_bootstrap=n_bootstrap,
                grid_size=grid_size,
                verbose=False,
                rederive=True,
                percentile=80,
                CI=CI,
                average=True,
            )

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)
            ci_left_list.append(ci_left)
            ci_right_list.append(ci_right)

        fig = plot_pdp(
            self.feature_names,
            x_values_list,
            mean_pdp_list,
            ci_left_list,
            ci_right_list,
            self.unscaled_feature_data,
            log_trans=log_trans,
            lower_lim=lower_lim,
            upper_lim=upper_lim,
        )

        plt.savefig(self.project_root + "partial_dependence.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_err(self, modelbase, thres=0.8):
        """
        Calculate and plot partial error dependency for each feature.
        :param thres: Points with loss higher than thres will be marked.
        :return: None
        """
        from src.core.model import TorchModel

        if not issubclass(type(modelbase), TorchModel):
            raise Exception("A TorchModel should be passed.")
        prediction, ground_truth, loss = test_tensor(
            self.tensors[0][self.test_dataset.indices, :],
            self._get_additional_tensors_slice(self.test_dataset.indices),
            self.tensors[-1][self.test_dataset.indices, :],
            modelbase.model,
            self.loss_fn,
        )
        plot_partial_err(
            self.feature_data.loc[np.array(self.test_dataset.indices), :].reset_index(
                drop=True
            ),
            ground_truth,
            prediction,
            thres=thres,
        )

        plt.savefig(self.project_root + "partial_err.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def plot_corr(self, fontsize=10, cmap="bwr"):
        """
        Plot Pearson correlation among features and the target.
        :return: None
        """
        feature_names = self.feature_names + self.label_name
        # sns.reset_defaults()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        plt.box(on=True)
        df_all = pd.concat([self.feature_data, self.label_data], axis=1)
        corr = df_all.corr().values
        im = ax.imshow(corr, cmap=cmap)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))

        ax.set_xticklabels(feature_names, fontsize=fontsize)
        ax.set_yticklabels(feature_names, fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        norm_corr = corr - (np.max(corr) + np.min(corr)) / 2
        norm_corr /= np.max(norm_corr)

        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
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

    def plot_feature_box(self):
        # sns.reset_defaults()
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        bp = sns.boxplot(
            data=self.feature_data,
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
                f"Partition {train} not available. Select among {list(indices_map.keys())}"
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
        if m_code not in code_df["Material_Code"].values:
            raise Exception(f"Material code {m_code} not available.")
        return code_df.index[np.where(code_df["Material_Code"] == m_code)[0]]

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
        original_train_indices = cp(m_train_indices)
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
            name for name in self.feature_names if "Stress" not in name
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

        # If no training points available, raise an exception.
        if len(m_train_indices) == 0:
            unique_r = np.unique(self.df.loc[original_train_indices, r_col])
            available_r = []
            for r in unique_r:
                if (
                    (self.df.loc[original_train_indices, s_col] * sgn > 0)
                    & (
                        (self.df.loc[original_train_indices, r_col] - r).__abs__()
                        < 1e-3
                    )
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
        model = self.get_modelbase(program=program)
        x_value, mean_pred, ci_left, ci_right = self._bootstrap(
            model=model,
            df=self.df.loc[m_train_indices, :],
            focus_feature=s_col,
            n_bootstrap=n_bootstrap,
            grid_size=grid_size,
            x_min=s_min,
            x_max=s_max,
            CI=CI,
            average=False,
            verbose=verbose,
            model_name=model_name,
            refit=refit,
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
        # if not (np.isnan(ci_left).any() or np.isnan(ci_right).any()):
        #     interval_plot_func(
        #         mean_pred, ci_left, ci_right, clr[1], f"Bootstrap CI {CI*100:.1f}\%"
        #     )
        # else:
        #     ax.plot(mean_pred, x_value, color=clr[1], zorder=10)

        _, ci_left, ci_right, psn_pred = get_interval_psn(
            s_train.values,
            n_train.values,
            x_value,
            n_pred_vals=mean_pred,
        )

        interval_plot_func(mean_pred, ci_left, ci_right, clr[1], f"{model_name} CI")
        psn_plot_func(psn_pred, color=clr[1], name=f"{model_name} 5\% PoF")

        # Get predictions, intervals and psn for lin-log and log-log SN.
        lin_pred, lin_ci_left, lin_ci_right, lin_psn_pred = get_interval_psn(
            s_train.values, n_train.values, x_value
        )
        log_pred, log_ci_left, log_ci_right, log_psn_pred = get_interval_psn(
            np.log10(s_train.values), n_train.values, np.log10(x_value)
        )

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

    def derive(self, df):
        feature_names = cp(self.feature_names)
        derived_data = {}
        derived_data_col_names = {}
        derivation_related_cols = []
        stacked_derivation_related_cols = []
        for deriver, kwargs in self.dataderivers:
            try:
                (
                    value,
                    name,
                    col_names,
                    stacked,
                    intermediate,
                    related_columns,
                ) = deriver.derive(df, **kwargs)
            except Exception as e:
                print(
                    f"Skip deriver {deriver.__class__.__name__} because of the following exception:"
                )
                print(f"\t{e}")
                continue
            if not stacked:
                derived_data[name] = value
                derived_data_col_names[name] = col_names
                derivation_related_cols += related_columns
            else:
                if not intermediate:
                    for col_name in col_names:
                        if col_name not in feature_names:
                            feature_names.append(col_name)
                df[col_names] = value
                stacked_derivation_related_cols += related_columns

        return (
            df,
            derived_data,
            feature_names,
            derived_data_col_names,
            derivation_related_cols,
            stacked_derivation_related_cols,
        )

    def _bootstrap(
        self,
        model,
        df,
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
        from src.core.model import TorchModel

        if not issubclass(type(model), TorchModel):
            raise Exception(f"Model {type(model)} is not a TorchModel.")
        x_value = np.linspace(
            np.nanpercentile(df[focus_feature].values, (100 - percentile) / 2)
            if x_min is None
            else x_min,
            np.nanpercentile(df[focus_feature].values, 100 - (100 - percentile) / 2)
            if x_max is None
            else x_max,
            grid_size,
        )

        expected_value_bootstrap_replications = []
        for i_bootstrap in range(n_bootstrap):
            if n_bootstrap != 1 and verbose:
                print(f"Bootstrap: {i_bootstrap + 1}/{n_bootstrap}")
            if resample:
                df_bootstrap = skresample(df).reset_index(drop=True)
            else:
                df_bootstrap = df.reset_index(drop=True)
            bootstrap_model = cp(model)
            if refit:
                bootstrap_model.fit(
                    df_bootstrap,
                    self.dataprocessors[0][0].record_features,
                    self.label_name,
                    verbose=False,
                    warm_start=True,
                )
            bootstrap_model_predictions = []
            for value in x_value:
                df_perm = df_bootstrap.copy()
                df_perm[focus_feature] = value
                bootstrap_model_predictions.append(
                    bootstrap_model.predict(df_perm, model_name=model_name)
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
