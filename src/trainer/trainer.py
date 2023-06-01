"""
The basic class for the project. It includes configuration, data processing, plotting,
and comparing baseline models.
"""
import os.path
import src
from src.utils import *
from src.config import UserConfig
from src.data import DataModule
from copy import deepcopy as cp
from skopt.space import Real, Integer, Categorical
import time
from typing import *
import torch.nn as nn
import torch.cuda
import torch.utils.data as Data
import scipy.stats as st
from captum.attr import FeaturePermutation
from sklearn.utils import resample as skresample
import argparse
import platform, psutil, subprocess
import shutil
import pickle

set_random_seed(src.setting["random_seed"])
sys.path.append("configs/")


class Trainer:
    def __init__(self, device: str = "cpu", project: str = None):
        """
        The bridge of all modules. It contains all configurations and data. It can analyze the dataset (correlation,
        description, etc.), train model-bases, and evaluate results (including feature importance, partial dependency,
        etc.).

        Parameters
        ----------
        device:
            Device on which models are trained. Choose from "cuda" and "cpu".
        project:
            The name of the trainer.
        """
        self.device = "cpu"
        self.project = project
        self.modelbases = []
        self.modelbases_names = []
        self.set_device(device)

    def set_device(self, device: str):
        """
        Set the device for model bases.

        Parameters
        ----------
        device
            "cpu" or "cuda"

        Notes
        ----------
        Multi-GPU training and training on a machine with multiple GPUs are not tested.
        """
        if device not in ["cpu", "cuda"]:
            raise Exception(
                f"Device {device} is an invalid selection. Choose among {['cpu', 'cuda']}."
                f"Note: Multi-GPU training and training on a machine with multiple GPUs are not tested."
            )
        self.device = device
        for modelbase in self.modelbases:
            modelbase.device = device

    def add_modelbases(self, models: List):
        """
        Add a list of model-bases and check whether their names conflict.

        Parameters
        ----------
        models:
            A list of AbstractModels.
        """
        self.modelbases += models
        self.modelbases_names = [x.program for x in self.modelbases]
        if len(self.modelbases_names) != len(list(set(self.modelbases_names))):
            raise Exception(f"Conflicted modelbase names: {self.modelbases_names}")

    def get_modelbase(self, program: str):
        """
        Get the selected modelbase by its name.

        Parameters
        ----------
        program
            The name of the modelbase.

        Returns
        -------
            An instance of AbstractModel.
        """
        if program not in self.modelbases_names:
            raise Exception(f"Program {program} not added to the trainer.")
        return self.modelbases[self.modelbases_names.index(program)]

    def clear_modelbase(self):
        """
        Delete all model bases in the trainer.
        """
        self.modelbases = []
        self.modelbases_names = []

    def detach_modelbase(self, program: str, verbose: bool = True) -> "Trainer":
        """
        Detach the selected modelbase to a separate trainer and save it to another directory. It is much cheaper than
        ``Trainer.copy()`` if only one model base is needed.

        Parameters
        ----------
        program
            The selected modelbase.
        verbose
            Verbosity

        Returns
        -------
        trainer
            An ``Trainer`` instance.

        See Also
        -------
        ``Trainer.copy``, ``Trainer.detach_model``, ``AbstractModel.detach_model``
        """
        modelbase = cp(self.get_modelbase(program=program))
        tmp_trainer = modelbase.trainer
        tmp_trainer.clear_modelbase()
        new_path = add_postfix(self.project_root)
        tmp_trainer.set_path(new_path, verbose=False)
        modelbase.set_path(os.path.join(new_path, modelbase.program))
        tmp_trainer.add_modelbases([modelbase])
        shutil.copytree(self.get_modelbase(program=program).root, modelbase.root)
        save_trainer(tmp_trainer, verbose=verbose)
        return tmp_trainer

    def detach_model(
        self, program: str, model_name: str, verbose: bool = True
    ) -> "Trainer":
        """
        Detach the selected model of the selected model base to a separate trainer and save it to another directory.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model.
        verbose
            Verbosity.

        Returns
        -------
        trainer
            An ``Trainer`` instance.
        """
        tmp_trainer = self.detach_modelbase(program=program, verbose=False)
        tmp_modelbase = tmp_trainer.get_modelbase(program=program)
        detached_model = tmp_modelbase.detach_model(
            model_name=model_name, program=f"{program}_{model_name}"
        )
        tmp_trainer.clear_modelbase()
        tmp_trainer.add_modelbases([detached_model])
        shutil.rmtree(tmp_modelbase.root)
        save_trainer(tmp_trainer, verbose=verbose)
        return tmp_trainer

    def copy(self) -> "Trainer":
        """
        Copy the trainer and save it to another directory. It might be time and space consuming because all model bases
        are copied as well.

        Returns
        -------
        trainer
            A ``Trainer`` instance.

        See Also
        -------
        ``Trainer.detach_modelbase``, ``Trainer.detach_model``, ``AbstractModel.detach_model``
        """
        tmp_trainer = cp(self)
        new_path = add_postfix(self.project_root)
        tmp_trainer.set_path(new_path, verbose=True)
        for modelbase in tmp_trainer.modelbases:
            modelbase.set_path(os.path.join(new_path, modelbase.program))
        shutil.copytree(self.project_root, tmp_trainer.project_root, dirs_exist_ok=True)
        save_trainer(tmp_trainer)
        return tmp_trainer

    def load_config(
        self,
        config: Union[str, UserConfig] = None,
        verbose: bool = True,
        manual_config: Dict = None,
        project_root_subfolder: str = None,
    ) -> None:
        """
        Load a config in json format.
        Arguments passed to python when executing the script are parsed if ``configfile_path`` is left None. All keys in
        ``src.config.UserConfig().available_keys()`` can be parsed, for example:
            For the loss function: ``--loss mse``,

            For the total epoch: ``--epoch 200``,

            For the option for bayes opt: ``--bayes_opt`` to turn on bayes opt, ``--no-bayes_opt`` to turn off.

        Default values can be seen in ``src.config.UserConfig().defaults()``.

        The loaded configuration will be saved in the project folder.

        Parameters
        ----------
        config
            Can be the path to the config in json or python format, or a UserConfig instance.
            If it is a path. Arguments passed to python will be parsed; therefore, do not leave it empty when
            ``argparse.ArgumentParser`` is used for other purposes. If the path does not contain "/" or is not a file,
            the file ``configs/{config}``(.json/.py) will be read. The path can end with or without .json/.py.
        verbose
            Verbosity.
        manual_config
            Set configurations after the config file is loaded with a dict. For example:
            ``manual_config={"bayes_opt": True}``
        project_root_subfolder
            The subfolder that the project will locate in. The folder name will be
            ``{PATH OF THE MAIN SCRIPT}/output/{project}/{project_root_subfolder}/{TIME OF EXECUTION}-{configfile_path}``
        """
        input_config = config is not None
        if isinstance(config, str) or not input_config:
            base_config = UserConfig()
            # The base config is loaded using the --base argument
            if is_notebook() and not input_config:
                raise Exception(
                    "A config file must be assigned in notebook environment."
                )
            elif is_notebook() or input_config:
                parse_res = {"base": config}
            else:  # not notebook and config is None
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
                        parser.add_argument(
                            f"--no-{key}", dest=key, action="store_false"
                        )
                        parser.set_defaults(**{key: base_config[key]})
                parse_res = parser.parse_args().__dict__

            self.configfile = parse_res["base"]
            config = UserConfig(path=self.configfile)
            # Then, several args can be modified using other arguments like --lr, --weight_decay
            # only when a config file is not given so that configs depend on input arguments.
            if not is_notebook() and not input_config:
                config.merge(parse_res)
            if manual_config is not None:
                config.merge(manual_config)
            self.args = config
        else:
            self.configfile = "UserInputConfig"
            if manual_config is not None:
                warnings.warn(f"manual_config is ignored when config is an UserConfig.")
            self.args = config

        self.datamodule = DataModule(self.args, verbose=verbose)

        self.project = self.args["database"] if self.project is None else self.project
        self._create_dir(project_root_subfolder=project_root_subfolder)
        config.to_file(os.path.join(self.project_root, "args.py"))

    @property
    def static_params(self):
        return {
            "patience": self.args["patience"],
            "epoch": self.args["epoch"],
        }

    @property
    def chosen_params(self):
        return {
            "lr": self.args["lr"],
            "weight_decay": self.args["weight_decay"],
            "batch_size": self.args["batch_size"],
        }

    def get_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.args["loss"] == "mse":
            return nn.MSELoss()
        elif self.args["loss"] == "r2":
            return r2_loss
        elif self.args["loss"] == "mae":
            return nn.L1Loss()
        else:
            raise Exception(f"Loss function {self.args['loss']} not implemented.")

    @property
    def SPACE(self):
        SPACE = []
        for var in self.args["SPACEs"].keys():
            setting = cp(self.args["SPACEs"][var])
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
        return SPACE

    @property
    def feature_data(self) -> pd.DataFrame:
        """
        Get scaled feature data.

        Returns
        -------
        df
            The scaled feature data.
        """
        return self.datamodule.feature_data if hasattr(self, "datamodule") else None

    @property
    def unscaled_feature_data(self):
        return (
            self.datamodule.unscaled_feature_data
            if hasattr(self, "datamodule")
            else None
        )

    @property
    def unscaled_label_data(self):
        return (
            self.datamodule.unscaled_label_data if hasattr(self, "datamodule") else None
        )

    @property
    def label_data(self) -> pd.DataFrame:
        """
        Get scaled label data.

        Returns
        -------
        df
            The scaled label data.
        """
        return self.datamodule.feature_data if hasattr(self, "datamodule") else None

    @property
    def derived_data(self):
        return self.datamodule.derived_data if hasattr(self, "datamodule") else None

    @property
    def cont_feature_names(self):
        return (
            self.datamodule.cont_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def cat_feature_names(self):
        return (
            self.datamodule.cat_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def all_feature_names(self):
        return (
            self.datamodule.all_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def label_name(self):
        return self.datamodule.label_name if hasattr(self, "datamodule") else None

    @property
    def train_indices(self):
        return self.datamodule.train_indices if hasattr(self, "datamodule") else None

    @property
    def val_indices(self):
        return self.datamodule.val_indices if hasattr(self, "datamodule") else None

    @property
    def test_indices(self):
        return self.datamodule.test_indices if hasattr(self, "datamodule") else None

    @property
    def df(self):
        return self.datamodule.df if hasattr(self, "datamodule") else None

    @property
    def tensors(self):
        return self.datamodule.tensors if hasattr(self, "datamodule") else None

    @property
    def cat_feature_mapping(self):
        return (
            self.datamodule.cat_feature_mapping if hasattr(self, "datamodule") else None
        )

    @property
    def derived_stacked_features(self):
        return (
            self.datamodule.derived_stacked_features
            if hasattr(self, "datamodule")
            else None
        )

    @property
    def training(self):
        return self.datamodule.training if hasattr(self, "datamodule") else None

    def get_material_code(self, *args, **kwargs):
        return self.datamodule.get_material_code(*args, **kwargs)

    def select_by_material_code(self, *args, **kwargs):
        return self.datamodule.select_by_material_code(*args, **kwargs)

    def set_status(self, training: bool):
        self.datamodule.set_status(training)

    def load_data(self, *args, **kwargs):
        if "save_path" in kwargs.keys():
            kwargs.__delitem__("save_path")
        self.datamodule.load_data(save_path=self.project_root, *args, **kwargs)

    def set_path(self, path: Union[os.PathLike, str], verbose=False):
        """
        Set the work directory of the trainer.

        Parameters
        ----------
        path
            The work directory.
        """
        self.project_root = path
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)
        if verbose:
            print(f"Project will be saved to {self.project_root}")

    def _create_dir(self, verbose: bool = True, project_root_subfolder: str = None):
        """
        Create the folder for the project.

        Parameters
        ----------
        verbose
            Whether to print the path to the project.
        project_root_subfolder
            See ``load_config``.
        """
        if not os.path.exists("output"):
            os.mkdir("output")
        if project_root_subfolder is not None:
            if not os.path.exists(os.path.join("output", project_root_subfolder)):
                os.makedirs(os.path.join("output", project_root_subfolder))
        subfolder = (
            self.project
            if project_root_subfolder is None
            else os.path.join(project_root_subfolder, self.project)
        )
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        folder_name = t + "-0" + "_" + os.path.split(self.configfile)[-1]
        if not os.path.exists(os.path.join("output", subfolder)):
            os.mkdir(os.path.join("output", subfolder))
        self.set_path(
            add_postfix(os.path.join("output", subfolder, folder_name)), verbose=verbose
        )

    def summarize_setting(self):
        print("Device:")
        print(pretty(self.summarize_device()))
        print("Configurations:")
        print(pretty(self.args))
        print(f"Global settings:")
        print(pretty(src.setting))

    def summarize_device(self):
        """
        Print a summary of the environment.
        https://www.thepythoncode.com/article/get-hardware-system-information-python
        """

        def get_size(bytes, suffix="B"):
            """
            Scale bytes to its proper format
            e.g:
                1253656 => '1.20MB'
                1253656678 => '1.17GB'
            """
            factor = 1024
            for unit in ["", "K", "M", "G", "T", "P"]:
                if bytes < factor:
                    return f"{bytes:.2f}{unit}{suffix}"
                bytes /= factor

        def get_processor_info():
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":
                return (
                    subprocess.check_output(
                        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
                    )
                    .strip()
                    .decode("utf-8")
                )
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo"
                all_info = (
                    subprocess.check_output(command, shell=True).strip().decode("utf-8")
                )

                for string in all_info.split("\n"):
                    if "model name\t: " in string:
                        return string.split("\t: ")[1]
            return ""

        uname = platform.uname()
        cpufreq = psutil.cpu_freq()
        svmem = psutil.virtual_memory()
        self.sys_summary = {
            "System": uname.system,
            "Node name": uname.node,
            "System release": uname.release,
            "System version": uname.version,
            "Machine architecture": uname.machine,
            "Processor architecture": uname.processor,
            "Processor model": get_processor_info(),
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "Max core frequency": f"{cpufreq.max:.2f}Mhz",
            "Total memory": get_size(svmem.total),
            "Python version": platform.python_version(),
            "Python implementation": platform.python_implementation(),
            "Python compiler": platform.python_compiler(),
            "Cuda availability": torch.cuda.is_available(),
            "GPU devices": [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ],
        }
        return self.sys_summary

    def train(
        self,
        programs: List[str] = None,
        verbose: bool = True,
    ):
        """
        Train all added modelbases.

        Parameters
        ----------
        programs
            A selected subset of modelbases.
        verbose
            Verbosity.
        """
        if programs is None:
            modelbases_to_train = self.modelbases
        else:
            modelbases_to_train = [self.get_modelbase(x) for x in programs]

        if len(modelbases_to_train) == 0:
            warnings.warn(
                f"No modelbase is trained. Please confirm that trainer.add_modelbases is called."
            )

        for modelbase in modelbases_to_train:
            modelbase.train(verbose=verbose)

    def cross_validation(
        self,
        programs: List[str],
        n_random: int,
        verbose: bool,
        test_data_only: bool,
        type: str = "random",
        load_from_previous: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Repeat loading data, training modelbases, and evaluating all models for multiple times.

        Parameters
        ----------
        programs
            A selected subset of modelbases.
        n_random
            The number of repeats.
        verbose
            Verbosity.
        test_data_only
            Whether to evaluate models only on testing datasets.
        type
            The type of data splitting. "random" is currently supported.
        load_from_previous
            Load the state of a previous run (mostly because of an unexpected interruption).

        Notes
        -------
        The results of a continuously run and a continued run (``load_from_previous=True``) are consistent.

        Returns
        -------
        res
            A dict in the following format:
            keys: programs
            values:
                keys: model names
                values:
                    keys: ["Training", "Testing", "Validation"]
                    values: (Predicted values, true values)
        """
        if not os.path.exists(os.path.join(self.project_root, "cv")):
            os.mkdir(os.path.join(self.project_root, "cv"))
        programs_predictions = {}
        for program in programs:
            programs_predictions[program] = {}
        if type == "random":
            set_data_handler = self.load_data
        else:
            raise Exception(f"{type} cross validation not implemented.")

        if load_from_previous:
            if not os.path.isfile(
                os.path.join(self.project_root, "cv", "cv_state.pkl")
            ):
                raise Exception(f"No previous state to load from.")
            with open(
                os.path.join(self.project_root, "cv", "cv_state.pkl"), "rb"
            ) as file:
                current_state = pickle.load(file)
            start_i = current_state["i_random"]
            self.load_state(current_state["trainer"])
            programs_predictions = current_state["programs_predictions"]
            if "once_predictions" in current_state.keys():
                reloaded_once_predictions = current_state["once_predictions"]
            else:
                # For compatibility
                reloaded_once_predictions = None
            skip_program = reloaded_once_predictions is not None
            if start_i >= n_random:
                raise Exception(
                    f"The loaded state is incompatible with the current setting."
                )
            print(f"Previous cross validation state is loaded.")
        else:
            start_i = 0
            skip_program = False
            reloaded_once_predictions = None

        def func_save_state(state):
            with open(
                os.path.join(self.project_root, "cv", "cv_state.pkl"), "wb"
            ) as file:
                pickle.dump(state, file)

        for i in range(start_i, n_random):
            if verbose:
                print(
                    f"----------------------------{i + 1}/{n_random} {type} cross validation----------------------------"
                )
            if not skip_program:
                current_state = {
                    "trainer": cp(self),
                    "i_random": i,
                    "programs_predictions": programs_predictions,
                    "once_predictions": None,
                }
                func_save_state(current_state)
            with HiddenPrints(disable_std=not verbose):
                set_random_seed(src.setting["random_seed"] + i)
                set_data_handler()
            once_predictions = {} if not skip_program else reloaded_once_predictions
            for program in programs:
                if skip_program:
                    if program in once_predictions.keys():
                        print(f"Skipping finished model base {program}")
                        continue
                    else:
                        skip_program = False
                modelbase = self.get_modelbase(program)
                modelbase.train(dump_trainer=True, verbose=verbose)
                predictions = modelbase._predict_all(
                    verbose=verbose, test_data_only=test_data_only
                )
                once_predictions[program] = predictions
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
                current_state = {
                    "trainer": cp(self),
                    "i_random": i,
                    "programs_predictions": programs_predictions,
                    "once_predictions": once_predictions,
                }
                func_save_state(current_state)
            df_once = self._cal_leaderboard(
                once_predictions, test_data_only=test_data_only, save=False
            )
            df_once.to_csv(
                os.path.join(self.project_root, "cv", f"leaderboard_cv_{i}.csv")
            )
            current_state = {
                "trainer": cp(self),
                "i_random": i + 1,
                "programs_predictions": programs_predictions,
                "once_predictions": None,
            }
            func_save_state(current_state)
            if verbose:
                print(
                    f"--------------------------End {i + 1}/{n_random} {type} cross validation--------------------------"
                )
        return programs_predictions

    def get_leaderboard(
        self,
        test_data_only: bool = False,
        dump_trainer: bool = True,
        cross_validation: int = 0,
        verbose: bool = True,
        load_from_previous: bool = False,
    ) -> pd.DataFrame:
        """
        Run all modelbases with/without cross validation for a leaderboard.

        Parameters
        ----------
        test_data_only
            Whether to evaluate models only on testing datasets.
        dump_trainer
            Whether to save trainer.
        cross_validation
            The number of cross validation. See Trainer.cross_validation. 0 to evaluate directly on current datasets.
        verbose
            Verbosity.
        load_from_previous
            Load the state of a previous run (mostly because of an unexpected interruption).

        Returns
        -------
        leaderboard
            The leaderboard dataframe.
        """
        if len(self.modelbases) == 0:
            raise Exception(
                f"No modelbase available. Run trainer.add_modelbases() first."
            )
        if cross_validation != 0:
            programs_predictions = self.cross_validation(
                programs=self.modelbases_names,
                n_random=cross_validation,
                verbose=verbose,
                test_data_only=test_data_only,
                load_from_previous=load_from_previous,
            )
        else:
            programs_predictions = {}
            for modelbase in self.modelbases:
                print(f"{modelbase.program} metrics")
                programs_predictions[modelbase.program] = modelbase._predict_all(
                    verbose=verbose, test_data_only=test_data_only
                )

        df_leaderboard = self._cal_leaderboard(
            programs_predictions, test_data_only=test_data_only
        )
        if dump_trainer:
            save_trainer(self)
        return df_leaderboard

    def _cal_leaderboard(
        self,
        programs_predictions: Dict[
            str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        ],
        metrics: List[str] = None,
        test_data_only: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the leaderboard based on results from cross_validation or AbstractModel._predict_all.

        Parameters
        ----------
        programs_predictions
            Results from Trainer.cross_validation, or assembled results from AbstractModel._predict_all. See
            Trainer.get_leaderboard for details.
        metrics
            The metrics that have been implemented in src.utils.metric_sklearn.
        test_data_only
            Whether to evaluate models only on testing datasets.
        save
            Whether to save the leaderboard locally and as an attribute in the trainer.

        Returns
        -------
        leaderboard
            The leaderboard dataframe.
        """
        if metrics is None:
            metrics = ["rmse", "mse", "mae", "mape", "r2", "rmse_conserv"]
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
        if save:
            df_leaderboard.to_csv(os.path.join(self.project_root, "leaderboard.csv"))
            self.leaderboard = df_leaderboard
        return df_leaderboard

    def plot_loss(self, train_ls: Any, val_ls: Any):
        """
        A utility function to plot loss value during training.

        Parameters
        ----------
        train_ls:
            An array of training loss.
        val_ls
            An array of validation loss.
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
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{self.args['loss'].upper()} Loss")
        plt.savefig(os.path.join(self.project_root, "loss_epoch.pdf"))
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred(self, program: str, log_trans: bool = True, upper_lim=9):
        """
        Comparing ground truth and prediction for all models in a modelbase.

        Parameters
        ----------
        program
            The selected modelbase.
        log_trans
            Whether the label data is in log scale.
        upper_lim
            The upper limit of x/y-axis.
        """
        modelbase = self.get_modelbase(program)
        model_names = modelbase.get_model_names()
        predictions = modelbase._predict_all()

        for idx, model_name in enumerate(model_names):
            print(model_name, f"{idx + 1}/{len(model_names)}")
            plt.figure()
            plt.rcParams["font.size"] = 14
            ax = plt.subplot(111)

            plot_truth_pred(
                predictions, ax, model_name, log_trans=log_trans, verbose=True
            )

            set_truth_pred(ax, log_trans, upper_lim=upper_lim)

            plt.legend(
                loc="upper left", markerscale=1.5, handlelength=0.2, handleheight=0.9
            )

            s = model_name.replace("/", "_")

            plt.savefig(os.path.join(self.project_root, program, f"{s}_truth_pred.pdf"))
            if is_notebook():
                plt.show()

            plt.close()

    def cal_feature_importance(
        self, program: str, model_name: str, method: str = "permutation"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate feature importance with a specified model. If the modelbase is ``TorchModel``, ``captum`` and ``shap``
        is called to make permutations. If the modelbase is only an ``AbstractModel``, calculation will be much slower.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model in the modelbase.
        method
            The method to calculate importance. "permutation" or "shap".

        Returns
        ----------
        attr
            Values of feature importance.
        importance_names
            Corresponding feature names. If the modelbase is a ``TorchModel``, all features including derived unstacked
            features will be included. Otherwise, only ``Trainer.all_feature_names`` will be considered.
        """
        from src.model.base import TorchModel

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
                        pin_memory=True,
                    )
                    prediction, _, _ = modelbase._test_step(
                        modelbase.model[model_name], loader
                    )
                    loss = float(
                        self._metric_sklearn(
                            ground_truth, prediction, self.args["loss"]
                        )
                    )
                    return loss

                feature_perm = FeaturePermutation(forward_func)
                attr = [
                    x.cpu().numpy().flatten()
                    for x in feature_perm.attribute(
                        tuple(
                            [
                                self.datamodule.get_first_tensor_slice(
                                    self.test_indices
                                ),
                                *self.datamodule.get_additional_tensors_slice(
                                    self.test_indices
                                ),
                            ]
                        )
                    )
                ]
                attr = np.abs(np.concatenate(attr))
            elif method == "shap":
                attr = self.cal_shap(program=program, model_name=model_name)
            else:
                raise NotImplementedError
            dims = self.datamodule.get_derived_data_sizes()
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
                test_data = self.datamodule.X_test
                base_pred = modelbase.predict(
                    test_data,
                    derived_data=self.datamodule.D_test,
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
                        derived_data=self.datamodule.derive_unstacked(df),
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
                attr = self.cal_shap(program=program, model_name=model_name)
            else:
                raise NotImplementedError
            importance_names = cp(self.all_feature_names)

        return attr, importance_names

    def cal_shap(self, program: str, model_name: str) -> np.ndarray:
        """
        Calculate SHAP values with a specified model. If the modelbase is a ``TorchModel``, the ``shap.DeepExplainer``
        is used. Otherwise, ``shap.KernelExplainer`` is called, which is much slower, and shap.kmeans is called to
        summarize training data to 10 samples as the background data and 10 random samples in the testing set is
        explained, which will bias the results.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model in the modelbase.

        Returns
        -------
        attr
            The SHAP values. If the modelbase is a `TorchModel`, all features including derived unstacked features will
            be included. Otherwise, only `Trainer.all_feature_names` will be considered.
        """
        from src.model.base import TorchModel
        import shap

        modelbase = self.get_modelbase(program)
        is_torch = issubclass(type(modelbase), TorchModel)
        if is_torch:
            bk_indices = np.random.choice(self.train_indices, size=100, replace=False)
            X_train_bk = self.datamodule.get_first_tensor_slice(bk_indices)
            D_train_bk = self.datamodule.get_additional_tensors_slice(bk_indices)
            background_data = [X_train_bk, *D_train_bk]

            X_test = self.datamodule.get_first_tensor_slice(self.test_indices)
            D_test = self.datamodule.get_additional_tensors_slice(self.test_indices)
            test_data = [X_test, *D_test]
            explainer = shap.DeepExplainer(modelbase.model[model_name], background_data)

            with HiddenPrints():
                shap_values = explainer.shap_values(test_data)
        else:
            background_data = shap.kmeans(
                self.df.loc[self.train_indices, self.all_feature_names], 10
            )
            warnings.filterwarnings(
                "ignore",
                message="The default of 'normalize' will be set to False in version 1.2 and deprecated in version 1.4.",
            )

            def func(data):
                df = pd.DataFrame(columns=self.all_feature_names, data=data)
                return modelbase.predict(
                    df,
                    model_name=model_name,
                    derived_data=self.datamodule.derive_unstacked(
                        df, categorical_only=True
                    ),
                    ignore_absence=True,
                ).flatten()

            test_indices = np.random.choice(self.test_indices, size=10, replace=False)
            test_data = self.df.loc[test_indices, self.all_feature_names].copy()
            shap_values = shap.KernelExplainer(func, background_data).shap_values(
                test_data
            )
        attr = (
            np.concatenate(
                [np.mean(np.abs(shap_values[0]), axis=0)]
                + [np.mean(np.abs(x), axis=0) for x in shap_values[1:]],
            )
            if type(shap_values) == list and len(shap_values) > 1
            else np.mean(np.abs(shap_values[0] if is_torch else shap_values), axis=0)
        )
        return attr

    def plot_feature_importance(
        self,
        program: str,
        model_name: str,
        fig_size: Tuple = (7, 4),
        method: str = "permutation",
    ):
        """
        Plot feature importance of a model using ``Trainer.cal_feature_importance``.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model in the modelbase.
        fig_size
            The figure size.
        method
            The method to calculate importance. "permutation" or "shap".
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

        dims = self.datamodule.get_derived_data_sizes()
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

        where_effective = np.abs(attr) > 1e-5
        effective_names = np.array(names)[where_effective]
        print(
            f"Feature importance less than 1e-5: {list(np.setdiff1d(names, effective_names))}"
        )
        attr = attr[where_effective]
        pal = [x for idx, x in enumerate(pal) if where_effective[idx]]

        plt.figure(figsize=fig_size)
        ax = plt.subplot(111)
        plot_importance(
            ax,
            effective_names,
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
            os.path.join(
                self.project_root,
                f"feature_importance_{program}_{model_name}_{method}.png",
            ),
            dpi=600,
        )
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_dependence(
        self,
        program: str,
        model_name: str,
        refit: bool = True,
        log_trans: bool = True,
        lower_lim: float = 2,
        upper_lim: float = 7,
        n_bootstrap: int = 1,
        grid_size: int = 30,
        CI: float = 0.95,
        verbose: bool = True,
    ):
        """
        Calculate and plot partial dependence plots with bootstrapping.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model in the modelbase.
        refit
            Whether to refit models on bootstrapped datasets. See Trainer._bootstrap.
        log_trans
            Whether the label data is in log scale.
        lower_lim
            Lower limit of each pdp.
        upper_lim
            Upper limit of each pdp.
        n_bootstrap
            The number of bootstrap evaluations. It should be greater than 0.
        grid_size
            The grid of pdp.
        CI
            Confidence interval of pdp results across multiple bootstrap runs.
        verbose
            Verbosity
        """
        (
            x_values_list,
            mean_pdp_list,
            ci_left_list,
            ci_right_list,
        ) = self.cal_partial_dependence(
            feature_subset=self.all_feature_names,
            program=program,
            model_name=model_name,
            df=self.datamodule.X_train,
            derived_data=self.datamodule.D_train,
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
            os.path.join(
                self.project_root, f"partial_dependence_{program}_{model_name}.pdf"
            )
        )
        if is_notebook():
            plt.show()
        plt.close()

    def cal_partial_dependence(
        self, feature_subset: List[str] = None, **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Calculate partial dependency. See ``Trainer.plot_partial_dependence`` for recommended usage.

        Parameters
        ----------
        feature_subset
            A subset of ``Trainer.all_feature_names``.
        kwargs
            Arguments for ``Trainer._bootstrap``.

        Returns
        -------
        res
            Lists of x values, pdp values, lower confidence limit, and upper confidence limit for each feature.
        """
        x_values_list = []
        mean_pdp_list = []
        ci_left_list = []
        ci_right_list = []

        for feature_idx, feature_name in enumerate(
            self.all_feature_names if feature_subset is None else feature_subset
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

    def plot_partial_err(self, program: str, model_name: str, thres: Any = 0.8):
        """
        Calculate prediction error on the testing dataset, and plot parallel histograms of high error samples and low
        error samples (considering absolute error) respectively.

        Parameters
        ----------
        program
            The selected modelbase.
        model_name
            The selected model in the modelbase.
        thres
            The absolute error threshold to identify high error samples and low error samples.

        """
        modelbase = self.get_modelbase(program)

        ground_truth = self.label_data.loc[self.test_indices, :].values.flatten()
        prediction = modelbase.predict(
            df=self.datamodule.X_test,
            derived_data=self.datamodule.D_test,
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

        plt.savefig(
            os.path.join(self.project_root, f"partial_err_{program}_{model_name}.pdf")
        )
        if is_notebook():
            plt.show()
        plt.close()

    def plot_corr(self, fontsize: Any = 10, cmap="bwr", imputed=False):
        """
        Plot Pearson correlation among features and the target.

        Parameters
        ----------
        fontsize
            The fontsize for matplotlib.
        cmap
            The colormap for matplotlib.
        imputed
            Whether the imputed dataset should be considered. If False, some NaN values may exit for features with
            missing value.
        """
        cont_feature_names = self.cont_feature_names + self.label_name
        # sns.reset_defaults()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        plt.box(on=True)
        corr = self.datamodule.cal_corr(imputed=imputed).values
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
        plt.savefig(os.path.join(self.project_root, "corr.pdf"))
        if is_notebook():
            plt.show()
        plt.close()

    def plot_pairplot(self, **kwargs):
        """
        Plot ``seaborn.pairplot`` among features and label. Kernel Density Estimation plots are on the diagonal.

        Parameters
        ----------
        kwargs
            Arguments for ``seaborn.pairplot``.
        """
        df_all = pd.concat(
            [self.unscaled_feature_data, self.unscaled_label_data], axis=1
        )
        sns.pairplot(df_all, corner=True, diag_kind="kde", **kwargs)
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_root, "pair.jpg"))
        if is_notebook():
            plt.show()
        plt.close()

    def plot_feature_box(self, imputed: bool = False):
        """
        Plot boxplot of the tabular data.

        Parameters
        ----------
        imputed
            Whether the imputed dataset should be considered.
        """
        # sns.reset_defaults()
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        bp = sns.boxplot(
            data=self.feature_data
            if imputed
            else self.datamodule.get_not_imputed_df()[self.cont_feature_names],
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
        plt.savefig(os.path.join(self.project_root, "feature_box.pdf"))
        plt.show()
        plt.close()

    def plot_data_split(self, bins: int = 30, percentile: str = "all"):
        """
        Visualize how the dataset is split into training/validation/testing datasets.

        Parameters
        ----------
        bins
            Number of bins to divide the target value of each material.
        percentile
            If "all", the limits of x-axis will be the overall range of the target. Otherwise, the limits will be [0,1]
            representing the individual percentile of target for each material.
        """
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
        partition_cycles = {"train": [], "val": [], "test": []}
        np.seterr(invalid="ignore")
        for idx, material in enumerate(all_m_code):
            cycle = all_cycle[
                self.select_by_material_code(m_code=material, partition="all")
            ]
            if percentile == "all":
                hist_range = (np.min(all_cycle), np.max(all_cycle))
            else:
                hist_range = (np.min(cycle), np.max(cycle))
            all_hist = np.histogram(cycle, bins=bins, range=hist_range)[0]

            def get_heat(partition):
                cycles = all_cycle[
                    self.select_by_material_code(m_code=material, partition=partition)
                ]
                partition_cycles[partition].append(cycles)
                return np.histogram(cycles, range=hist_range, bins=bins)[0] / all_hist

            train_heat[idx, :] = get_heat(partition="train")
            val_heat[idx, :] = get_heat(partition="val")
            test_heat[idx, :] = get_heat(partition="test")

        train_heat[np.isnan(train_heat)] = 0
        val_heat[np.isnan(val_heat)] = 0
        test_heat[np.isnan(test_heat)] = 0
        fig = plt.figure(figsize=(8, 2.5))
        gs = GridSpec(100, 100, figure=fig)

        def plot_im(heat, pos, hide_y_ticks=False):
            ax = fig.add_subplot(pos)
            im = ax.imshow(heat, aspect="auto", cmap="Oranges")
            if hide_y_ticks:
                ax.set_yticks([])
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

        def plot_kde(heat, pos, name, hide_y_ticks=False):
            ax = fig.add_subplot(pos)
            x = np.linspace(np.min(all_cycle), np.max(all_cycle), 100)
            kde = st.gaussian_kde(np.hstack(heat))(x)
            ax.plot(x, kde, c=plt.get_cmap("Oranges")(200), lw=1)
            where_max_kde = np.argmax(kde)
            ax.plot(
                np.array([x[where_max_kde]] * 10),
                np.linspace(0, 0.5, 10),
                "--",
                color="grey",
                alpha=0.5,
                lw=1,
            )
            ax.text(
                x[where_max_kde] + 0.1,
                0.37,
                f"{x[where_max_kde]:.2f}",
                fontsize=plt.rcParams["font.size"] * 0.8,
            )
            if hide_y_ticks:
                ax.set_yticks([])
            else:
                ax.set_yticks([0, 0.5])
            ax.set_xticks([])
            ax.set_xlim([np.min(all_cycle), np.max(all_cycle)])
            ax.set_ylim([0, 0.5])
            ax.set_title(name)

        plot_kde(
            partition_cycles["train"],
            gs[:18, 0:30],
            name="Training set",
            hide_y_ticks=False,
        )
        plot_kde(
            partition_cycles["val"],
            gs[:18, 33:63],
            name="Validation set",
            hide_y_ticks=True,
        )
        plot_kde(
            partition_cycles["test"],
            gs[:18, 66:96],
            name="Testing set",
            hide_y_ticks=True,
        )
        plot_im(train_heat, gs[25:, 0:30], hide_y_ticks=False)
        plot_im(val_heat, gs[25:, 33:63], hide_y_ticks=True)
        im = plot_im(test_heat, gs[25:, 66:96], hide_y_ticks=True)
        # plt.colorbar(mappable=im)
        cax = fig.add_subplot(gs[50:98, 98:])
        cbar = plt.colorbar(cax=cax, mappable=im)
        cax.set_ylabel("Density")
        ax = fig.add_subplot(gs[:20, :], frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_ylabel(f"KDE\n")
        ax = fig.add_subplot(gs[25:, :], frameon=False)
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
        warnings.filterwarnings("ignore", message="Tight layout not applied.")
        plt.savefig(
            os.path.join(
                self.project_root,
                f"{self.datamodule.datasplitter.__class__.__name__}_{percentile}.pdf",
            ),
            bbox_inches="tight",
            dpi=1000,
        )
        plt.close()

    def plot_multiple_S_N(
        self, m_codes: List[str], hide_plt_show: bool = True, **kwargs
    ):
        """
        A utility function to call ``Trainer.plot_S_N`` for multiple times with the same settings.

        Parameters
        ----------
        m_codes
            A list of material codes.
        hide_plt_show
            Whether to prevent the matplotlib figure showing in canvas.
        kwargs
            Arguments for ``Trainer.plot_S_N``.
        """
        for m_code in m_codes:
            print(m_code)
            if hide_plt_show:
                with HiddenPltShow():
                    self.plot_S_N(m_code=m_code, **kwargs)
            else:
                self.plot_S_N(m_code=m_code, **kwargs)

    def plot_S_N(
        self,
        s_col: str,
        n_col: str,
        r_col: str,
        m_code: str,
        r_value: float,
        f_col: str = None,
        freq: float = None,
        load_dir: str = "tension",
        avg_feature: List[str] = None,
        ax=None,
        CI: float = 0.95,
        method: str = "statistical",
        verbose: bool = True,
        program: str = "ThisWork",
        model_name: str = "ThisWork",
        refit: bool = True,
        **kwargs,
    ):
        """
        Calculate and plot the SN curve for the selected material using a selected model with bootstrap resampling.

        Parameters
        ----------
        s_col
            The name of the stress column.
        n_col
            The name of the log(fatigue life) column.
        r_col
            The name of the R-value column.
        m_code
            The selected material code.
        r_value
            The selected R-value to plot.
        f_col
            The name of the frequency column.
        freq
            The selected frequency to plot.
        load_dir
            "tension" or else. If "tension" is selected, samples with s_col>0 will be selected for evaluation.
            Otherwise, those with s_col<0 will be selected.
        avg_feature
            A list of features to ignore when predicting SN curves by setting their values to their average.
        ax
            A matplotlib Axis instance. If not None, a new figure will be initialized.
        CI
            The confidence interval for PSN curves and for predicted SN curves across multiple bootstrap runs and
            multiple samples. The latter usage is different from the CI argument in``Trainer.cal_partial_dependence``.
        method
            The method to calculate the confidence interval. See ``Trainer._sn_interval`` for details.
        verbose
            Verbosity.
        program
            The selected database.
        model_name
            The selected model in the database.
        refit
            Whether to refit models on bootstrapped datasets. See Trainer._bootstrap.
        **kwargs
            Other arguments for ``Trainer._bootstrap``
        """
        # Check whether columns exist.
        if s_col not in self.df.columns:
            raise Exception(f"{s_col} not in features.")
        if n_col not in self.label_name:
            raise Exception(f"{n_col} is not the target.")

        # Find the selected material.
        m_train_indices = self.select_by_material_code(m_code, partition="train")
        m_test_indices = self.select_by_material_code(m_code, partition="test")
        m_val_indices = self.select_by_material_code(m_code, partition="val")
        original_indices = self.select_by_material_code(m_code, partition="all")

        if len(original_indices) == 0:
            raise Exception(f"Material_Code {m_code} not available.")

        sgn = 1 if load_dir == "tension" else -1

        include_f = (
            f_col is not None and freq is not None and f_col in self.cont_feature_names
        )
        get_indices_wo_f = lambda indices, r, f: indices[
            (self.df.loc[indices, s_col] * sgn > 0)
            & ((self.df.loc[indices, r_col] - r).__abs__() < 1e-3)
        ]
        get_indices_w_f = lambda indices, r, f: indices[
            (self.df.loc[indices, s_col] * sgn > 0)
            & ((self.df.loc[indices, r_col] - r).__abs__() < 1e-3)
            & ((self.df.loc[indices, f_col] - f).__abs__() < 1e-3)
        ]
        get_indices_handle = get_indices_w_f if include_f else get_indices_wo_f

        m_train_indices = get_indices_handle(m_train_indices, r_value, freq)
        m_test_indices = get_indices_handle(m_test_indices, r_value, freq)
        m_val_indices = get_indices_handle(m_val_indices, r_value, freq)

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
        if avg_feature is not None:
            warnings.warn(
                f"Some features are set to their average value: {avg_feature}"
            )
            processed_df = self.df.copy()
            for feature in avg_feature:
                processed_df[feature] = np.mean(processed_df[feature])
        else:
            processed_df = self.df.copy()

        # If no training or validation points available, raise an exception.
        if (
            len(m_train_indices) == 0
            and len(m_val_indices) == 0
            and len(m_test_indices) == 0
        ):
            if include_f:
                r = self.df.loc[original_indices, r_col].values.flatten()
                f = self.df.loc[original_indices, f_col].values.flatten()
                unique_fr = set(list(zip(r, f)))
                available_fr = []
                for r, f in unique_fr:
                    if len(get_indices_handle(original_indices, r, f)) > 0:
                        available_fr.append((r, f))
                raise Exception(
                    f"The combination of R-value {r_value} and frequency {freq} is not available. Choose among "
                    f"{available_fr}."
                )
            else:
                unique_r = np.unique(self.df.loc[original_indices, r_col])
                available_r = []
                for r in unique_r:
                    if len(get_indices_handle(original_indices, r, freq)) > 0:
                        available_r.append(r)
                raise Exception(
                    f"R-value {r_value} is not available. Choose among {available_r}."
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
            df=processed_df.loc[chosen_indices, :],
            derived_data=self.datamodule.get_derived_data_slice(
                self.derived_data, chosen_indices
            ),
            focus_feature=s_col,
            x_min=s_min,
            x_max=s_max,
            CI=CI,
            average=False,
            verbose=verbose,
            model_name=model_name,
            refit=refit if len(m_train_indices) != 0 else False,
            **kwargs,
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
            if np.isfinite(interv_left).all() and np.isfinite(interv_right).all():
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
            else:
                warnings.warn(
                    f"Invalid value encountered when calculating intervals. It is probably because only one"
                    f"unique stress value exists in the training set."
                )

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
        if len(m_train_indices) > 3:
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

        ax.legend(
            loc="upper right",
            markerscale=1.5,
            handlelength=1,
            handleheight=0.9,
            fontsize=plt.rcParams["font.size"] * 0.8,
        )
        ax.set_xlabel(n_col)
        ax.set_ylabel(s_col)
        ax.set_xlim([0, 10])
        ax.set_title(f"{m_code} R={r_value} CI={CI * 100:.1f}\%")

        path = os.path.join(self.project_root, f"SN_curves_{program}_{model_name}")
        if not os.path.exists(path):
            os.mkdir(path=path)
        fig_name = (
            m_code.replace("/", "_")
            + f"_r_{r_value}{f'_f_{freq}' if include_f else ''}{f'_refit' if refit else ''}.pdf"
        )
        plt.savefig(path + "/" + fig_name)

        if is_notebook() and new_ax:
            plt.show()
        if new_ax:
            plt.close()

    @staticmethod
    def _sn_interval(
        method: str,
        y: np.ndarray,
        y_pred: np.ndarray,
        x: np.ndarray,
        xvals: np.ndarray,
        CI: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SN confidence intervals based on statistical or ASTM method.

        Parameters
        ----------
        method
            "statistical" or "astm".

            "statistical": Schneider, C. R. A., and S. J. Maddox. "Best practice guide on statistical analysis of
            fatigue data." Weld Inst Stat Rep (2003).

            "astm": According to ASTM E739-10(2015). x can be stress, log(stress), strain, log(strain), etc. It is
            valid when y and x follow the linear assumption. Barbosa, Joelton Fonseca, et al. "Probabilistic SN fields
            based on statistical distributions applied to metallic and composite materials: State of the art." Advances
            in Mechanical Engineering 11.8 (2019): 1687814019870395.
        y
            The true value of fatigue life (in log scale).
        y_pred
            The predicted value of fatigue life (in log scale).
        x
            The true value of stress.
        xvals
            The value of stress to be evaluated on.
        CI
            The confidence interval.

        Returns
        -------
        ci
            The widths of left and right confidence bounds.
        """
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "statistical":
            # The two-sided prediction limits are symmetrical, so we calculate one-sided limit instead; therefore, in
            # st.t.ppf or st.f.ppf, the first probability argument is (CI+1)/2 instead of CI for one-sided prediction limit.
            # Because, for example for two-sided CI=95%, the lower limit is equivalent to one-sided 97.5% limit.
            tinv = st.t.ppf((CI + 1) / 2, n - 2)
            CL = tinv * STEYX * (1 + 1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            return CL.flatten(), CL.flatten()
        elif method == "astm":
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
    def _psn(
        method: str,
        y: np.ndarray,
        y_pred: np.ndarray,
        x: np.ndarray,
        xvals: np.ndarray,
        CI: float,
        p: float,
    ) -> np.ndarray:
        """
        Calculate probabilistic SN curves.

        Parameters
        ----------
        method
            "iso" is currently supported. See ISO 12107.
        y
            The true value of fatigue life (in log scale).
        y_pred
            The predicted value of fatigue life (in log scale).
        x
            The true value of stress.
        xvals
            The value of stress to be evaluated on.
        CI
            The confidence interval.
        p
            The probability to failure.

        Returns
        -------
        cl
            The probabilistic SN curve.
        """
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "iso":

            def oneside_normal(p, CI, sample_size, n_random=100000, ddof=1):
                # The one-sided tolerance limits of normal distribution in ISO are given in a table. We find that the
                # analytical calculation is difficult to implement (https://statpages.info/tolintvl.html gives an
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
        program: str,
        df: pd.DataFrame,
        derived_data: Dict[str, np.ndarray],
        focus_feature: str,
        n_bootstrap: int = 1,
        grid_size: int = 30,
        verbose: bool = True,
        rederive: bool = True,
        refit: bool = True,
        resample: bool = True,
        percentile: float = 100,
        x_min: float = None,
        x_max: float = None,
        CI: float = 0.95,
        average: bool = True,
        model_name: str = "ThisWork",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make bootstrap resampling, fit the selected model on the resampled data, and assign sequential values to the
        selected feature to see how the prediction changes with respect to the feature.

        Cook, Thomas R., et al. Explaining Machine Learning by Bootstrapping Partial Dependence Functions and Shapley
        Values. No. RWP 21-12. 2021.

        Parameters
        ----------
        program
            The selected modelbase.
        df
            The tabular dataset.
        derived_data
            The derived data calculated using ``Trainer.derive_unstacked``.
        focus_feature
            The feature to assign sequential values.
        n_bootstrap
            The number of bootstrapping, fitting, and assigning runs.
        grid_size
            The length of sequential values.
        verbose
            Ignored.
        rederive
            Ignored. If the focus_feature is a derived stacked feature, derivation will not perform on the bootstrap
            dataset. Otherwise, stacked/unstacked features will be rederived.
        refit
            Whether to fit the model on the bootstrap dataset with warm_start=True.
        resample
            Whether to make bootstrap resample. Only recommended to False when n_bootstrap=1.
        percentile
            The percentile of the feature to generate sequential values for the selected feature.
        x_min
            The lower limit of the generated sequential values. It will override the left percentile.
        x_max
            The upper limit of the generated sequential values. It will override the right percentile.
        CI
            The confidence interval to evaluate bootstrapped predictions.
        average
            If True, CI will be calculated on results ``(grid_size, n_bootstrap)`` across multiple bootstrap runs.
            Predictions for all samples are averaged for each bootstrap run. This case is used in `
            `Trainer.cal_partial_dependence``

            If False, CI will be calculated on results ``(grid_size, n_bootstrap*len(df))`` across multiple bootstrap
            runs and all samples. This case is used in ``Trainer.plot_S_N``.
        model_name
            The selected model in the modelbase.

        Returns
        -------
        res
            The generated sequential values for the feature, averaged predictions on the sequential values across
            multiple bootstrap runs and all samples, left confidence interval, and right confidence interval.
        """
        from .utils import NoBayesOpt

        modelbase = self.get_modelbase(program)
        derived_data = self.datamodule.sort_derived_data(derived_data)
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
            tmp_derived_data = self.datamodule.get_derived_data_slice(
                derived_data, list(df_bootstrap.index)
            )
            df_bootstrap = df_bootstrap.reset_index(drop=True)
            bootstrap_model = modelbase.detach_model(model_name=model_name)
            if refit:
                with NoBayesOpt(self):
                    bootstrap_model.fit(
                        df_bootstrap,
                        model_subset=[model_name],
                        cont_feature_names=self.datamodule.dataprocessors[0][
                            0
                        ].record_cont_features,
                        cat_feature_names=self.datamodule.dataprocessors[0][
                            0
                        ].record_cat_features,
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

    def load_state(self, trainer: "Trainer"):
        """
        Restore the trainer from a deepcopied state.

        Parameters
        ----------
        trainer
            A deepcopied previous status of the trainer.
        """
        # https://stackoverflow.com/questions/1216356/is-it-safe-to-replace-a-self-object-by-another-object-of-the-same-type-in-a-meth
        current_root = cp(self.project_root)
        self.__dict__.update(trainer.__dict__)
        # The update operation does not change the location of self. However, model bases contains another trainer
        # that points to another location if the state is loaded from disk.
        for model in self.modelbases:
            model.trainer = self
        self.set_path(current_root, verbose=False)
        for modelbase in self.modelbases:
            modelbase.set_path(os.path.join(current_root, modelbase.program))

    def get_best_model(self) -> Tuple[str, str]:
        """
        Get the best model in the leaderboard.

        Returns
        -------
        program and model_name
            The name of a modelbase which the best model is located and the name of the best model.
        """
        if not hasattr(self, "leaderboard"):
            self.get_leaderboard(test_data_only=True, dump_trainer=False)
        return (
            self.leaderboard["Program"].values[0],
            self.leaderboard["Model"].values[0],
        )

    @staticmethod
    def _metrics(
        predictions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        metrics: List[str],
        test_data_only: bool,
    ) -> pd.DataFrame:
        """
        Calculate metrics for predictions.

        Parameters
        ----------
        predictions
            Results from AbstractModel._predict_all.
        metrics
            The metrics that have been implemented in src.utils.metric_sklearn.
        test_data_only
            Whether to evaluate models only on testing datasets.

        Returns
        -------
        df_metrics
            A dataframe of metrics.
        """
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
    def _metric_sklearn(y_true: np.ndarray, y_pred: np.ndarray, metric: str):
        """
        Evaluate a prediction using a certain metric. It is a wrapper method to call ``src.utils.metric_sklearn``.

        Parameters
        ----------
        y_true
            The true value of the target.
        y_pred
            The predicted value of the target.
        metric
            A metric that has been implemented in src.utils.metric_sklearn.

        Returns
        -------
        metric_value
            The metric of prediction.
        """
        return metric_sklearn(y_true, y_pred, metric)


def save_trainer(
    trainer: Trainer, path: Union[os.PathLike, str] = None, verbose: bool = True
):
    """
    Pickling the trainer instance.

    Parameters
    ----------
    trainer
        The Trainer to be saved.
    path
        The folder path to save the trainer.
    verbose
        Verbosity.
    """
    import pickle

    path = os.path.join(trainer.project_root, "trainer.pkl") if path is None else path
    with open(path, "wb") as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(
            f"Trainer saved. To load the trainer, run trainer = load_trainer(path='{path}')"
        )


def load_trainer(path: Union[os.PathLike, str]) -> Trainer:
    """
    Loading a pickled Trainer. Paths of the trainer and its model bases will be changed (i.e. ``Trainer.project_root``,
    ``AbstractModel.root``, ``AbstractModel.model.root``, and ``AbstractModel.model.model_path.keys()``)

    Parameters
    ----------
    path
        Path of the Trainer.

    Returns
    -------
    trainer
        The loaded Trainer.
    """
    import pickle

    with open(path, "rb") as inp:
        trainer = pickle.load(inp)
    root = os.path.join(*os.path.split(path)[:-1])
    trainer.set_path(root, verbose=False)
    for modelbase in trainer.modelbases:
        modelbase.set_path(os.path.join(root, modelbase.program))
    return trainer
