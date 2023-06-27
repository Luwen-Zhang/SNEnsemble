import os
import pickle
import numpy as np
import pandas as pd
import torch.optim.optimizer
import src
from src.utils import *
from src.trainer import Trainer, save_trainer
from src.data import DataModule
import skopt
from skopt import gp_minimize
import torch.utils.data as Data
import torch.nn as nn
from copy import deepcopy as cp
from typing import *
from skopt.space import Real, Integer, Categorical
import time
import pytorch_lightning as pl
from functools import partial
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections.abc import Iterable


class AbstractModel:
    def __init__(
        self,
        trainer: Trainer,
        program: str = None,
        model_subset: List[str] = None,
        exclude_models: List[str] = None,
        store_in_harddisk: bool = True,
        **kwargs,
    ):
        """
        The base class for all model-bases.

        Parameters
        ----------
        trainer:
            A trainer instance that contains all information and datasets. The trainer has loaded configs and data.
        program:
            The name of the modelbase. If None, the name from :func:``_get_program_name`` is used.
        model_subset:
            The names of specific models selected to be trained in the modelbase.
        exclude_models:
            The names of specific models that should not be trained. Only one of ``model_subset`` and ``exclude_models`` can
            be specified.
        store_in_harddisk:
            Whether to save sub-models in the hard disk. If the global setting ``low_memory`` is True, True is used.
        **kwargs:
            Ignored.
        """
        self.trainer = trainer
        if not hasattr(trainer, "args"):
            trainer.load_config(config="default")
        self.model = None
        self.leaderboard = None
        self.model_subset = model_subset
        self.exclude_models = exclude_models
        if self.model_subset is not None and self.exclude_models is not None:
            raise Exception(
                f"Only one of model_subset and exclude_models can be specified."
            )
        self.store_in_harddisk = (
            True if src.setting["low_memory"] else store_in_harddisk
        )
        self.program = self._get_program_name() if program is None else program
        self.model_params = {}
        self._check_space()
        self._mkdir()

    @property
    def device(self):
        return self.trainer.device

    def fit(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        model_subset: List[str] = None,
        derived_data: Dict[str, np.ndarray] = None,
        verbose: bool = True,
        warm_start: bool = False,
        bayes_opt: bool = False,
    ):
        """
        Fit models using a tabular dataset. Data of the trainer will be changed.

        Parameters
        ----------
        df:
            A tabular dataset.
        cont_feature_names:
            The names of continuous features.
        cat_feature_names:
            The names of categorical features.
        label_name:
            The name of the target.
        model_subset:
            The names of a subset of all available models (in :func:``get_model_names``). Only these models will be
            trained.
        derived_data:
            Data derived from :func:``DataModule.derive_unstacked``. If not None, unstacked data will be re-derived.
        verbose:
            Verbosity.
        warm_start:
            Whether to train models based on previous trained models.
        bayes_opt:
            Whether to perform Gaussian-process-based Bayesian Hyperparameter Optimization for each model.
        """
        self.trainer.set_status(training=True)
        trainer_state = cp(self.trainer)
        self.trainer.datamodule.set_data(
            df,
            cont_feature_names=cont_feature_names,
            cat_feature_names=cat_feature_names,
            label_name=label_name,
            derived_data=derived_data,
            warm_start=warm_start if self._trained else False,
            verbose=verbose,
            all_training=True,
        )
        if bayes_opt != self.trainer.args["bayes_opt"]:
            self.trainer.args["bayes_opt"] = bayes_opt
            if verbose:
                print(
                    f"The argument bayes_opt of fit() conflicts with Trainer.bayes_opt. Use the former one."
                )
        self.train(
            dump_trainer=False,
            verbose=verbose,
            model_subset=model_subset,
            warm_start=warm_start if self._trained else False,
        )
        self.trainer.load_state(trainer_state)
        self.trainer.set_status(training=False)

    def train(self, *args, **kwargs):
        """
        Training the model using data in the trainer directly.
        The method can be rewritten to implement other training strategies.

        Parameters
        ----------
        *args:
            Arguments of :func:``_train`` for models.
        **kwargs:
            Arguments of :func:``_train`` for models.
        """
        self.trainer.set_status(training=True)
        verbose = "verbose" not in kwargs.keys() or kwargs["verbose"]
        if verbose:
            print(f"\n-------------Run {self.program}-------------\n")
        self._train(*args, **kwargs)
        if self.model is None or len(self.model) == 0:
            warnings.warn(f"No model has been trained for {self.__class__.__name__}.")
        if verbose:
            print(f"\n-------------{self.program} End-------------\n")
        self.trainer.set_status(training=False)

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str,
        derived_data: dict = None,
        ignore_absence: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict a new dataset using the selected model.

        Parameters
        ----------
        df:
            A new tabular dataset.
        model_name:
            A selected name of a model, which is already trained.
        derived_data:
            Data derived from :func:``DataModule.derive_unstacked``. If not None, unstacked data will be re-derived.
        ignore_absence:
            Whether to ignore absent keys in derived_data. Use True only when the model does not use derived_data.
        **kwargs:
            Arguments of :func:``_predict`` for models.

        Returns
        -------
        prediction:
            Predicted target.
        """
        self.trainer.set_status(training=False)
        if self.model is None:
            raise Exception("Run fit() before predict().")
        if model_name not in self.get_model_names():
            raise Exception(
                f"Model {model_name} is not available. Select among {self.get_model_names()}"
            )
        df, derived_data = self.trainer.datamodule.prepare_new_data(
            df, derived_data, ignore_absence
        )
        return self._predict(
            df,
            model_name,
            derived_data,
            **kwargs,
        )

    def detach_model(self, model_name: str, program: str = None) -> "AbstractModel":
        """
        Detach the chosen sub-model to a separate AbstractModel with the same trainer.

        Parameters
        ----------
        model_name:
            The name of the sub-model to be detached.
        program:
            The new name of the detached database. If the name is the same as the original one, the detached model is
            stored in memory to avoid overwriting the original model.

        Returns
        -------
        model:
            An AbstractModel containing the chosen model.
        """
        if not type(self.model) in [ModelDict, Dict]:
            raise Exception(f"The modelbase does not support model detaching.")
        program = program if program is not None else self.program
        tmp_model = cp(self)
        tmp_model.trainer = self.trainer
        tmp_model.program = program
        tmp_model.model_subset = [model_name]
        if tmp_model.store_in_harddisk and program != self.program:
            tmp_model._mkdir()
            tmp_model.model = ModelDict(path=tmp_model.root)
        else:
            tmp_model.store_in_harddisk = False
            tmp_model.model = {}
        tmp_model.model[model_name] = cp(self.model[model_name])
        if model_name in self.model_params.keys():
            tmp_model.model_params[model_name] = cp(self.model_params[model_name])
        return tmp_model

    def set_path(self, path: Union[os.PathLike, str]):
        """
        Set the path of the model base (usually a trained one), including paths of its models. It is used when migrating
        models to another directory.

        Parameters
        ----------
        path
            The path of the model base.
        """
        if hasattr(self, "root"):
            self.root = path
        if self.store_in_harddisk:
            if hasattr(self, "model") and self.model is not None:
                self.model.root = path
                for name in self.model.model_path.keys():
                    self.model.model_path[name] = os.path.join(self.root, name) + ".pkl"

    def new_model(self, model_name: str, verbose: bool, **kwargs):
        """
        A wrapper method to generate a new model while keeping the random seed constant.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to generate the model. It should contain all arguments in :func:``_initial_values``.

        Returns
        -------
        model:
            A new model (without any restriction to its type). It will be passed to :func:``_train_single_model`` and
            :func:``_pred_single_model``.
        """
        set_random_seed(src.setting["random_seed"])
        required_models = self._get_required_models(model_name=model_name)
        if required_models is not None:
            kwargs["required_models"] = required_models
        return self._new_model(model_name=model_name, verbose=verbose, **kwargs)

    def _get_required_models(self, model_name):
        required_model_names = self.required_models(model_name)
        if required_model_names is not None:
            required_models = {}
            for name in required_model_names:
                if name == model_name:
                    raise Exception(f"The model {model_name} is required by itself.")
                if name in self._get_model_names():
                    if name not in self.model.keys():
                        raise Exception(
                            f"Model {name} is required for model {model_name}, but is not trained."
                        )
                    required_models[name] = self.model[name]
                elif name.startswith("EXTERN_"):
                    if len(name.split("_")) != 3:
                        raise Exception(
                            f"Unrecognized required model name {name} from external model bases."
                        )
                    _, program, ext_model_name = name.split("_")
                    try:
                        modelbase = self.trainer.get_modelbase(program=program)
                    except:
                        raise Exception(
                            f"Model base {program} is required for model {model_name}, but does not exist."
                        )
                    try:
                        detached_model = modelbase.detach_model(
                            model_name=ext_model_name
                        )
                    except Exception as e:
                        raise Exception(
                            f"Model {ext_model_name} can not be detached from model base {program}. Exception:\n{e}"
                        )
                    required_models[name] = detached_model
                else:
                    raise Exception(
                        f"Unrecognized model name {name} required by {model_name}."
                    )
            return required_models
        else:
            return None

    def required_models(self, model_name: str) -> Union[List[str], None]:
        """
        The name of the model required by the requested model. If not None and the required model is
        trained, the required model is passed to `_new_model`.
        If models from other model bases are required, the return name should be
        ``EXTERN_{Name of the model base}_{Name of the model}``

        Notes
        -------
        For TorchModel, if the required model is in the TorchModel itself, the AbstractNN is passed to ``_new_model``;
        if the required model is in another model base, the AbstractModel is passed.
        """
        return None

    def _predict_all(
        self, verbose: bool = True, test_data_only: bool = False
    ) -> Dict[str, Dict]:
        """
        Predict training/validation/testing datasets to evaluate the performance of all models.

        Parameters
        ----------
        verbose:
            Verbosity.
        test_data_only:
            Whether to predict only testing datasets. If True, the whole dataset will be evaluated.

        Returns
        -------
        predictions:
            A dict of results. Its keys are "Training", "Testing", and "Validation". Its values are tuples containing
            predicted values and ground truth values
        """
        self.trainer.set_status(training=False)
        self._check_train_status()

        model_names = self.get_model_names()
        data = self.trainer.datamodule

        predictions = {}
        tc = TqdmController()
        tc.disable_tqdm()
        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f"{idx + 1}/{len(model_names)}")
            if not test_data_only:
                y_train_pred = self._predict(
                    data.X_train,
                    derived_data=data.D_train,
                    model_name=model_name,
                )
                y_val_pred = self._predict(
                    data.X_val, derived_data=data.D_val, model_name=model_name
                )
                y_train = data.y_train
                y_val = data.y_val
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = self._predict(
                data.X_test, derived_data=data.D_test, model_name=model_name
            )

            predictions[model_name] = {
                "Training": (y_train_pred, y_train),
                "Testing": (y_test_pred, data.y_test),
                "Validation": (y_val_pred, y_val),
            }

        tc.enable_tqdm()
        return predictions

    def _predict(
        self, df: pd.DataFrame, model_name: str, derived_data: Dict = None, **kwargs
    ) -> np.ndarray:
        """
        Make prediction based on a tabular dataset using the selected model.

        Parameters
        ----------
        df:
            A new tabular dataset that has the same structure as self.trainer.datamodule.X_test.
        model_name:
            A name of a selected model, which is already trained.
        derived_data:
            Data derived from datamodule.derive that has the same structure as self.trainer.datamodule.D_test.
        **kwargs:
            Ignored.

        Returns
        -------
        pred:
            Prediction of the target.
        """
        self.trainer.set_status(training=False)
        X_test = self._data_preprocess(df, derived_data, model_name=model_name)
        return self._pred_single_model(
            self.model[model_name],
            X_test=X_test,
            verbose=False,
        )

    def _train(
        self,
        model_subset: List[str] = None,
        dump_trainer: bool = True,
        verbose: bool = True,
        warm_start: bool = False,
        **kwargs,
    ):
        """
        The basic framework of training models, including processing the dataset, training each model (with/without
        bayesian hyperparameter optimization), and make simple predictions.

        Parameters
        ----------
        model_subset:
            The names of a subset of all available models (in :func:``get_model_names`). Only these models will be
            trained.
        dump_trainer:
            Whether to save the trainer after models are trained.
        verbose:
            Verbosity.
        warm_start:
            Whether to train models based on previous trained models.
        **kwargs:
            Ignored.
        """
        self.trainer.set_status(training=True)
        self.total_epoch = (
            self.trainer.args["epoch"] if not src.setting["debug_mode"] else 2
        )
        if self.model is None:
            if self.store_in_harddisk:
                self.model = ModelDict(path=self.root)
            else:
                self.model = {}
        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if verbose:
                print(f"Training {model_name}")
            data = self._train_data_preprocess(model_name)
            tmp_params = self._get_params(model_name, verbose=verbose)
            space = self._space(model_name=model_name)
            if self.trainer.args["bayes_opt"] and not warm_start and len(space) > 0:
                min_calls = len(tmp_params)
                callback = BayesCallback(
                    total=self.trainer.args["n_calls"]
                    if not src.setting["debug_mode"]
                    else min_calls
                )
                global _bayes_objective

                @skopt.utils.use_named_args(space)
                def _bayes_objective(**params):
                    try:
                        with HiddenPrints():
                            model = self.new_model(
                                model_name=model_name, verbose=False, **params
                            )

                            self._train_single_model(
                                model,
                                epoch=self.trainer.args["bayes_epoch"]
                                if not src.setting["debug_mode"]
                                else 1,
                                X_train=data["X_train"],
                                y_train=data["y_train"],
                                X_val=data["X_val"],
                                y_val=data["y_val"],
                                verbose=False,
                                warm_start=False,
                                in_bayes_opt=True,
                                **params,
                            )

                        res = self._bayes_eval(
                            model,
                            data["X_train"],
                            data["y_train"],
                            data["X_val"],
                            data["y_val"],
                        )
                    except Exception as e:
                        print(
                            f"An exception occurs when evaluating a bayes call: {e}. Returning a large value instead."
                        )
                        res = 100
                    # If a result from one bayes opt iteration is very large (over 10000) caused by instability of the
                    # model, it can not be fully reproduced during another execution and has error (though small, it
                    # disturbs bayes optimization).
                    if res > 1000:
                        print(
                            f"The loss value ({res}) is greater than 1000 and 1000 will be returned. Consider "
                            f"debugging such instability of the model, or check whether the loss value is normalized by"
                            f"the number of samples."
                        )
                        return 1000
                    # To guarantee reproducibility on different machines.
                    return round(res, 4)

                with warnings.catch_warnings():
                    # To obtain clean progress bar.
                    warnings.filterwarnings(
                        "ignore",
                        message="The objective has been evaluated at this point before",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="`pytorch_lightning.utilities.cloud_io.get_filesystem` has been deprecated in v1.8.0 and will be removed in v1.10.0.",
                    )
                    result = gp_minimize(
                        _bayes_objective,
                        self._space(model_name=model_name),
                        n_calls=self.trainer.args["n_calls"]
                        if not src.setting["debug_mode"]
                        else min_calls,
                        n_initial_points=10 if not src.setting["debug_mode"] else 0,
                        callback=callback.call,
                        random_state=0,
                        x0=list(tmp_params.values()),
                    )
                params = {}
                for key, value in zip(tmp_params.keys(), result.x):
                    params[key] = value
                self.model_params[model_name] = cp(params)
                callback.close()
                skopt.dump(
                    result,
                    add_postfix(os.path.join(self.root, f"{model_name}_skopt.pt")),
                )
                tmp_params = self._get_params(
                    model_name=model_name, verbose=verbose
                )  # to announce the optimized params.

            if not warm_start or (warm_start and not self._trained):
                model = self.new_model(
                    model_name=model_name, verbose=verbose, **tmp_params
                )
            else:
                model = self.model[model_name]

            self._train_single_model(
                model,
                epoch=self.total_epoch,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                verbose=verbose,
                warm_start=warm_start,
                in_bayes_opt=False,
                **tmp_params,
            )

            def pred_set(X, y, name):
                pred = self._pred_single_model(model, X, verbose=False)
                mse = metric_sklearn(pred, y, "mse")
                if verbose:
                    print(f"{name} MSE loss: {mse:.5f}, RMSE loss: {np.sqrt(mse):.5f}")

            pred_set(data["X_train"], data["y_train"], "Training")
            pred_set(data["X_val"], data["y_val"], "Validation")
            pred_set(data["X_test"], data["y_test"], "Testing")
            self.model[model_name] = model
            torch.cuda.empty_cache()

        self.trainer.set_status(training=False)
        if dump_trainer:
            save_trainer(self.trainer)

    def _bayes_eval(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        """
        Evaluating the model for bayesian optimization iterations. The validation error is returned directly.

        Returns
        -------
        result
            The evaluation of bayesian hyperparameter optimization.
        """
        y_val_pred = self._pred_single_model(model, X_val, verbose=False)
        val_loss = metric_sklearn(y_val_pred, y_val, self.trainer.args["loss"])
        y_train_pred = self._pred_single_model(model, X_train, verbose=False)
        train_loss = metric_sklearn(y_train_pred, y_train, self.trainer.args["loss"])
        return max([train_loss, val_loss])

    def _check_train_status(self):
        """
        Raise exception if _predict is called and the modelbase is not trained.
        """
        if not self._trained:
            raise Exception(
                f"{self.program} not trained, run {self.__class__.__name__}.train() first."
            )

    def _get_params(self, model_name: str, verbose=True) -> Dict[str, Any]:
        """
        Load default parameters or optimized parameters of the selected model.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity

        Returns
        -------
        params:
            A dict of parameters
        """
        if model_name not in self.model_params.keys():
            return self._initial_values(model_name=model_name)
        else:
            if verbose:
                print(f"Previous params loaded: {self.model_params[model_name]}")
            return self.model_params[model_name]

    @property
    def _trained(self) -> bool:
        if self.model is None:
            return False
        else:
            return True

    def _check_space(self):
        any_mismatch = False
        for model_name in self.get_model_names():
            tmp_params = self._get_params(model_name, verbose=False)
            space = self._space(model_name=model_name)
            for k, s in zip(tmp_params.keys(), space):
                if k != s.name:
                    print(
                        f"Keys of {self.program} - {model_name} in _initial_values and _space does not match.\n"
                        f"_initial_values: {list(tmp_params.keys())}\n"
                        f"_space: {[s.name for s in space]}"
                    )
                    any_mismatch = True
        if any_mismatch:
            raise Exception(f"Defined space and initial values do not match.")

    def _mkdir(self):
        """
        Create a directory for the modelbase under the root of the trainer.
        """
        self.root = os.path.join(self.trainer.project_root, self.program)
        if not os.path.exists(self.root):
            os.mkdir(self.root)

    def get_model_names(self) -> List[str]:
        """
        Get names of available models. It can be selected when initializing the modelbase.

        Returns
        -------
        names:
            Names of available models.
        """
        if self.model_subset is not None:
            for model in self.model_subset:
                if model not in self._get_model_names():
                    raise Exception(f"Model {model} not available for {self.program}.")
            res = self.model_subset
        elif self.exclude_models is not None:
            names = self._get_model_names()
            used_names = [x for x in names if x not in self.exclude_models]
            res = used_names
        else:
            res = self._get_model_names()
        res = [x for x in res if self._conditional_validity(x)]
        return res

    def _get_model_names(self) -> List[str]:
        """
        Get all available models implemented in the modelbase.

        Returns
        -------
        names:
            Names of available models.
        """
        raise NotImplementedError

    def _get_program_name(self) -> str:
        """
        Get the default name of the modelbase.

        Returns
        -------
        name:
            The default name of the modelbase.
        """
        raise NotImplementedError

    # Following methods are for the default _train and _predict methods. If users directly overload _train and _predict,
    # following methods are not required to be implemented.
    def _new_model(self, model_name: str, verbose: bool, **kwargs):
        """
        Generate a new selected model based on kwargs.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to generate the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        model:
            A new model (without any restriction to its type). It will be passed to :func:`_train_single_model` and
            :func:`_pred_single_model`.
        """
        raise NotImplementedError

    def _train_data_preprocess(self, model_name) -> Union[DataModule, dict]:
        """
        Processing the data from self.trainer.datamodule for training.

        Parameters
        -------
        model_name:
            The name of a selected model.

        Returns
        -------
        data
            The returned value should be a ``Dict`` that has the following keys:
            X_train, y_train, X_val, y_val, X_test, y_test.
            Those with postfixes ``_train`` or ``_val`` will be passed to `_train_single_model` and ``_bayes_eval`.
            All of them will be passed to ``_pred_single_model``.

        Notes
        -------
        self.trainer.datamodule.X_train/val/test are not scaled for the sake of further treatments. To scale the df,
        run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        raise NotImplementedError

    def _data_preprocess(
        self, df: pd.DataFrame, derived_data: Dict[str, np.ndarray], model_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Perform the same preprocessing as :func:`_train_data_preprocess` on a new dataset.

        Parameters
        ----------
        df:
            The new tabular dataset that has the same structure as self.trainer.datamodule.X_test
        derived_data:
            Data derived from datamodule.derive that has the same structure as self.trainer.datamodule.D_test.
        model_name:
            The name of a selected model.

        Returns
        -------
        data:
            The processed data (X_test).

        Notes
        -------
        The input df is not scaled for the sake of further treatments. To scale the df,
        run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        raise NotImplementedError

    def _train_single_model(
        self,
        model: Any,
        epoch: Optional[int],
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        y_val: Any,
        verbose: bool,
        warm_start: bool,
        in_bayes_opt: bool,
        **kwargs,
    ):
        """
        Training the model (initialized in :func:`_new_model`).

        Parameters
        ----------
        model:
            The model initialized in :func:`_new_model`.
        epoch:
            Total epochs to train the model.
        X_train:
            The training data from :func:`_train_data_preprocess`.
        y_train:
            The training target from :func:`_train_data_preprocess`.
        X_val:
            The validation data from :func:`_train_data_preprocess`.
        y_val:
            The validation target from :func:`_train_data_preprocess`.
        verbose:
            Verbosity.
        warm_start:
            Whether to train models based on previous trained models.
        in_bayes_opt:
            Whether is in bayes optimization loop.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.
        """
        raise NotImplementedError

    def _pred_single_model(
        self, model: Any, X_test: Any, verbose: bool, **kwargs
    ) -> np.ndarray:
        """
        Predict with the model trained in :func:`_train_single_model`.

        Parameters
        ----------
        model:
            The model trained in :func:`_train_single_model`.
        X_test:
            The testing data from :func:`_data_preprocess`.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        pred:
            Prediction of the target.

        Notes
        -------
        For deep learning models with mini-batch training (dataloaders), if an AbstractWrapper will be used for the model
        base, the ``batch_size`` when inferring should be the length of the dataset. See ``PytorchTabular._pred_single_model``
        and ``WideDeep._pred_single_model``.
        """
        raise NotImplementedError

    def _space(self, model_name: str) -> List[Union[Integer, Real, Categorical]]:
        """
        A list of scikit-optimize search space for the selected model.

        Parameters
        ----------
        model_name:
            The name of a selected model that is currently going through bayes optimization.

        Returns
        -------
        space:
            A list of skopt.space.
        """
        raise NotImplementedError

    def _initial_values(self, model_name: str) -> Dict[str, Union[int, float]]:
        """
        Initial values of hyperparameters to be optimized. The order should be the same as those in :func:`_space`.

        Parameters
        ----------
        model_name:
            The name of a selected model.

        Returns
        -------
        params:
            A dict of initial hyperparameters.
        """
        raise NotImplementedError

    def _conditional_validity(self, model_name: str) -> bool:
        """
        Check the validity of a model.

        Parameters
        ----------
        model_name:
            The name of a model in _get_model_names().

        Returns
        -------
            Whether the model is valid for training under certain settings.
        """
        return True


class BayesCallback:
    """
    Print information when performing bayes optimization.
    """

    def __init__(self, total):
        self.total = total
        self.cnt = 0
        self.init_time = time.time()
        self.postfix = {
            "ls": 1e8,
            "param": [],
            "min ls": 1e8,
            "min param": [],
            "min at": 0,
        }

    def call(self, result):
        self.postfix["ls"] = result.func_vals[-1]
        self.postfix["param"] = [
            round(x, 5) if hasattr(x, "__round__") else x for x in result.x_iters[-1]
        ]
        if result.fun < self.postfix["min ls"]:
            self.postfix["min ls"] = result.fun
            self.postfix["min param"] = [
                round(x, 5) if hasattr(x, "__round__") else x for x in result.x
            ]
            self.postfix["min at"] = len(result.func_vals)
        self.cnt += 1
        tot_time = time.time() - self.init_time
        print(
            f"Bayes-opt {self.cnt}/{self.total}, tot {tot_time:.2f}s, avg {tot_time/self.cnt:.2f}s/it: {self.postfix}"
        )

    def close(self):
        torch.cuda.empty_cache()


class TorchModel(AbstractModel):
    """
    The specific class for PyTorch-like models. Some abstract methods in AbstractModel are implemented.
    """

    def _train_data_preprocess(self, model_name):
        datamodule = self._prepare_custom_datamodule(model_name)
        train_dataset, val_dataset, test_dataset = self._generate_dataset(
            datamodule, model_name
        )
        return {
            "X_train": train_dataset,
            "y_train": datamodule.y_train,
            "X_val": val_dataset,
            "y_val": datamodule.y_val,
            "X_test": test_dataset,
            "y_test": datamodule.y_test,
        }

    def _prepare_custom_datamodule(self, model_name) -> DataModule:
        """
        Change this method if a customized preprocessing stage is needed.

        Notes
        -------
        Call datamodule._update_dataset_auto() after manual modifications on the datamodule.
        See ``transformer.py`` for example.
        """
        return self.trainer.datamodule

    def _generate_dataset(self, datamodule, model_name):
        required_models = self._get_required_models(model_name)
        if required_models is None:
            train_dataset, val_dataset, test_dataset = (
                datamodule.train_dataset,
                datamodule.val_dataset,
                datamodule.test_dataset,
            )
        else:
            dataset = self._generate_dataset_for_required_models(
                df=datamodule.df,
                derived_data=datamodule.derived_data,
                tensors=datamodule.tensors,
                required_models=required_models,
            )
            train_dataset, val_dataset, test_dataset = datamodule.generate_subset(
                dataset
            )
        return train_dataset, val_dataset, test_dataset

    def _generate_dataset_for_required_models(
        self, df, derived_data, tensors, required_models
    ):
        full_data_required_models = {
            name: mod._data_preprocess(
                df=df,
                derived_data=derived_data,
                model_name=name,
            )
            for name, mod in required_models.items()
            if not isinstance(mod, AbstractNN)
        }
        for mod, val in zip(
            required_models.values(), full_data_required_models.values()
        ):
            if (
                not isinstance(val, pd.DataFrame)
                and not isinstance(val, np.ndarray)
                and not isinstance(mod, TorchModel)
            ):
                raise Exception(
                    f"Requiring a model (except TorchModel) that does not return a pd.DataFrame or np.ndarray by "
                    f"_data_preprocess is currently not supported."
                )
        tensor_dataset = Data.TensorDataset(*tensors)
        if len(full_data_required_models) == 0:
            dataset = tensor_dataset
        else:
            dict_df_dataset = DictMixDataset(full_data_required_models)
            dataset = DictDataset(
                ListDataset([tensor_dataset, dict_df_dataset]),
                keys=["self", "required"],
            )
        return dataset

    def _run_custom_data_module(self, df, derived_data, model_name):
        """
        Change this method if a customized preprocessing stage is implemented in ``_prepare_custom_datamodule``. The
        returned datamodule only provide feature names and a scaler (AbstractScaler). See ``transformer.py`` for example.
        """
        return df, derived_data, self.trainer.datamodule

    def _data_preprocess(self, df, derived_data, model_name):
        df, derived_data, datamodule = self._run_custom_data_module(
            df, derived_data, model_name
        )
        scaled_df = datamodule.data_transform(df, scaler_only=True)
        X = torch.tensor(
            scaled_df[datamodule.cont_feature_names].values.astype(np.float32),
            dtype=torch.float32,
        )
        D = [
            torch.tensor(value, dtype=torch.float32) for value in derived_data.values()
        ]
        y = torch.tensor(
            np.zeros((len(scaled_df), len(datamodule.label_name))),
            dtype=torch.float32,
        )
        tensors = (X, *D, y)
        required_models = self._get_required_models(model_name)
        if required_models is None:
            dataset = Data.TensorDataset(*tensors)
        else:
            dataset = self._generate_dataset_for_required_models(
                df=df.reset_index(drop=True),  # Use the unscaled one here
                derived_data=derived_data,
                tensors=tensors,
                required_models=required_models,
            )
        return dataset

    def _train_single_model(
        self,
        model: "AbstractNN",
        epoch,
        X_train,
        y_train,
        X_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        if not isinstance(model, AbstractNN):
            raise Exception(
                f"_new_model must return an AbstractNN instance, but got {model}."
            )

        warnings.filterwarnings(
            "ignore", "The dataloader, val_dataloader 0, does not have many workers"
        )
        warnings.filterwarnings(
            "ignore", "The dataloader, train_dataloader, does not have many workers"
        )
        warnings.filterwarnings("ignore", "Checkpoint directory")

        train_loader = Data.DataLoader(
            X_train,
            batch_size=int(kwargs["batch_size"]),
            sampler=torch.utils.data.RandomSampler(
                data_source=X_train, replacement=False
            ),
            pin_memory=True,
        )
        val_loader = Data.DataLoader(
            X_val,
            batch_size=len(X_val),
            pin_memory=True,
        )

        es_callback = EarlyStopping(
            monitor="early_stopping_eval",
            min_delta=0.001,
            patience=self.trainer.static_params["patience"],
            mode="min",
        )
        ckpt_callback = ModelCheckpoint(
            monitor="early_stopping_eval",
            dirpath=self.root,
            filename="early_stopping_ckpt",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
        )
        trainer = pl.Trainer(
            max_epochs=epoch,
            min_epochs=1,
            callbacks=[
                PytorchLightningLossCallback(verbose=True, total_epoch=epoch),
                es_callback,
                ckpt_callback,
            ],
            fast_dev_run=False,
            max_time=None,
            gpus=None,
            accelerator="auto",
            devices=None,
            accumulate_grad_batches=1,
            auto_lr_find=False,
            auto_select_gpus=True,
            check_val_every_n_epoch=1,
            gradient_clip_val=0.0,
            overfit_batches=0.0,
            deterministic=False,
            profiler=None,
            logger=False,
            track_grad_norm=-1,
            precision=32,
            enable_checkpointing=True,
            enable_progress_bar=False,
        )

        ckpt_path = os.path.join(self.root, "early_stopping_ckpt.ckpt")
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

        with HiddenPrints(
            disable_std=not verbose,
            disable_logging=not verbose,
        ):
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

        model.to("cpu")
        model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
        trainer.strategy.remove_checkpoint(
            os.path.join(self.root, "early_stopping_ckpt.ckpt")
        )
        # pl.Trainer is not pickle-able. When pickling, "ReferenceError: weakly-referenced object no longer exists."
        # may be raised occasionally. Set the trainer to None.
        # https://deepforest.readthedocs.io/en/latest/FAQ.html
        model.trainer = None
        torch.cuda.empty_cache()

    def _pred_single_model(self, model: "AbstractNN", X_test, verbose, **kwargs):
        test_loader = Data.DataLoader(
            X_test,
            batch_size=len(X_test),
            shuffle=False,
            pin_memory=True,
        )
        model.to(self.device)
        y_test_pred, _, _ = model.test_epoch(test_loader, **kwargs)
        model.to("cpu")
        torch.cuda.empty_cache()
        return y_test_pred

    def _space(self, model_name):
        return self.trainer.SPACE

    def _initial_values(self, model_name):
        return self.trainer.chosen_params

    def count_params(self, model_name, trainable_only=False):
        if self.model is not None and model_name in self.model.keys():
            model = self.model[model_name]
        else:
            model = self.new_model(
                model_name, verbose=False, **self._get_params(model_name, verbose=False)
            )
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())


class AbstractWrapper:
    """
    For those deep learning models required by TorchModel, this is a wrapper to make them have hidden information like
    ``hidden_representation`` or something else from the forward process.
    """

    def __init__(self, model: AbstractModel):
        if len(model.get_model_names()) > 1:
            raise Exception(
                f"More than one model is included in the input model base: {model.get_model_names()}."
            )
        self.wrapped_model = model
        self.model_name = self.wrapped_model.get_model_names()[0]
        self.original_forward = None
        self.wrap_forward()

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            return getattr(self.wrapped_model, item)

    def eval(self):
        pass

    def __call__(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simulate ``AbstractNN._forward`` by calling ``AbstractNN._call_required_model``.
        """
        return AbstractNN._call_required_model(self.wrapped_model, x, derived_tensors)

    def wrap_forward(self):
        raise NotImplementedError

    def reset_forward(self):
        raise NotImplementedError

    @property
    def hidden_rep_dim(self):
        raise NotImplementedError

    @property
    def hidden_representation(self):
        raise NotImplementedError

    def __getstate__(self):
        self.reset_forward()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.wrap_forward()


class TorchModelWrapper(AbstractWrapper):
    def __init__(self, model: TorchModel):
        super(TorchModelWrapper, self).__init__(model=model)

    def wrap_forward(self):
        pass

    @property
    def hidden_rep_dim(self):
        return self.wrapped_model.model[self.model_name].hidden_rep_dim

    @property
    def hidden_representation(self):
        return self.wrapped_model.model[self.model_name].hidden_representation


class AbstractNN(pl.LightningModule):
    def __init__(self, datamodule: DataModule, **kwargs):
        """
        PyTorch model that contains derived data names and dimensions from the trainer.

        Parameters
        ----------
        datamodule:
            A DataModule instance.
        """
        super(AbstractNN, self).__init__()
        self.default_loss_fn = nn.MSELoss()
        self.cont_feature_names = cp(datamodule.cont_feature_names)
        self.cat_feature_names = cp(datamodule.cat_feature_names)
        self.n_cont = len(self.cont_feature_names)
        self.n_cat = len(self.cat_feature_names)
        self.derived_feature_names = list(datamodule.derived_data.keys())
        self.derived_feature_dims = datamodule.get_derived_data_sizes()
        self.derived_feature_names_dims = {}
        self.automatic_optimization = False
        self.hidden_representation = None
        self.hidden_rep_dim = None
        if len(kwargs) > 0:
            self.save_hyperparameters(
                *list(kwargs.keys()),
                ignore=["trainer", "datamodule", "required_models"],
            )
        for name, dim in zip(
            datamodule.derived_data.keys(), datamodule.get_derived_data_sizes()
        ):
            self.derived_feature_names_dims[name] = dim
        self._device_var = nn.Parameter(torch.empty(0, requires_grad=False))

    @property
    def device(self):
        return self._device_var.device

    def forward(
        self,
        *tensors: torch.Tensor,
        data_required_models: Dict[str, pd.DataFrame] = None,
    ) -> torch.Tensor:
        """
        A wrapper of the original forward of nn.Module. Input data are tensors with no names, but their names are
        obtained during initialization, so that a dict of derived data with names is generated and passed to
        :func:`_forward`.

        Parameters
        ----------
        tensors:
            Input tensors to the torch model.
        data_required_models:
            The corresponding data processed by the required models (see ``AbstractModel.required_models`` and
            ``AbstractModel._data_preprocess``).

        Returns
        -------
        result:
            The obtained tensor.
        """
        with torch.no_grad() if src.setting[
            "test_with_no_grad"
        ] and not self.training else torch_with_grad():
            x = tensors[0]
            additional_tensors = tensors[1:]
            if type(additional_tensors[0]) == dict:
                derived_tensors = additional_tensors[0]
            else:
                derived_tensors = {}
                for tensor, name in zip(additional_tensors, self.derived_feature_names):
                    derived_tensors[name] = tensor
            if data_required_models is not None:
                derived_tensors["data_required_models"] = data_required_models
            res = self._forward(x, derived_tensors)
            if len(res.shape) == 1:
                res = res.view(-1, 1)
            if self.hidden_representation is None:
                self.hidden_representation = res
            if self.hidden_rep_dim is None:
                self.hidden_rep_dim = res.shape[1]
            return res

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if type(batch) == dict:
            tensors, data_required_models = batch["self"], batch["required"]
        else:
            tensors, data_required_models = batch, None
        self.cal_zero_grad()
        yhat = tensors[-1]
        data = tensors[0]
        additional_tensors = [x for x in tensors[1 : len(tensors) - 1]]
        y = self(
            *([data] + additional_tensors), data_required_models=data_required_models
        )
        loss = self.loss_fn(yhat, y, *([data] + additional_tensors))
        self.cal_backward_step(loss)
        mse = self.default_loss_fn(yhat, y)
        self.log(
            "train_mean_squared_error",
            mse.item(),
            on_step=False,
            on_epoch=True,
            batch_size=y.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) == dict:
            tensors, data_required_models = batch["self"], batch["required"]
        else:
            tensors, data_required_models = batch, None
        with torch.no_grad():
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = [x for x in tensors[1 : len(tensors) - 1]]
            y = self(
                *([data] + additional_tensors),
                data_required_models=data_required_models,
            )
            mse = self.default_loss_fn(yhat, y)
            self.log(
                "valid_mean_squared_error",
                mse.item(),
                on_step=False,
                on_epoch=True,
                batch_size=y.shape[0],
            )
        return yhat, y

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def test_epoch(
        self, test_loader: Data.DataLoader, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate a torch.nn.Module model in a single epoch.

        Parameters
        ----------
        test_loader:
            The DataLoader of the testing dataset.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        results:
            The prediction, ground truth, and loss of the model on the testing dataset.
        """
        self.eval()
        pred = []
        truth = []
        with torch.no_grad() if src.setting["test_with_no_grad"] else torch_with_grad():
            # print(test_dataset)
            avg_loss = 0
            for idx, batch in enumerate(test_loader):
                if type(batch) == dict:
                    tensors, data_required_models = batch["self"], batch["required"]
                else:
                    tensors, data_required_models = batch, None
                yhat = tensors[-1].to(self.device)
                data = tensors[0].to(self.device)
                additional_tensors = [
                    x.to(self.device) for x in tensors[1 : len(tensors) - 1]
                ]
                y = self(
                    *([data] + additional_tensors),
                    data_required_models=data_required_models,
                )
                loss = self.default_loss_fn(y, yhat)
                avg_loss += loss.item() * len(y)
                pred += list(y.cpu().detach().numpy())
                truth += list(yhat.cpu().detach().numpy())
            avg_loss /= len(test_loader.dataset)
        return np.array(pred), np.array(truth), avg_loss

    def loss_fn(self, y_true, y_pred, *data, **kwargs):
        """
        User defined loss function.

        Parameters
        ----------
        y_true:
            Ground truth value.
        y_pred:
            Predicted value by the model.
        *data:
            Tensors of continuous data and derived data.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        loss:
            A torch-like loss.
        """
        return self.default_loss_fn(y_pred, y_true)

    def cal_zero_grad(self):
        """
        Call optimizer.zero_grad() of the optimizer initialized in `init_optimizer`.
        """
        opt = self.optimizers()
        if isinstance(opt, list):
            for o in opt:
                o.zero_grad()
        else:
            opt.zero_grad()

    def cal_backward_step(self, loss):
        """
        Call loss.backward() and optimizer.step().

        Parameters
        ----------
        loss
            The loss returned by `loss_fn`.
        """
        self.manual_backward(loss)
        opt = self.optimizers()
        opt.step()

    def set_requires_grad(
        self, model: nn.Module, requires_grad: bool = None, state=None
    ):
        if (requires_grad is None and state is None) or (
            requires_grad is not None and state is not None
        ):
            raise Exception(
                f"One of `requires_grad` and `state` should be specified to determine the action. If `requires_grad` is "
                f"not None, requires_grad of all parameters in the model is set. If state is not None, state of "
                f"requires_grad in the model is restored."
            )
        if state is not None:
            for s, param in zip(state, model.parameters()):
                param.requires_grad_(s)
        else:
            state = []
            for param in model.parameters():
                state.append(param.requires_grad)
                param.requires_grad_(requires_grad)
            return state

    def _early_stopping_eval(self, train_loss: float, val_loss: float) -> float:
        """
        Calculate the loss value (criteria) for early stopping. The validation loss is returned directly.

        Parameters
        ----------
        train_loss
            Training loss at the epoch.
        val_loss
            Validation loss at the epoch.

        Returns
        -------
        result
            The early stopping evaluation.
        """
        return val_loss

    @staticmethod
    def _test_required_model(
        n_inputs: int,
        required_model: Union[AbstractModel, "AbstractNN", AbstractWrapper],
    ) -> Tuple[bool, int]:
        """
        Test whether a required model has attribute ``hidden_rep_dim`` and find its value.

        Parameters
        ----------
        n_inputs
            The dimension of input features (i.e. x of _forward)
        required_model
            A required model specified in ``required_models()``.

        Returns
        -------
        use_hidden_rep, hidden_rep_dim
            Whether the required model has ``hidden_rep_dim`` and its value. If the required model does not have `
            `hidden_rep_dim``, 1+``n_inputs`` is returned.
        """
        if isinstance(required_model, AbstractWrapper):
            hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            use_hidden_rep = True
        elif not hasattr(required_model, "hidden_representation") or not hasattr(
            required_model, "hidden_rep_dim"
        ):
            if not hasattr(required_model, "hidden_rep_dim"):
                print(
                    f"`hidden_rep_dim` is not given. The output of the backbone and the input features are used instead."
                )
                hidden_rep_dim = 1 + n_inputs
            else:
                hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            if not hasattr(required_model, "hidden_representation") or not hasattr(
                required_model, "hidden_rep_dim"
            ):
                print(
                    f"The backbone should have an attribute called `hidden_representation` that records the "
                    f"final output of the hidden layer, and `hidden_rep_dim` that records the dim of "
                    f"`hidden_representation`. Now the output of the backbone is used instead."
                )
            use_hidden_rep = False
        else:
            hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            use_hidden_rep = True
        return use_hidden_rep, hidden_rep_dim

    @staticmethod
    def _call_required_model(required_model, x, derived_tensors) -> torch.Tensor:
        """
        Call a required model and return its result.

        Parameters
        ----------
        required_model
            A required model specified in ``required_models()``.
        x
            See AbstractNN._forward.
        derived_tensors
            See AbstractNN._forward.

        Returns
        -------
        dl_pred
            The result of the required model.
        """
        if isinstance(required_model, nn.Module) or isinstance(
            required_model, AbstractWrapper
        ):
            required_model.eval()
            dl_pred = required_model(x, derived_tensors)
        elif isinstance(required_model, AbstractModel):
            name = required_model.get_model_names()[0]
            full_name = f"EXTERN_{required_model.program}_{name}"
            ml_pred = required_model._pred_single_model(
                required_model.model[name],
                X_test=derived_tensors["data_required_models"][full_name],
                verbose=False,
            )
            dl_pred = torch.tensor(ml_pred, device=x.device)
        else:
            raise Exception(
                f"The required model should be a nn.Module, an AbstractWrapper, or an AbstractModel, but got"
                f"{type(required_model)} instead."
            )
        return dl_pred


class ModelDict:
    def __init__(self, path):
        self.root = path
        self.model_path = {}

    def __setitem__(self, key, value):
        self.model_path[key] = os.path.join(self.root, key) + ".pkl"
        with open(self.model_path[key], "wb") as file:
            pickle.dump((key, value), file, pickle.HIGHEST_PROTOCOL)
        del value
        torch.cuda.empty_cache()

    def __getitem__(self, item):
        torch.cuda.empty_cache()
        with open(self.model_path[item], "rb") as file:
            key, model = pickle.load(file)
        return model

    def __len__(self):
        return len(self.model_path)

    def keys(self):
        return self.model_path.keys()


def init_weights(m, nonlinearity="leaky_relu"):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)


def get_sequential(
    layers,
    n_inputs,
    n_outputs,
    act_func,
    dropout=0,
    use_norm=True,
    norm_type="batch",
    out_activate=False,
    out_norm_dropout=False,
):
    net = nn.Sequential()
    if norm_type == "batch":
        norm = nn.BatchNorm1d
    elif norm_type == "layer":
        norm = nn.LayerNorm
    else:
        raise Exception(f"Normalization {norm_type} not implemented.")
    if act_func == nn.ReLU:
        nonlinearity = "relu"
    elif act_func == nn.LeakyReLU:
        nonlinearity = "leaky_relu"
    else:
        nonlinearity = "leaky_relu"
    if len(layers) > 0:
        if use_norm:
            net.add_module(f"norm_0", norm(n_inputs))
        net.add_module(
            "input", get_linear(n_inputs, layers[0], nonlinearity=nonlinearity)
        )
        net.add_module("activate_0", act_func())
        if dropout != 0:
            net.add_module(f"dropout_0", nn.Dropout(dropout))
        for idx in range(1, len(layers)):
            if use_norm:
                net.add_module(f"norm_{idx}", norm(layers[idx - 1]))
            net.add_module(
                str(idx),
                get_linear(layers[idx - 1], layers[idx], nonlinearity=nonlinearity),
            )
            net.add_module(f"activate_{idx}", act_func())
            if dropout != 0:
                net.add_module(f"dropout_{idx}", nn.Dropout(dropout))
        if out_norm_dropout and use_norm:
            net.add_module(f"norm_out", norm(layers[-1]))
        net.add_module(
            "output", get_linear(layers[-1], n_outputs, nonlinearity=nonlinearity)
        )
        if out_activate:
            net.add_module("activate_out", act_func())
        if out_norm_dropout and dropout != 0:
            net.add_module(f"dropout_out", nn.Dropout(dropout))
    else:
        if use_norm:
            net.add_module("norm", norm(n_inputs))
        net.add_module("single_layer", nn.Linear(n_inputs, n_outputs))
        net.add_module("activate", act_func())
        if dropout != 0:
            net.add_module("dropout", nn.Dropout(dropout))

    net.apply(partial(init_weights, nonlinearity=nonlinearity))
    return net


def get_linear(n_inputs, n_outputs, nonlinearity="leaky_relu"):
    linear = nn.Linear(n_inputs, n_outputs)
    init_weights(linear, nonlinearity=nonlinearity)
    return linear


class PytorchLightningLossCallback(Callback):
    def __init__(self, verbose, total_epoch):
        super(PytorchLightningLossCallback, self).__init__()
        self.val_ls = []
        self.es_val_ls = []
        self.verbose = verbose
        self.total_epoch = total_epoch
        self.start_time = 0

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        train_loss = logs["train_mean_squared_error"].detach().cpu().numpy()
        val_loss = logs["valid_mean_squared_error"].detach().cpu().numpy()
        self.val_ls.append(val_loss)
        if hasattr(pl_module, "_early_stopping_eval"):
            early_stopping_eval = pl_module._early_stopping_eval(
                trainer.logged_metrics["train_mean_squared_error"],
                trainer.logged_metrics["valid_mean_squared_error"],
            ).item()
            pl_module.log("early_stopping_eval", early_stopping_eval)
            self.es_val_ls.append(early_stopping_eval)
        else:
            early_stopping_eval = None
        epoch = trainer.current_epoch
        if (
            (epoch + 1) % src.setting["verbose_per_epoch"] == 0 or epoch == 0
        ) and self.verbose:
            if early_stopping_eval is not None:
                print(
                    f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                    f"Min val loss: {np.min(self.val_ls):.4f}, Min ES val loss: {np.min(self.es_val_ls):.4f}, "
                    f"Epoch time: {time.time()-self.start_time:.3f}s."
                )
            else:
                print(
                    f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                    f"Min val loss: {np.min(self.val_ls):.4f}, Epoch time: {time.time() - self.start_time:.3f}s."
                )


class DataFrameDataset(Data.Dataset):
    def __init__(self, df: pd.DataFrame):
        # If predicting for a new dataframe, the index might be a mess.
        self.df = df.reset_index(drop=True)
        self.df_dict = {
            key: row[1] for key, row in zip(self.df.index, self.df.iterrows())
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df_dict[item]


class NDArrayDataset(Data.Dataset):
    def __init__(self, array: np.ndarray):
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, item):
        return self.array[item]


class SubsetDataset(Data.Dataset):
    def __init__(self, dataset: Data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return Data.Subset(self.dataset, [item])


class ListDataset(Data.Dataset):
    def __init__(self, datasets: List[Data.Dataset]):
        self.datasets = datasets
        for dataset in self.datasets:
            if len(dataset) != len(self.datasets[0]):
                raise Exception(f"All datasets should have the equal length.")

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item):
        return [dataset.__getitem__(item) for dataset in self.datasets]


class DictDataset(Data.Dataset):
    def __init__(self, ls_dataset: ListDataset, keys: List[str]):
        self.keys = keys
        self.datasets = ls_dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return {
            key: data for key, data in zip(self.keys, self.datasets.__getitem__(item))
        }


class DictDataFrameDataset(DictDataset):
    def __init__(self, dict_dfs: Dict[str, pd.DataFrame]):
        keys = list(dict_dfs.keys())
        df_ls = list(dict_dfs.values())
        ls_dataset = ListDataset([DataFrameDataset(df) for df in df_ls])
        super(DictDataFrameDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)


class DictNDArrayDataset(DictDataset):
    def __init__(self, dict_array: Dict[str, np.ndarray]):
        keys = list(dict_array.keys())
        array_ls = list(dict_array.values())
        ls_dataset = ListDataset([NDArrayDataset(array) for array in array_ls])
        super(DictNDArrayDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)


class DictMixDataset(DictDataset):
    def __init__(self, dict_mix: Dict[str, Union[pd.DataFrame, np.ndarray]]):
        keys = list(dict_mix.keys())
        item_ls = list(dict_mix.values())
        ls_data = []
        for item in item_ls:
            if isinstance(item, pd.DataFrame):
                ls_data.append(DataFrameDataset(item))
            elif isinstance(item, np.ndarray):
                ls_data.append(NDArrayDataset(item))
            elif isinstance(item, torch.Tensor):
                ls_data.append(Data.TensorDataset(item))
            elif isinstance(item, Data.Dataset):
                ls_data.append(SubsetDataset(item))
            else:
                raise Exception(
                    f"Generating a mixed type dataset for type {type(item)}."
                )

        ls_dataset = ListDataset(ls_data)
        super(DictMixDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)
