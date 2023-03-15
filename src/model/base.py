import pickle
import torch.optim.optimizer
import src
from src.utils import *
from src.trainer import Trainer, save_trainer
import skopt
from skopt import gp_minimize
import torch.utils.data as Data
import torch.nn as nn
from copy import deepcopy as cp
from typing import *
from skopt.space import Real, Integer, Categorical
from tqdm.auto import tqdm
import time
from contextlib import nullcontext


class AbstractModel:
    def __init__(
        self,
        trainer: Trainer = None,
        program: str = None,
        model_subset: List[str] = None,
        low_memory: bool = True,
        **kwargs,
    ):
        """
        The base class for all model-bases.

        Parameters
        ----------
        trainer:
            A trainer instance that contains all information and datasets. The trainer has loaded configs and data.
        program:
            The name of the modelbase. If None, the name from :func:`_get_program_name` is used.
        model_subset:
            The names of specific models selected in the modelbase. Only these models will be trained.
        low_memory:
            Whether to save sub-models directly in a Dict (memory). If True, they will be saved locally. If the device
            is `cpu`, low_memory=False is used.
        """
        self.device = trainer.device
        self.trainer = trainer
        if not hasattr(trainer, "database"):
            trainer.load_config(default_configfile="base_config")
        self.model = None
        self.leaderboard = None
        self.model_subset = model_subset
        self.low_memory = low_memory and trainer.device == "cpu"
        self.program = self._get_program_name() if program is None else program
        self.model_params = {}
        self._check_space()
        self._mkdir()

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
            The names of a subset of all available models (in :func:`get_model_names`). Only these models will be
            trained.
        derived_data:
            Data derived from :func:`Trainer.derive_unstacked`. If not None, unstacked data will be re-derived.
        verbose:
            Verbosity.
        warm_start:
            Whether to train models based on previous trained models.
        bayes_opt:
            Whether to perform Gaussian-process-based Bayesian Hyperparameter Optimization for each model.
        """
        self.trainer.set_data(
            df,
            cont_feature_names=cont_feature_names,
            cat_feature_names=cat_feature_names,
            label_name=label_name,
            derived_data=derived_data,
            warm_start=warm_start if self._trained else False,
            verbose=verbose,
            all_training=True,
        )
        if bayes_opt != self.trainer.bayes_opt:
            self.trainer.bayes_opt = bayes_opt
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

    def train(self, *args, **kwargs):
        """
        Training the model using data in the trainer directly.
        The method can be rewritten to implement other training strategies.

        Parameters
        ----------
        *args:
            Arguments of :func:`_train` for models.
        **kwargs:
            Arguments of :func:`_train` for models.
        """

        verbose = "verbose" not in kwargs.keys() or kwargs["verbose"]
        if verbose:
            print(f"\n-------------Run {self.program}-------------\n")
        self._train(*args, **kwargs)
        if verbose:
            print(f"\n-------------{self.program} End-------------\n")

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
            Data derived from :func:`Trainer.derive_unstacked`. If not None, unstacked data will be re-derived.
        ignore_absence:
            Whether to ignore absent keys in derived_data. Use True only when the model does not use derived_data.
        **kwargs:
            Arguments of :func:`_predict` for models.

        Returns
        -------
        prediction:
            Predicted target.
        """
        if self.model is None:
            raise Exception("Run fit() before predict().")
        if model_name not in self.get_model_names():
            raise Exception(
                f"Model {model_name} is not available. Select among {self.get_model_names()}"
            )
        absent_features = [
            x
            for x in np.setdiff1d(
                self.trainer.all_feature_names, self.trainer.derived_stacked_features
            )
            if x not in df.columns
        ]
        absent_derived_features = [
            x for x in self.trainer.derived_stacked_features if x not in df.columns
        ]
        if len(absent_features) > 0:
            raise Exception(f"Feature {absent_features} not in the input dataframe.")

        if derived_data is None or len(absent_derived_features) > 0:
            df, _, derived_data = self.trainer.derive(df)
        else:
            absent_keys = [
                key
                for key in self.trainer.derived_data.keys()
                if key not in derived_data.keys()
            ]
            if len(absent_keys) > 0 and not ignore_absence:
                raise Exception(
                    f"Additional feature {absent_keys} not in the input derived_data."
                )
        df = self.trainer.dataimputer.transform(df.copy(), self.trainer)
        return self._predict(
            df,
            model_name,
            self.trainer.sort_derived_data(derived_data, ignore_absence=ignore_absence),
            **kwargs,
        )

    def detach_model(self, model_name: str, program: str):
        """
        Detach the chosen submodel to a seperate AbstractModel.
        Parameters
        ----------
        model_name:
            The name of the submodel to be detached.
        program:
            The new name of the detached database.

        Returns
        -------
        model:
            An AbstractModel containing the chosen model.
        """
        if not type(self.model) in [ModelDict, Dict]:
            raise Exception(f"The modelbase does not support model detaching.")
        tmp_model = self.__class__(
            trainer=self.trainer, program=program, model_subset=[model_name]
        )
        if type(self.model) == ModelDict:
            tmp_model.model = ModelDict(path=tmp_model.root)
        else:
            tmp_model.model = {}
        tmp_model.model[model_name] = cp(self.model[model_name])
        if model_name in self.model_params.keys():
            tmp_model.model_params[model_name] = cp(self.model_params[model_name])
        return tmp_model

    def new_model(self, model_name: str, verbose: bool, **kwargs):
        set_random_seed(0)
        return self._new_model(model_name=model_name, verbose=verbose, **kwargs)

    def _base_train_data_preprocess(
        self,
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, np.ndarray],
        np.ndarray,
        pd.DataFrame,
        Dict[str, np.ndarray],
        np.ndarray,
        pd.DataFrame,
        Dict[str, np.ndarray],
        np.ndarray,
    ]:
        """
        Load tabular training/validation/testing datasets from the trainer.

        Returns
        -------
        datasets:
            The training datasets: X_train, D_train, y_train;
            The validation datasets: X_val, D_val, y_val;
            The testing datasets: X_test, D_test, y_test
        """
        label_name = self.trainer.label_name
        df = self.trainer.df
        train_indices = self.trainer.train_indices
        val_indices = self.trainer.val_indices
        test_indices = self.trainer.test_indices
        X_train = df.loc[train_indices, :].copy()
        X_val = df.loc[val_indices, :].copy()
        X_test = df.loc[test_indices, :].copy()
        y_train = df.loc[train_indices, label_name].values
        y_val = df.loc[val_indices, label_name].values
        y_test = df.loc[test_indices, label_name].values
        D_train = self.trainer.get_derived_data_slice(
            derived_data=self.trainer.derived_data, indices=self.trainer.train_indices
        )
        D_val = self.trainer.get_derived_data_slice(
            derived_data=self.trainer.derived_data, indices=self.trainer.val_indices
        )
        D_test = self.trainer.get_derived_data_slice(
            derived_data=self.trainer.derived_data, indices=self.trainer.test_indices
        )
        return X_train, D_train, y_train, X_val, D_val, y_val, X_test, D_test, y_test

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
        self._check_train_status()

        model_names = self.get_model_names()
        (
            X_train,
            D_train,
            y_train,
            X_val,
            D_val,
            y_val,
            X_test,
            D_test,
            y_test,
        ) = self._base_train_data_preprocess()

        predictions = {}
        disable_tqdm()
        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f"{idx + 1}/{len(model_names)}")
            if not test_data_only:
                y_train_pred = self._predict(
                    X_train,
                    derived_data=D_train,
                    model_name=model_name,
                )
                y_val_pred = self._predict(
                    X_val, derived_data=D_val, model_name=model_name
                )
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = self._predict(
                X_test, derived_data=D_test, model_name=model_name
            )

            predictions[model_name] = {
                "Training": (y_train_pred, y_train),
                "Testing": (y_test_pred, y_test),
                "Validation": (y_val_pred, y_val),
            }

        enable_tqdm()
        return predictions

    def _predict(
        self, df: pd.DataFrame, model_name: str, derived_data: Dict = None, **kwargs
    ) -> np.ndarray:
        """
        Make prediction based on a tabular dataset using the selected model.

        Parameters
        ----------
        df:
            A new tabular dataset.
        model_name:
            A name of a selected model, which is already trained.
        derived_data:
            Data derived from :func:`Trainer.derive_unstacked`. If not None, unstacked data will be re-derived.
        **kwargs:
            Ignored.

        Returns
        -------
        pred:
            Prediction of the target.
        """
        X_df, derived_data = self._data_preprocess(
            df, derived_data, model_name=model_name
        )
        return self._pred_single_model(
            self.model[model_name],
            X_test=X_df,
            D_test=derived_data,
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
            The names of a subset of all available models (in :func:`get_model_names`). Only these models will be
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
        # disable_tqdm()
        warnings.filterwarnings(
            "ignore",
            message="`np.int` is a deprecated alias for the builtin `int`.",
        )
        data = self._base_train_data_preprocess()
        (
            X_train,
            D_train,
            y_train,
            X_val,
            D_val,
            y_val,
            X_test,
            D_test,
            y_test,
        ) = self._train_data_preprocess(*data)
        self.total_epoch = self.trainer.args["epoch"]
        if self.model is None:
            if self.low_memory:
                self.model = ModelDict(path=self.root)
            else:
                self.model = {}

        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if verbose:
                print(f"Training {model_name}")
            tmp_params = self._get_params(model_name, verbose=verbose)
            space = self._space(model_name=model_name)
            if self.trainer.bayes_opt and not warm_start and len(space) > 0:
                callback = BayesCallback(
                    tqdm(total=self.trainer.n_calls, disable=not verbose)
                )
                global _bayes_objective

                @skopt.utils.use_named_args(space)
                def _bayes_objective(**params):
                    with HiddenPrints(disable_logging=True):
                        model = self.new_model(
                            model_name=model_name, verbose=False, **params
                        )

                        self._train_single_model(
                            model,
                            epoch=self.trainer.args["bayes_epoch"],
                            X_train=X_train,
                            D_train=D_train,
                            y_train=y_train,
                            X_val=X_val,
                            D_val=D_val,
                            y_val=y_val,
                            verbose=False,
                            warm_start=False,
                            in_bayes_opt=True,
                            **params,
                        )

                    pred = self._pred_single_model(model, X_val, D_val, verbose=False)
                    res = metric_sklearn(pred, y_val, self.trainer.loss)
                    return res

                with warnings.catch_warnings():
                    # To obtain clean progress bar.
                    warnings.filterwarnings(
                        "ignore",
                        message="The objective has been evaluated at this point before",
                    )
                    result = gp_minimize(
                        _bayes_objective,
                        self._space(model_name=model_name),
                        n_calls=self.trainer.n_calls,
                        callback=callback.call,
                        random_state=0,
                        x0=list(tmp_params.values()),
                    )
                params = {}
                for key, value in zip(tmp_params.keys(), result.x):
                    params[key] = value
                self.model_params[model_name] = cp(params)
                callback.close()
                skopt.dump(result, self.trainer.project_root + "skopt.pt")
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
                X_train=X_train,
                D_train=D_train,
                y_train=y_train,
                X_val=X_val,
                D_val=D_val,
                y_val=y_val,
                verbose=verbose,
                warm_start=warm_start,
                in_bayes_opt=False,
                **tmp_params,
            )

            def pred_set(X, D, y, name):
                pred = self._pred_single_model(model, X, D, verbose=False)
                mse = metric_sklearn(pred, y, "mse")
                if verbose:
                    print(f"{name} MSE loss: {mse:.5f}, RMSE loss: {np.sqrt(mse):.5f}")

            pred_set(X_train, D_train, y_train, "Training")
            pred_set(X_val, D_val, y_val, "Validation")
            pred_set(X_test, D_test, y_test, "Testing")
            self.model[model_name] = model

        # enable_tqdm()
        if dump_trainer:
            save_trainer(self.trainer)

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
        for model_name in self._get_model_names():
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
        self.root = self.trainer.project_root + self.program + "/"
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
            return self.model_subset
        else:
            return self._get_model_names()

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
            A new model (without any restriction). It will be passed to :func:`_train_single_model` and
            :func:`_pred_single_model`.
        """
        raise NotImplementedError

    def _train_data_preprocess(
        self,
        X_train: pd.DataFrame,
        D_train: Dict[str, np.ndarray],
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        D_val: Dict[str, np.ndarray],
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        D_test: Dict[str, np.ndarray],
        y_test: np.ndarray,
    ):
        """
        Processing the dataset returned by :func:`_base_train_data_preprocess`. Returned values will be passed to
        :func:`_train_single_model` and :func:`_pred_single_model`. This function can be used to train e.g. imputers
        and scalers.

        Parameters
        ----------
        The training datasets: X_train, D_train, y_train;
        The validation datasets: X_val, D_val, y_val;
        The testing datasets: X_test, D_test, y_test

        Returns
        -------
        The training datasets: X_train, D_train, y_train;
        The validation datasets: X_val, D_val, y_val;
        The testing datasets: X_test, D_test, y_test
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
            The new tabular dataset.
        derived_data:
            Data derived from :func:`Trainer.derive_unstacked`.
        model_name:
            The name of a selected model.

        Returns
        -------
        data:
            The processed tabular dataset and derived data.
        """
        raise NotImplementedError

    def _train_single_model(
        self,
        model: Any,
        epoch: Optional[int],
        X_train: Any,
        D_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        D_val: Any,
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
        D_train:
            The training derived data from :func:`_train_data_preprocess`.
        y_train:
            The training target from :func:`_train_data_preprocess`.
        X_val:
            The validation data from :func:`_train_data_preprocess`.
        D_val:
            The validation derived data from :func:`_train_data_preprocess`.
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
        self, model: Any, X_test: Any, D_test: Any, verbose: bool, **kwargs
    ) -> np.ndarray:
        """
        Predict with the model trained in :func:`_train_single_model`.

        Parameters
        ----------
        model:
            The model trained in :func:`_train_single_model`.
        X_test:
            The testing data from :func:`_train_data_preprocess`.
        D_test:
            The testing derived data from :func:`_train_data_preprocess`.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        pred:
            Prediction of the target.
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


class BayesCallback:
    """
    Show a tqdm progress bar when performing bayes optimization.
    """

    def __init__(self, bar):
        self.bar = bar
        self.postfix = {
            "Current loss": 1e8,
            "Current Params": [],
            "Minimum": 1e8,
            "Best Params": [],
            "Minimum at call": 0,
        }
        self.bar.set_postfix(**self.postfix)

    def call(self, result):
        self.postfix["Current loss"] = result.func_vals[-1]
        self.postfix["Current Params"] = [round(x, 8) for x in result.x_iters[-1]]
        if result.fun < self.postfix["Minimum"]:
            self.postfix["Minimum"] = result.fun
            self.postfix["Best Params"] = [round(x, 8) for x in result.x]
            self.postfix["Minimum at call"] = len(result.func_vals)

        self.bar.set_postfix(**self.postfix)
        self.bar.update(1)

    def close(self):
        self.bar.close()
        del self.bar


class TorchModel(AbstractModel):
    """
    The specific class for PyTorch-like models. Some abstract methods in AbstractModel are implemented.
    """

    def _train_step(
        self,
        model: nn.Module,
        train_loader: Data.DataLoader,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> float:
        """
        Train a torch.nn.Module model in a single epoch.

        Parameters
        ----------
        model:
            The torch model initialized in :func:`_new_model`.
        train_loader:
            The DataLoader of the training dataset.
        optimizer:
            A torch optimizer.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        loss:
            The loss of the model on the training dataset.
        """
        model.train()
        avg_loss = 0
        for idx, tensors in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(*([data] + additional_tensors))
            loss = model.loss_fn(
                yhat, y, model, *([data] + additional_tensors), **kwargs
            )
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * len(y)

        avg_loss /= len(train_loader.dataset)
        return avg_loss

    def _test_step(
        self, model: nn.Module, test_loader: Data.DataLoader, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate a torch.nn.Module model in a single epoch.

        Parameters
        ----------
        model:
            The torch model initialized in :func:`_new_model`.
        test_loader:
            The DataLoader of the testing dataset.
        **kwargs:
            Parameters to train the model. It contains all arguments in :func:`_initial_values`.

        Returns
        -------
        results:
            The prediction, ground truth, and loss of the model on the testing dataset.
        """
        model.eval()
        pred = []
        truth = []
        with torch.no_grad() if src.setting["test_with_no_grad"] else nullcontext():
            # print(test_dataset)
            avg_loss = 0
            for idx, tensors in enumerate(test_loader):
                yhat = tensors[-1]
                data = tensors[0]
                additional_tensors = tensors[1 : len(tensors) - 1]
                y = model(*([data] + additional_tensors))
                loss = model.loss_fn(
                    yhat, y, model, *([data] + additional_tensors), **kwargs
                )
                avg_loss += loss.item() * len(y)
                pred += list(y.cpu().detach().numpy())
                truth += list(yhat.cpu().detach().numpy())
            avg_loss /= len(test_loader.dataset)
        return np.array(pred), np.array(truth), avg_loss

    def _train_data_preprocess(
        self,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        X_test,
        D_test,
        y_test,
    ):
        from torch.utils.data.dataloader import default_collate

        collate_fn = lambda x: list(x_.to(self.device) for x_ in default_collate(x))
        train_loader = Data.DataLoader(
            self.trainer.train_dataset,
            batch_size=len(self.trainer.train_dataset),
            generator=torch.Generator().manual_seed(0),
            collate_fn=collate_fn,
        )
        val_loader = Data.DataLoader(
            self.trainer.val_dataset,
            batch_size=len(self.trainer.val_dataset),
            generator=torch.Generator().manual_seed(0),
            collate_fn=collate_fn,
        )
        test_loader = Data.DataLoader(
            self.trainer.test_dataset,
            batch_size=len(self.trainer.test_dataset),
            generator=torch.Generator().manual_seed(0),
            collate_fn=collate_fn,
        )
        return (
            train_loader,
            None,
            y_train,
            val_loader,
            None,
            y_val,
            test_loader,
            None,
            y_test,
        )

    def _data_preprocess(self, df, derived_data, model_name):
        df = self.trainer.data_transform(df)
        X = torch.tensor(
            df[self.trainer.cont_feature_names].values.astype(np.float32),
            dtype=torch.float32,
        ).to(self.device)
        if src.setting["input_require_grad"]:
            X.require_grad = True
        D = [
            torch.tensor(value, dtype=torch.float32).to(self.device)
            for value in derived_data.values()
        ]
        y = torch.tensor(np.zeros((len(df), 1)), dtype=torch.float32).to(self.device)

        loader = Data.DataLoader(
            Data.TensorDataset(X, *D, y), batch_size=len(df), shuffle=False
        )
        return loader, derived_data

    def _train_single_model(
        self,
        model,
        epoch,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        from torch.utils.data.dataloader import default_collate

        collate_fn = lambda x: list(x_.to(self.device) for x_ in default_collate(x))
        model.to(self.device)
        optimizer = model.get_optimizer(warm_start, **kwargs)

        train_loader = Data.DataLoader(
            X_train.dataset,
            batch_size=int(kwargs["batch_size"]),
            generator=torch.Generator().manual_seed(0),
            collate_fn=collate_fn,
        )
        val_loader = X_val

        train_ls = []
        val_ls = []
        stop_epoch = self.trainer.args["epoch"]

        early_stopping = EarlyStopping(
            patience=self.trainer.static_params["patience"],
            verbose=False,
            path=self.trainer.project_root + "fatigue.pt",
        )

        for i_epoch in range(epoch):
            t_start = time.time()
            train_loss = self._train_step(model, train_loader, optimizer, **kwargs)
            train_ls.append(train_loss)
            _, _, val_loss = self._test_step(model, val_loader, **kwargs)
            val_ls.append(val_loss)
            t_end = time.time()

            if verbose and (
                (i_epoch + 1) % src.setting["verbose_per_epoch"] == 0 or i_epoch == 0
            ):
                print(
                    f"Epoch: {i_epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Min "
                    f"val loss: {np.min(val_ls):.4f}, Epoch time: {t_end-t_start:.4f}s."
                )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                if verbose:
                    idx = val_ls.index(min(val_ls))
                    print(
                        f"Early stopping at epoch {i_epoch + 1}, Checkpoint at epoch {idx + 1}, Train loss: {train_ls[idx]:.4f}, Val loss: {val_ls[idx]:.4f}"
                    )
                break

        idx = val_ls.index(min(val_ls))
        min_loss = val_ls[idx]
        model.to("cpu")
        model.load_state_dict(torch.load(self.trainer.project_root + "fatigue.pt"))

        if verbose:
            print(f"Minimum loss: {min_loss:.5f}")

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        model.to(self.device)
        y_test_pred, _, _ = self._test_step(model, X_test, **kwargs)
        model.to("cpu")
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


class AbstractNN(nn.Module):
    def __init__(self, trainer: Trainer):
        """
        PyTorch model that contains derived data names and dimensions from the trainer.

        Parameters
        ----------
        trainer:
            A Trainer instance.
        """
        super(AbstractNN, self).__init__()
        self.default_loss_fn = trainer.loss_fn
        self.derived_feature_names = list(trainer.derived_data.keys())
        self.derived_feature_dims = trainer.get_derived_data_sizes()
        self.derived_feature_names_dims = {}
        for name, dim in zip(
            trainer.derived_data.keys(), trainer.get_derived_data_sizes()
        ):
            self.derived_feature_names_dims[name] = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        A wrapper of the original forward of nn.Module. Input data are tensors with no names, but their names are
        obtained during initialization, so that a dict of derived data with names is generated and passed to
        :func:`_forward`.

        Parameters
        ----------
        tensors:
            Input tensors to the torch model.

        Returns
        -------
        result:
            The obtained tensor.
        """
        x = tensors[0]
        additional_tensors = tensors[1:]
        derived_tensors = {}
        for tensor, name in zip(additional_tensors, self.derived_feature_names):
            derived_tensors[name] = tensor
        return self._forward(x, derived_tensors)

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        """
        User defined loss function.

        Parameters
        ----------
        y_true:
            Ground truth value.
        y_pred:
            Predicted value by the model.
        model:
            The model predicting y_pred.
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

    def get_optimizer(self, warm_start, **kwargs):
        return torch.optim.Adam(
            self.parameters(),
            lr=kwargs["lr"] / 10 if warm_start else kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )


class ModelDict:
    def __init__(self, path):
        self.root = path
        self.model_path = {}

    def __setitem__(self, key, value):
        self.model_path[key] = os.path.join(self.root, key) + ".pkl"
        with open(self.model_path[key], "wb") as file:
            pickle.dump((key, value), file, pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        with open(self.model_path[item], "rb") as file:
            key, model = pickle.load(file)
        return model

    def keys(self):
        return self.model_path.keys()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def get_sequential(
    layers, n_inputs, n_outputs, act_func, dropout=0, use_norm=True, norm_type="batch"
):
    net = nn.Sequential()
    if norm_type == "batch":
        norm = nn.BatchNorm1d
    elif norm_type == "layer":
        norm = nn.LayerNorm
    else:
        raise Exception(f"Normalization {norm_type} not implemented.")
    if len(layers) > 0:
        net.add_module("input", nn.Linear(n_inputs, layers[0]))
        net.add_module("activate_0", act_func())
        if use_norm:
            net.add_module(f"norm_0", norm(layers[0]))
        if dropout != 0:
            net.add_module(f"dropout_0", nn.Dropout(dropout))
        for idx in range(1, len(layers)):
            net.add_module(str(idx), nn.Linear(layers[idx - 1], layers[idx]))
            net.add_module(f"activate_{idx}", act_func())
            if use_norm:
                net.add_module(f"norm_{idx}", norm(layers[idx]))
            if dropout != 0:
                net.add_module(f"dropout_{idx}", nn.Dropout(dropout))
        net.add_module("output", nn.Linear(layers[-1], n_outputs))
    else:
        net.add_module("single_layer", nn.Linear(n_inputs, n_outputs))
        net.add_module("activate", act_func())
        if use_norm:
            net.add_module("norm", norm(n_outputs))
        if dropout != 0:
            net.add_module("dropout", nn.Dropout(dropout))

    net.apply(init_weights)
    return net
