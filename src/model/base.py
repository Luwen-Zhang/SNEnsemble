from src.utils import *
from src.trainer import Trainer, save_trainer
import skopt
from skopt import gp_minimize
import torch.utils.data as Data
import torch.nn as nn
from copy import deepcopy as cp


class AbstractModel:
    def __init__(self, trainer: Trainer = None, program=None, model_subset=None):
        self.trainer = trainer
        if not hasattr(trainer, "database"):
            trainer.load_config(default_configfile="base_config")
        self.model = None
        self.leaderboard = None
        self.model_subset = model_subset
        self.program = self._get_program_name() if program is None else program
        self.model_params = {}
        self._mkdir()

    def fit(
        self,
        df,
        cont_feature_names: list,
        cat_feature_names: list,
        label_name: list,
        model_subset: list = None,
        derived_data: dict = None,
        verbose=True,
        warm_start=False,
        bayes_opt=False,
    ):
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
        # Training the model using data in the trainer directly.
        # The method can be rewritten to implement other training strategies.
        verbose = "verbose" not in kwargs.keys() or kwargs["verbose"]
        if verbose:
            print(f"\n-------------Run {self.program}-------------\n")
        self._train(*args, **kwargs)
        if verbose:
            print(f"\n-------------{self.program} End-------------\n")

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
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
            if len(absent_keys) > 0:
                raise Exception(
                    f"Additional feature {absent_keys} not in the input derived_data."
                )
        df = self.trainer.dataimputer.transform(df.copy(), self.trainer)
        return self._predict(
            df, model_name, self.trainer.sort_derived_data(derived_data), **kwargs
        )

    def _base_train_data_preprocess(self):
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

    def _predict_all(self, verbose=True, test_data_only=False):
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

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
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
        model_subset=None,
        dump_trainer=True,
        verbose=True,
        warm_start=False,
        **kwargs,
    ):
        # disable_tqdm()
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
        self.model = {}

        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if verbose:
                print(f"Training {model_name}")
            tmp_params = self._get_params(model_name, verbose=verbose)

            if self.trainer.bayes_opt and not warm_start:
                callback = BayesCallback(
                    tqdm(total=self.trainer.n_calls, disable=not verbose)
                )
                global _bayes_objective

                @skopt.utils.use_named_args(self._space(model_name=model_name))
                def _bayes_objective(**params):
                    with HiddenPrints(disable_logging=True):
                        model = self._new_model(
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
                            **params,
                        )

                    pred = self._pred_single_model(model, X_val, D_val, verbose=False)
                    res = Trainer._metric_sklearn(pred, y_val, self.trainer.loss)
                    return res

                with warnings.catch_warnings():
                    # To obtain clean progress bar.
                    warnings.simplefilter("ignore")
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
                self.model[model_name] = self._new_model(
                    model_name=model_name, verbose=verbose, **tmp_params
                )

            self._train_single_model(
                self.model[model_name],
                epoch=self.total_epoch,
                X_train=X_train,
                D_train=D_train,
                y_train=y_train,
                X_val=X_val,
                D_val=D_val,
                y_val=y_val,
                verbose=verbose,
                warm_start=warm_start,
                **tmp_params,
            )

            test_pred = self._pred_single_model(
                self.model[model_name], X_test, D_test, verbose=False
            )
            test_res = Trainer._metric_sklearn(test_pred, y_test, self.trainer.loss)

            if verbose:
                if self.trainer.loss == "mse":
                    print(
                        f"Test MSE loss: {test_res:.5f}, RMSE loss: {np.sqrt(test_res):.5f}"
                    )
                else:
                    print(f"Test {self.trainer.loss} loss: {test_res:.5f}.")

        # enable_tqdm()
        if dump_trainer:
            save_trainer(self.trainer)

    def _check_train_status(self):
        if not self._trained:
            raise Exception(
                f"{self.program} not trained, run {self.__class__.__name__}.train() first."
            )

    def _get_params(self, model_name, verbose=True):
        if model_name not in self.model_params.keys():
            return self._initial_values(model_name=model_name)
        else:
            if verbose:
                print(f"Previous params loaded: {self.model_params[model_name]}")
            return self.model_params[model_name]

    @property
    def _trained(self):
        if self.model is None:
            return False
        else:
            return True

    def _mkdir(self):
        self.root = self.trainer.project_root + self.program + "/"
        if not os.path.exists(self.root):
            os.mkdir(self.root)

    def get_model_names(self):
        if self.model_subset is not None:
            for model in self.model_subset:
                if model not in self._get_model_names():
                    raise Exception(f"Model {model} not available for {self.program}.")
            return self.model_subset
        else:
            return self._get_model_names()

    def _get_model_names(self):
        raise NotImplementedError

    def _get_program_name(self):
        raise NotImplementedError

    # Following methods are for the default _train and _predict methods. If users directly overload _train and _predict,
    # following methods are not required to be implemented.
    def _new_model(self, model_name, verbose, **kwargs):
        raise NotImplementedError

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
        raise NotImplementedError

    def _data_preprocess(self, df, derived_data, model_name):
        raise NotImplementedError

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
        **kwargs,
    ):
        raise NotImplementedError

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        raise NotImplementedError

    def _space(self, model_name):
        raise NotImplementedError

    def _initial_values(self, model_name):
        raise NotImplementedError


class BayesCallback:
    def __init__(self, bar):
        self.bar = bar
        self.postfix = {
            "Current loss": 1e8,
            "Minimum": 1e8,
            "Params": [],
            "Minimum at call": 0,
        }
        self.bar.set_postfix(**self.postfix)

    def call(self, result):
        self.postfix["Current loss"] = result.func_vals[-1]

        if result.fun < self.postfix["Minimum"]:
            self.postfix["Minimum"] = result.fun
            self.postfix["Params"] = [round(x, 8) for x in result.x]
            self.postfix["Minimum at call"] = len(result.func_vals)

        self.bar.set_postfix(**self.postfix)
        self.bar.update(1)

    def close(self):
        self.bar.close()
        del self.bar


class TorchModel(AbstractModel):
    def __init__(self, trainer=None, program=None, model_subset=None):
        super(TorchModel, self).__init__(
            trainer, program=program, model_subset=model_subset
        )

    def _train_step(self, model, train_loader, optimizer, loss_fn):
        model.train()
        avg_loss = 0
        for idx, tensors in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(*([data] + additional_tensors))
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * len(y)

        avg_loss /= len(train_loader.dataset)
        return avg_loss

    def _test_step(self, model, test_loader, loss_fn):
        model.eval()
        pred = []
        truth = []
        with torch.no_grad():
            # print(test_dataset)
            avg_loss = 0
            for idx, tensors in enumerate(test_loader):
                yhat = tensors[-1]
                data = tensors[0]
                additional_tensors = tensors[1 : len(tensors) - 1]
                y = model(*([data] + additional_tensors))
                loss = loss_fn(yhat, y)
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
        train_loader = Data.DataLoader(
            self.trainer.train_dataset,
            batch_size=len(self.trainer.train_dataset),
            generator=torch.Generator().manual_seed(0),
        )
        val_loader = Data.DataLoader(
            self.trainer.val_dataset,
            batch_size=len(self.trainer.val_dataset),
            generator=torch.Generator().manual_seed(0),
        )
        test_loader = Data.DataLoader(
            self.trainer.test_dataset,
            batch_size=len(self.trainer.test_dataset),
            generator=torch.Generator().manual_seed(0),
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
        ).to(self.trainer.device)
        D = [
            torch.tensor(value, dtype=torch.float32).to(self.trainer.device)
            for value in derived_data.values()
        ]
        y = torch.tensor(np.zeros((len(df), 1)), dtype=torch.float32).to(
            self.trainer.device
        )

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
        **kwargs,
    ):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=kwargs["lr"] / 10 if warm_start else kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )

        train_loader = Data.DataLoader(
            X_train.dataset,
            batch_size=int(kwargs["batch_size"]),
            generator=torch.Generator().manual_seed(0),
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
            train_loss = self._train_step(
                model, train_loader, optimizer, self.trainer.loss_fn
            )
            train_ls.append(train_loss)
            _, _, val_loss = self._test_step(model, val_loader, self.trainer.loss_fn)
            val_ls.append(val_loss)

            if verbose and ((i_epoch + 1) % 20 == 0 or i_epoch == 0):
                print(
                    f"Epoch: {i_epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Min val loss: {np.min(val_ls):.4f}"
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

        model.load_state_dict(torch.load(self.trainer.project_root + "fatigue.pt"))

        if verbose:
            print(f"Minimum loss: {min_loss:.5f}")

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        y_test_pred, _, _ = self._test_step(model, X_test, self.trainer.loss_fn)
        return y_test_pred

    def _space(self, model_name):
        return self.trainer.SPACE

    def _initial_values(self, model_name):
        return self.trainer.chosen_params


class AbstractNN(nn.Module):
    def __init__(self, trainer):
        super(AbstractNN, self).__init__()
        self.derived_feature_names = list(trainer.derived_data.keys())
        self.derived_feature_dims = trainer.get_derived_data_sizes()

    def forward(self, *tensors):
        x = tensors[0]
        additional_tensors = tensors[1:]
        derived_tensors = {}
        for tensor, name in zip(additional_tensors, self.derived_feature_names):
            derived_tensors[name] = tensor
        return self._forward(x, derived_tensors)

    def _forward(self, x, derived_tensors):
        raise NotImplementedError
