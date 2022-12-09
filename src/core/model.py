from .trainer import *
import skopt
from skopt import gp_minimize


class AbstractModel:
    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer
        if not hasattr(trainer, "project"):
            trainer.load_config(default_configfile="base_config")
        self.model = None
        self.leaderboard = None
        self.program = None

    def fit(
        self,
        df,
        feature_names: list,
        label_name: list,
        derived_data: dict = None,
        verbose=True,
        warm_start=False,
    ):
        self.trainer.df = df
        self.trainer.feature_names = feature_names
        self.trainer.label_name = label_name
        self.trainer.derived_data = derived_data
        indices = np.arange(len(df))
        self.trainer._data_process(
            preprocess=True,
            train_indices=indices,
            val_indices=indices,
            test_indices=indices,
            warm_start=warm_start if self._trained else False,
        )
        self.trainer._update_dataset_auto()
        self._train(
            dump_trainer=False,
            verbose=verbose,
            warm_start=warm_start if self._trained else False,
        )

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        if self.model is None:
            raise Exception("Run fit() before predict().")
        if model_name not in self._get_model_names():
            raise Exception(
                f"Model {model_name} is not available. Select among {self._get_model_names()}"
            )
        absent_features = []
        for feature_name in self.trainer.feature_names:
            if feature_name not in df.columns:
                absent_features.append(feature_name)
        if len(absent_features) > 0:
            raise Exception(f"Feature {absent_features} not in the input dataframe.")

        additional_data = []
        absent_keys = []
        for key in self.trainer.derived_data.keys():
            if key not in derived_data.keys():
                absent_keys.append(key)
            else:
                additional_data.append(derived_data[key])
        if len(absent_keys) > 0:
            raise Exception(
                f"Additional feature {absent_keys} not in the input derived_data."
            )
        return self._predict(df, model_name, additional_data, **kwargs)

    def _predict_all(self, verbose=True, test_data_only=False):
        self._check_train_status()

        model_names = self._get_model_names()
        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
        train_data = tabular_dataset.loc[self.trainer.train_dataset.indices, :].copy()
        val_data = tabular_dataset.loc[self.trainer.val_dataset.indices, :].copy()
        test_data = tabular_dataset.loc[self.trainer.test_dataset.indices, :].copy()

        predictions = {}
        disable_tqdm()
        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f"{idx + 1}/{len(model_names)}")
            if not test_data_only:
                y_train_pred = self._predict(
                    train_data, model_name=model_name, as_pandas=False
                )
                y_train = train_data[self.trainer.label_name[0]].values

                y_val_pred = self._predict(
                    val_data, model_name=model_name, as_pandas=False
                )
                y_val = val_data[self.trainer.label_name[0]].values
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = self._predict(
                test_data, model_name=model_name, as_pandas=False
            )
            y_test = test_data[self.trainer.label_name[0]].values

            predictions[model_name] = {
                "Training": (y_train_pred, y_train),
                "Testing": (y_test_pred, y_test),
                "Validation": (y_val_pred, y_val),
            }

        enable_tqdm()
        return predictions

    def _predict(self, df: pd.DataFrame, model_name, additional_data=None, **kwargs):
        raise NotImplementedError

    def _train(self, dump_trainer=True, verbose=True, warm_start=False, **kwargs):
        raise NotImplementedError

    def _get_model_names(self):
        raise NotImplementedError

    def _check_train_status(self):
        if not self._trained:
            raise Exception(
                f"{self.program} not trained, run {self.__class__.__name__}._train() first."
            )

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


class AutoGluon(AbstractModel):
    def __init__(self, trainer=None):
        super(AutoGluon, self).__init__(trainer)
        self.program = "AutoGluon"
        self._mkdir()

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        print("\n-------------Run AutoGluon Tests-------------\n")
        disable_tqdm()
        import warnings

        warnings.simplefilter(action="ignore", category=UserWarning)
        from autogluon.tabular import TabularPredictor

        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
        predictor = TabularPredictor(label=label_name[0], path=self.root)
        with HiddenPrints(disable_logging=True if not verbose else False):
            predictor.fit(
                tabular_dataset.loc[self.trainer.train_dataset.indices, :],
                tuning_data=tabular_dataset.loc[self.trainer.val_dataset.indices, :],
                presets="best_quality"
                if not debug_mode
                else "medium_quality_faster_train",
                hyperparameter_tune_kwargs="bayesopt"
                if (not debug_mode) and self.trainer.bayes_opt
                else None,
                use_bag_holdout=True,
                verbosity=0 if not verbose else 2,
            )
        self.leaderboard = predictor.leaderboard(
            tabular_dataset.loc[self.trainer.test_dataset.indices, :], silent=True
        )
        self.leaderboard.to_csv(self.root + "leaderboard.csv")
        self.model = predictor
        enable_tqdm()
        warnings.simplefilter(action="default", category=UserWarning)
        if dump_trainer:
            save_trainer(self.trainer)
        print("\n-------------AutoGluon Tests End-------------\n")

    def _predict(self, df: pd.DataFrame, model_name, additional_data=None, **kwargs):
        return self.model.predict(
            df[self.trainer.feature_names], model=model_name, **kwargs
        )

    def _get_model_names(self):
        return list(self.leaderboard["model"])


class PytorchTabular(AbstractModel):
    def __init__(self, trainer=None):
        super(PytorchTabular, self).__init__(trainer)
        self.program = "PytorchTabular"
        self._mkdir()

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        print("\n-------------Run Pytorch-tabular Tests-------------\n")
        disable_tqdm()
        import warnings

        warnings.simplefilter(action="ignore", category=UserWarning)
        from functools import partialmethod
        from pytorch_tabular.config import ExperimentRunManager

        ExperimentRunManager.__init__ = partialmethod(
            ExperimentRunManager.__init__, self.root + "exp_version_manager.yml"
        )
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import (
            CategoryEmbeddingModelConfig,
            NodeConfig,
            TabNetModelConfig,
            TabTransformerConfig,
            AutoIntConfig,
            FTTransformerConfig,
        )
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()

        data_config = DataConfig(
            target=label_name, continuous_cols=feature_names, num_workers=8
        )

        trainer_config = TrainerConfig(
            max_epochs=self.trainer.static_params["epoch"] if not debug_mode else 10,
            early_stopping_patience=self.trainer.static_params["patience"],
        )
        optimizer_config = OptimizerConfig()

        model_configs = [
            CategoryEmbeddingModelConfig(task="regression"),
            # NodeConfig(task='regression'),
            TabNetModelConfig(task="regression"),
            TabTransformerConfig(task="regression"),
            AutoIntConfig(task="regression"),
            # FTTransformerConfig(task='regression')
        ]

        # dtype=int because pytorch-tabular can not convert np.int64 to Integer
        # All commented lines in SPACEs cause error. Do not uncomment them.
        SPACEs = {
            "CategoryEmbeddingModel": {
                "SPACE": [
                    Real(low=0, high=0.8, prior="uniform", name="dropout"),  # 0.5
                    Real(
                        low=0, high=0.8, prior="uniform", name="embedding_dropout"
                    ),  # 0.5
                    Real(
                        low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                    ),  # 0.001
                ],
                "defaults": [0.5, 0.5, 0.001],
                "class": CategoryEmbeddingModelConfig,
            },
            "NODEModel": {
                "SPACE": [
                    Integer(
                        low=2, high=6, prior="uniform", name="depth", dtype=int
                    ),  # 6
                    Real(
                        low=0, high=0.8, prior="uniform", name="embedding_dropout"
                    ),  # 0.0
                    Real(low=0, high=0.8, prior="uniform", name="input_dropout"),  # 0.0
                    Real(
                        low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                    ),  # 0.001
                    Integer(
                        low=128, high=512, prior="uniform", name="num_trees", dtype=int
                    ),
                ],
                "defaults": [6, 0.0, 0.0, 0.001, 256],
                "class": NodeConfig,
            },
            "TabNetModel": {
                "SPACE": [
                    Integer(
                        low=4, high=64, prior="uniform", name="n_d", dtype=int
                    ),  # 8
                    Integer(
                        low=4, high=64, prior="uniform", name="n_a", dtype=int
                    ),  # 8
                    Integer(
                        low=3, high=10, prior="uniform", name="n_steps", dtype=int
                    ),  # 3
                    Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
                    Integer(
                        low=1, high=5, prior="uniform", name="n_independent", dtype=int
                    ),  # 2
                    Integer(
                        low=1, high=5, prior="uniform", name="n_shared", dtype=int
                    ),  # 2
                    Real(
                        low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                    ),  # 0.001
                ],
                "defaults": [8, 8, 3, 1.3, 2, 2, 0.001],
                "class": TabNetModelConfig,
            },
            "TabTransformerModel": {
                "SPACE": [
                    Real(
                        low=0, high=0.8, prior="uniform", name="embedding_dropout"
                    ),  # 0.1
                    Real(low=0, high=0.8, prior="uniform", name="ff_dropout"),  # 0.1
                    # Categorical([16, 32, 64, 128], name='input_embed_dim'),  # 32
                    Real(
                        low=0,
                        high=0.5,
                        prior="uniform",
                        name="shared_embedding_fraction",
                    ),  # 0.25
                    # Categorical([2, 4, 8, 16], name='num_heads'),  # 8
                    Integer(
                        low=4,
                        high=8,
                        prior="uniform",
                        name="num_attn_blocks",
                        dtype=int,
                    ),  # 6
                    Real(low=0, high=0.8, prior="uniform", name="attn_dropout"),  # 0.1
                    Real(
                        low=0, high=0.8, prior="uniform", name="add_norm_dropout"
                    ),  # 0.1
                    Integer(
                        low=2,
                        high=6,
                        prior="uniform",
                        name="ff_hidden_multiplier",
                        dtype=int,
                    ),  # 4
                    Real(
                        low=0, high=0.8, prior="uniform", name="out_ff_dropout"
                    ),  # 0.0
                    Real(
                        low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                    ),  # 0.001
                ],
                "defaults": [0.1, 0.1, 0.25, 6, 0.1, 0.1, 4, 0.0, 0.001],
                "class": TabTransformerConfig,
            },
            "AutoIntModel": {
                "SPACE": [
                    Real(
                        low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                    ),  # 0.001
                    Real(low=0, high=0.8, prior="uniform", name="attn_dropouts"),  # 0.0
                    # Categorical([16, 32, 64, 128], name='attn_embed_dim'),  # 32
                    Real(low=0, high=0.8, prior="uniform", name="dropout"),  # 0.0
                    # Categorical([8, 16, 32, 64], name='embedding_dim'),  # 16
                    Real(
                        low=0, high=0.8, prior="uniform", name="embedding_dropout"
                    ),  # 0.0
                    Integer(
                        low=1,
                        high=4,
                        prior="uniform",
                        name="num_attn_blocks",
                        dtype=int,
                    ),  # 3
                    # Integer(low=1, high=4, prior='uniform', name='num_heads', dtype=int),  # 2
                ],
                "defaults": [0.001, 0.0, 0.0, 0.0, 3],
                "class": AutoIntConfig,
            },
            "FTTransformerModel": {
                "SPACE": [
                    Real(
                        low=0, high=0.8, prior="uniform", name="embedding_dropout"
                    ),  # 0.1
                    Real(
                        low=0,
                        high=0.5,
                        prior="uniform",
                        name="shared_embedding_fraction",
                    ),  # 0.25
                    Integer(
                        low=4,
                        high=8,
                        prior="uniform",
                        name="num_attn_blocks",
                        dtype=int,
                    ),  # 6
                    Real(low=0, high=0.8, prior="uniform", name="attn_dropout"),  # 0.1
                    Real(
                        low=0, high=0.8, prior="uniform", name="add_norm_dropout"
                    ),  # 0.1
                    Real(low=0, high=0.8, prior="uniform", name="ff_dropout"),  # 0.1
                    Integer(
                        low=2,
                        high=6,
                        prior="uniform",
                        name="ff_hidden_multiplier",
                        dtype=int,
                    ),  # 4
                    Real(
                        low=0, high=0.8, prior="uniform", name="out_ff_dropout"
                    ),  # 0.0
                ],
                "defaults": [0.1, 0.25, 6, 0.1, 0.1, 0.1, 4, 0.0],
                "class": FTTransformerConfig,
            },
        }

        metrics = {"model": [], "mse": []}
        models = {}
        for model_config in model_configs:
            model_name = model_config._model_name
            print("Training", model_name)

            SPACE = SPACEs[model_name]["SPACE"]
            defaults = SPACEs[model_name]["defaults"]
            config_class = SPACEs[model_name]["class"]
            exceptions = []
            global _pytorch_tabular_bayes_objective

            @skopt.utils.use_named_args(SPACE)
            def _pytorch_tabular_bayes_objective(**params):
                if verbose:
                    print(params, end=" ")
                with HiddenPrints(disable_logging=True):
                    tabular_model = TabularModel(
                        data_config=data_config,
                        model_config=config_class(task="regression", **params),
                        optimizer_config=optimizer_config,
                        trainer_config=TrainerConfig(
                            max_epochs=self.trainer.bayes_epoch,
                            early_stopping_patience=self.trainer.bayes_epoch,
                        ),
                    )
                    tabular_model.config.checkpoints = None
                    tabular_model.config["progress_bar_refresh_rate"] = 0
                    try:
                        tabular_model.fit(
                            train=tabular_dataset.loc[
                                self.trainer.train_dataset.indices, :
                            ],
                            validation=tabular_dataset.loc[
                                self.trainer.val_dataset.indices, :
                            ],
                            loss=self.trainer.loss_fn,
                        )
                    except Exception as e:
                        exceptions.append(e)
                        return 1e3
                    res = tabular_model.evaluate(
                        tabular_dataset.loc[self.trainer.val_dataset.indices, :]
                    )[0]["test_mean_squared_error"]
                if verbose:
                    print(res)
                return res

            if not debugger_is_active() and self.trainer.bayes_opt:
                # otherwise: AssertionError: can only test a child process
                result = gp_minimize(
                    _pytorch_tabular_bayes_objective,
                    SPACE,
                    x0=defaults,
                    n_calls=self.trainer.n_calls if not debug_mode else 11,
                    random_state=0,
                )
                param_names = [x.name for x in SPACE]
                params = {}
                for key, value in zip(param_names, result.x):
                    params[key] = value
                model_config = config_class(task="regression", **params)

                if len(exceptions) > 0 and verbose:
                    print("Exceptions in bayes optimization:", exceptions)

            with HiddenPrints(disable_logging=True if not verbose else False):
                tabular_model = TabularModel(
                    data_config=data_config,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config,
                )
                tabular_model.config.checkpoints_path = self.root
                tabular_model.config["progress_bar_refresh_rate"] = 0

                tabular_model.fit(
                    train=tabular_dataset.loc[self.trainer.train_dataset.indices, :],
                    validation=tabular_dataset.loc[self.trainer.val_dataset.indices, :],
                    loss=self.trainer.loss_fn,
                )
                metrics["model"].append(tabular_model.config._model_name)
                metrics["mse"].append(
                    tabular_model.evaluate(
                        tabular_dataset.loc[self.trainer.test_dataset.indices, :]
                    )[0]["test_mean_squared_error"]
                )
                models[tabular_model.config._model_name] = tabular_model

        metrics = pd.DataFrame(metrics)
        metrics["rmse"] = metrics["mse"] ** (1 / 2)
        metrics.sort_values("rmse", inplace=True)
        self.leaderboard = metrics
        self.model = models

        if verbose:
            print(self.leaderboard)
        self.leaderboard.to_csv(self.root + "leaderboard.csv")

        enable_tqdm()
        warnings.simplefilter(action="default", category=UserWarning)
        if dump_trainer:
            save_trainer(self.trainer)
        print("\n-------------Pytorch-tabular Tests End-------------\n")

    def _predict(self, df: pd.DataFrame, model_name, additional_data=None, **kwargs):
        model = self.model[model_name]
        target = model.config.target[0]
        return np.array(model.predict(df)[f"{target}_prediction"])

    def _get_model_names(self):
        return list(self.leaderboard["model"])


class TabNet(AbstractModel):
    def __init__(self, trainer=None):
        super(TabNet, self).__init__(trainer)
        self.program = "TabNet"
        self._mkdir()

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        print("\n-------------Run TabNet Test-------------\n")
        train_indices = self.trainer.train_dataset.indices
        val_indices = self.trainer.val_dataset.indices
        test_indices = self.trainer.test_dataset.indices

        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
        feature_data = tabular_dataset[feature_names].copy()
        label_data = tabular_dataset[label_name].copy()

        train_x = feature_data.values[np.array(train_indices), :].astype(np.float32)
        test_x = feature_data.values[np.array(test_indices), :].astype(np.float32)
        val_x = feature_data.values[np.array(val_indices), :].astype(np.float32)
        train_y = (
            label_data.values[np.array(train_indices), :]
            .reshape(-1, 1)
            .astype(np.float32)
        )
        test_y = (
            label_data.values[np.array(test_indices), :]
            .reshape(-1, 1)
            .astype(np.float32)
        )
        val_y = (
            label_data.values[np.array(val_indices), :]
            .reshape(-1, 1)
            .astype(np.float32)
        )

        eval_set = [(val_x, val_y)]

        from pytorch_tabnet.tab_model import TabNetRegressor

        SPACE = [
            Integer(low=4, high=64, prior="uniform", name="n_d", dtype=int),  # 8
            Integer(low=4, high=64, prior="uniform", name="n_a", dtype=int),  # 8
            Integer(low=3, high=10, prior="uniform", name="n_steps", dtype=int),  # 3
            Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
            Integer(
                low=1, high=5, prior="uniform", name="n_independent", dtype=int
            ),  # 2
            Integer(low=1, high=5, prior="uniform", name="n_shared", dtype=int),  # 2
        ]

        defaults = [8, 8, 3, 1.3, 2, 2]

        global _tabnet_bayes_objective

        @skopt.utils.use_named_args(SPACE)
        def _tabnet_bayes_objective(**params):
            if verbose:
                print(params, end=" ")
            with HiddenPrints(disable_logging=True):
                model = TabNetRegressor(verbose=0)
                model.set_params(**params)
                model.fit(
                    train_x,
                    train_y,
                    eval_set=eval_set,
                    max_epochs=self.trainer.bayes_epoch,
                    patience=self.trainer.bayes_epoch,
                    loss_fn=self.trainer.loss_fn,
                    eval_metric=[self.trainer.loss],
                )

                res = self.trainer._metric_sklearn(
                    model.predict(val_x).reshape(-1, 1), val_y, self.trainer.loss
                )
            if verbose:
                print(res)
            return res

        if not debugger_is_active() and self.trainer.bayes_opt:
            # otherwise: AssertionError: can only test a child process
            result = gp_minimize(
                _tabnet_bayes_objective,
                SPACE,
                x0=defaults,
                n_calls=self.trainer.n_calls if not debug_mode else 11,
                random_state=0,
            )
            param_names = [x.name for x in SPACE]
            params = {}
            for key, value in zip(param_names, result.x):
                params[key] = value
        else:
            param_names = [x.name for x in SPACE]
            params = {}
            for key, value in zip(param_names, defaults):
                params[key] = value

        model = TabNetRegressor(verbose=100 if verbose else 0)
        model.set_params(**params)
        model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            max_epochs=self.trainer.static_params["epoch"],
            patience=self.trainer.static_params["patience"],
            loss_fn=self.trainer.loss_fn,
            eval_metric=[self.trainer.loss],
        )

        y_test_pred = model.predict(test_x).reshape(-1, 1)
        print(
            "MSE Loss:",
            self.trainer._metric_sklearn(y_test_pred, test_y, "mse"),
            "RMSE Loss:",
            self.trainer._metric_sklearn(y_test_pred, test_y, "rmse"),
        )
        self.model = model
        if dump_trainer:
            save_trainer(self.trainer)
        print("\n-------------TabNet Tests End-------------\n")

    def _predict(
        self, df: pd.DataFrame, model_name=None, additional_data=None, **kwargs
    ):
        return self.model.predict(
            df[self.trainer.feature_names].values.astype(np.float32)
        ).reshape(-1, 1)

    def _get_model_names(self):
        return ["TabNet"]


class TorchModel(AbstractModel):
    def __init__(self, trainer=None):
        super(TorchModel, self).__init__(trainer)

    def _new_model(self):
        raise NotImplementedError

    def _bayes(self) -> dict:
        """
        Running Gaussian process bayesian optimization on hyperparameters. Configurations are given in the configfile.
        chosen_params will be optimized.
        :return: A dict of optimized hyperparameters, which can be assigned to Trainer.params.
        """
        if not self.trainer.bayes_opt:
            print(
                "Bayes optimization not activated in configuration file. Return preset chosen_params."
            )
            return self.trainer.chosen_params

        # If def is not global, pickle will raise 'Can't get local attribute ...'
        # IT IS NOT SAFE, BUT I DID NOT FIND A BETTER SOLUTION
        global _trainer_bayes_objective, _trainer_bayes_callback

        bar = tqdm(total=self.trainer.n_calls)

        from copy import deepcopy as cp

        tmp_static_params = cp(self.trainer.static_params)
        # https://forums.fast.ai/t/hyperparameter-tuning-and-number-of-epochs/54935/2
        tmp_static_params["epoch"] = self.trainer.bayes_epoch
        tmp_static_params["patience"] = self.trainer.bayes_epoch

        @skopt.utils.use_named_args(self.trainer.SPACE)
        def _trainer_bayes_objective(**params):
            res, _, _ = self._model_train(
                model=self._new_model(),
                verbose=False,
                **{**params, **tmp_static_params},
            )

            return res

        postfix = {
            "Current loss": 1e8,
            "Minimum": 1e8,
            "Params": list(self.trainer.chosen_params.values()),
            "Minimum at call": 0,
        }

        def _trainer_bayes_callback(result):
            postfix["Current loss"] = result.func_vals[-1]

            if result.fun < postfix["Minimum"]:
                postfix["Minimum"] = result.fun
                postfix["Params"] = result.x
                postfix["Minimum at call"] = len(result.func_vals)
            skopt.dump(result, self.trainer.project_root + "skopt.pt")

            # if len(result.func_vals) % 5 == 0:
            #     plt.figure()
            #     ax = plt.subplot(111)
            #     ax = plot_convergence(result, ax)
            #     plt.savefig(self.project_root + 'skopt_convergence.pdf')
            #     plt.close()

            bar.set_postfix(**postfix)
            bar.update(1)

        result = gp_minimize(
            _trainer_bayes_objective,
            self.trainer.SPACE,
            n_calls=self.trainer.n_calls,
            random_state=0,
            x0=list(self.trainer.chosen_params.values()),
            callback=_trainer_bayes_callback,
        )
        print(result.func_vals.min())

        params = {}
        for key, value in zip(self.trainer.chosen_params.keys(), result.x):
            params[key] = value

        return params

    def _predict_all(self, verbose=True, test_data_only=False):
        self._check_train_status()

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

        if not test_data_only:
            y_train_pred, y_train, _ = test(
                self.model, train_loader, self.trainer.loss_fn
            )
            y_val_pred, y_val, _ = test(self.model, val_loader, self.trainer.loss_fn)
        else:
            y_train_pred = y_train = None
            y_val_pred = y_val = None
        y_test_pred, y_test, _ = test(self.model, test_loader, self.trainer.loss_fn)

        predictions = {}
        predictions[self._get_model_names()[0]] = {
            "Training": (y_train_pred, y_train),
            "Validation": (y_val_pred, y_val),
            "Testing": (y_test_pred, y_test),
        }

        return predictions

    def _predict(
        self, input_df: pd.DataFrame, model_name, additional_data: list = None, **kwargs
    ):
        df = self.trainer._data_transform(input_df.copy())
        X = torch.tensor(
            df[self.trainer.feature_names].values.astype(np.float32),
            dtype=torch.float32,
        ).to(self.trainer.device)
        D = [
            torch.tensor(value, dtype=torch.float32).to(self.trainer.device)
            for value in additional_data
        ]
        y = torch.tensor(np.zeros((len(df), 1)), dtype=torch.float32).to(
            self.trainer.device
        )

        loader = Data.DataLoader(
            Data.TensorDataset(X, *D, y), batch_size=len(df), shuffle=False
        )

        pred, _, _ = test(self.model, loader, self.trainer.loss_fn)
        return pred

    def _model_train(
        self,
        model,
        verbose=True,
        verbose_per_epoch=100,
        **params,
    ):
        train_loader = Data.DataLoader(
            self.trainer.train_dataset,
            batch_size=int(params["batch_size"]),
            generator=torch.Generator().manual_seed(0),
        )
        val_loader = Data.DataLoader(
            self.trainer.val_dataset,
            batch_size=len(self.trainer.val_dataset),
            generator=torch.Generator().manual_seed(0),
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        train_ls = []
        val_ls = []
        stop_epoch = params["epoch"]

        early_stopping = EarlyStopping(
            patience=params["patience"],
            verbose=False,
            path=self.trainer.project_root + "fatigue.pt",
        )

        for epoch in range(params["epoch"]):
            train_loss = train(model, train_loader, optimizer, self.trainer.loss_fn)
            train_ls.append(train_loss)
            _, _, val_loss = test(model, val_loader, self.trainer.loss_fn)
            val_ls.append(val_loss)

            if verbose and ((epoch + 1) % verbose_per_epoch == 0 or epoch == 0):
                print(
                    f"Epoch: {epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Min val loss: {np.min(val_ls):.4f}"
                )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                if verbose:
                    idx = val_ls.index(min(val_ls))
                    print(
                        f"Early stopping at epoch {epoch + 1}, Checkpoint at epoch {idx + 1}, Train loss: {train_ls[idx]:.4f}, Val loss: {val_ls[idx]:.4f}"
                    )
                break

        idx = val_ls.index(min(val_ls))
        min_loss = val_ls[idx]

        return min_loss, train_ls, val_ls

    def _train(
        self,
        verbose_per_epoch=100,
        verbose: bool = True,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        if not warm_start or (warm_start and not self._trained):
            self.model = self._new_model()

        min_loss, self.train_ls, self.val_ls = self._model_train(
            model=self.model,
            verbose=verbose,
            verbose_per_epoch=verbose_per_epoch,
            **{**self.trainer.params, **self.trainer.static_params},
        )

        self.model.load_state_dict(torch.load(self.trainer.project_root + "fatigue.pt"))

        if verbose:
            print(f"Minimum loss: {min_loss:.5f}")

        test_loader = Data.DataLoader(
            self.trainer.test_dataset,
            batch_size=len(self.trainer.test_dataset),
            generator=torch.Generator().manual_seed(0),
        )

        _, _, mse = test(self.model, test_loader, torch.nn.MSELoss())
        rmse = np.sqrt(mse)
        self.metrics = {"mse": mse, "rmse": rmse}

        if verbose:
            print(f"Test MSE loss: {mse:.5f}, RMSE loss: {rmse:.5f}")

        if dump_trainer:
            save_trainer(self.trainer, verbose=verbose)


class ThisWork(TorchModel):
    def __init__(self, trainer=None):
        super(ThisWork, self).__init__(trainer)
        self.program = "ThisWork"
        self._mkdir()

    def _new_model(self):
        return NN(
            len(self.trainer.feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            self.trainer._get_derived_data_sizes(),
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["ThisWork"]


class MLP(TorchModel):
    def __init__(self, trainer=None):
        super(MLP, self).__init__(trainer)
        self.program = "MLP"
        self._mkdir()

    def _new_model(self):
        return NN(
            len(self.trainer.feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            self.trainer._get_derived_data_sizes(),
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["MLP"]


class ModelAssembly(AbstractModel):
    def __init__(self, trainer=None, models=None, program=None):
        super(ModelAssembly, self).__init__(trainer)
        self.models = (
            [TabNet(self.trainer), MLP(self.trainer)] if models is None else models
        )
        self.program = "ModelAssembly" if program is None else program

    def fit(self, **kwargs):
        for submodel in self.models:
            submodel.fit(**kwargs)

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        return self.models[self._get_model_idx(model_name)].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def _train(
        self,
        **kwargs,
    ):
        print(f"\n-------------Run {self.program}-------------\n")
        for submodel in self.models:
            submodel._train(**kwargs)
        print(f"\n-------------{self.program} End-------------\n")

    def _predict(self, df: pd.DataFrame, model_name, additional_data=None, **kwargs):
        return self.models[self._get_model_idx(model_name)].predict(
            df=df, model_name=model_name, additional_data=additional_data, **kwargs
        )

    def _predict_all(self, **kwargs):
        self._check_train_status()
        predictions = {}
        for submodel in self.models:
            sub_predictions = submodel._predict_all(**kwargs)
            for key, value in sub_predictions.items():
                predictions[key] = value
        return predictions

    def _get_model_names(self):
        model_names = []
        for submodel in self.models:
            model_names += submodel._get_model_names()
        return model_names

    def _check_train_status(self):
        for submodel in self.models:
            try:
                submodel._check_train_status()
            except:
                raise Exception(
                    f"{self.program} not trained, run {self.__class__.__name__}._train() first."
                )

    def _get_model_idx(self, model_name):
        available_idx = []
        available_program = []
        for idx, submodel in enumerate(self.models):
            if model_name in submodel._get_model_names():
                available_idx.append(idx)
                available_program.append(submodel.program)
        if len(available_idx) == 1:
            return available_idx[0]
        elif len(available_idx) == 0:
            raise Exception(f"{model_name} not in the ModelAssembly.")
        else:
            raise Exception(
                f"Multiple {model_name} in the ModelAssembly (in {available_program})."
            )
