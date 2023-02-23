import numpy as np
import pandas as pd
import torch.optim

from .trainer import *
import skopt
from skopt import gp_minimize


class AbstractModel:
    def __init__(self, trainer: Trainer = None, program=None):
        self.trainer = trainer
        if not hasattr(trainer, "database"):
            trainer.load_config(default_configfile="base_config")
        self.model = None
        self.leaderboard = None
        self.program = self._get_program_name() if program is None else program
        self.params = None
        self._mkdir()

    def fit(
        self,
        df,
        feature_names: list,
        label_name: list,
        derived_data: dict = None,
        verbose=True,
        warm_start=False,
        bayes_opt=False,
    ):
        self.trainer.set_data(
            df,
            feature_names=feature_names,
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
        if model_name not in self._get_model_names():
            raise Exception(
                f"Model {model_name} is not available. Select among {self._get_model_names()}"
            )
        absent_features = [
            x
            for x in np.setdiff1d(
                self.trainer.feature_names, self.trainer.derived_stacked_features
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
        return self._predict(df, model_name, derived_data, **kwargs)

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

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        raise NotImplementedError

    def _train(self, dump_trainer=True, verbose=True, warm_start=False, **kwargs):
        raise NotImplementedError

    def _get_model_names(self):
        raise NotImplementedError

    def _get_program_name(self):
        raise NotImplementedError

    def _check_train_status(self):
        if not self._trained:
            raise Exception(
                f"{self.program} not trained, run {self.__class__.__name__}.train() first."
            )

    def _get_params(self, verbose=True):
        if self.params is None:
            return self._get_initial_params()
        else:
            if verbose:
                print(f"Previous params loaded: {self.params}")
            return self.params

    def _get_initial_params(self):
        return cp(self.trainer.chosen_params)

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
    def __init__(self, trainer=None, program=None):
        super(AutoGluon, self).__init__(trainer, program=program)

    def _get_program_name(self):
        return "AutoGluon"

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        disable_tqdm()
        import warnings

        warnings.simplefilter(action="ignore", category=UserWarning)
        from autogluon.tabular import TabularPredictor
        from autogluon.features import AutoMLPipelineFeatureGenerator

        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
        predictor = TabularPredictor(
            label=label_name[0], path=self.root, problem_type="regression"
        )
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
                feature_generator=AutoMLPipelineFeatureGenerator(
                    enable_categorical_features=False
                ),
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

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.model.predict(
            df[self.trainer.feature_names], model=model_name, **kwargs
        )

    def _get_model_names(self):
        return list(self.leaderboard["model"])


class WideDeep(AbstractModel):
    def __init__(self, trainer=None, program=None):
        super(WideDeep, self).__init__(trainer, program=program)
        self.model_params = {}

    def _get_program_name(self):
        return "WideDeep"

    def _get_model_params(self, model_name, verbose=True):
        if model_name not in self.model_params.keys():
            return self._get_initial_params()
        else:
            if verbose:
                print(f"Previous params loaded: {self.model_params[model_name]}.")
            return self.model_params[model_name]

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        # disable_tqdm()
        from pytorch_widedeep import Trainer as wd_Trainer
        from pytorch_widedeep.preprocessing import TabPreprocessor
        from pytorch_widedeep.callbacks import Callback, EarlyStopping
        from pytorch_widedeep.models import (
            WideDeep,
            TabMlp,
            TabResnet,
            TabTransformer,
            TabNet,
            SAINT,
            ContextAttentionMLP,
            SelfAttentionMLP,
            FTTransformer,
            TabPerceiver,
            TabFastFormer,
        )
        from typing import Optional, Dict

        tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
        tab_preprocessor = TabPreprocessor(continuous_cols=feature_names)
        X_tab_train = tab_preprocessor.fit_transform(
            tabular_dataset.loc[self.trainer.train_indices, :]
        )
        y_train = tabular_dataset.loc[self.trainer.train_indices, label_name].values
        X_tab_val = tab_preprocessor.transform(
            tabular_dataset.loc[self.trainer.val_indices, :]
        )
        y_val = tabular_dataset.loc[self.trainer.val_indices, label_name].values
        self.tab_preprocessor = tab_preprocessor

        args = dict(
            column_idx=tab_preprocessor.column_idx,
            continuous_cols=feature_names,
        )
        tab_models = {
            "TabMlp": TabMlp(**args),
            "TabResnet": TabResnet(**args),
            "TabTransformer": TabTransformer(embed_continuous=True, **args),
            "TabNet": TabNet(**args),
            "SAINT": SAINT(**args),
            "ContextAttentionMLP": ContextAttentionMLP(**args),
            "SelfAttentionMLP": SelfAttentionMLP(**args),
            "FTTransformer": FTTransformer(**args),
            "TabPerceiver": TabPerceiver(**args),
            "TabFastFormer": TabFastFormer(**args),
        }

        total_epoch = self.trainer.static_params["epoch"]

        global _WideDeepCallback

        class _WideDeepCallback(Callback):
            def __init__(self):
                super(_WideDeepCallback, self).__init__()
                self.val_ls = []

            def on_epoch_end(
                self,
                epoch: int,
                logs: Optional[Dict] = None,
                metric: Optional[float] = None,
            ):
                train_loss = logs["train_loss"]
                val_loss = logs["val_loss"]
                self.val_ls.append(val_loss)
                if epoch % 20 == 0 and verbose:
                    print(
                        f"Epoch: {epoch + 1}/{total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                        f"Min val loss: {np.min(self.val_ls):.4f}"
                    )

        self.model = {}
        for name, tab_model in tab_models.items():
            if verbose:
                print(f"Training {name}")
            self.params = self._get_model_params(name, verbose=verbose)
            model = WideDeep(deeptabular=tab_model)

            optimizer = torch.optim.Adam(
                model.deeptabular.parameters(),
                lr=self.params["lr"],
                weight_decay=self.params["weight_decay"],
            )

            wd_trainer = wd_Trainer(
                model,
                objective="regression",
                verbose=0,
                callbacks=[
                    EarlyStopping(
                        patience=self.trainer.static_params["patience"],
                        verbose=1 if verbose else 0,
                        restore_best_weights=True,
                    ),
                    _WideDeepCallback(),
                ],
                optimizers={"deeptabular": optimizer}
                if self.trainer.bayes_opt
                else None,
                num_workers=16,
                device=self.trainer.device,
            )

            if self.trainer.bayes_opt:
                callback = BayesCallback(
                    tqdm(total=self.trainer.n_calls, disable=not verbose)
                )
                global _widedeep_bayes_objective

                @skopt.utils.use_named_args(self.trainer.SPACE)
                def _widedeep_bayes_objective(**params):
                    with HiddenPrints(disable_logging=True):
                        tmp_model = cp(model)
                        tmp_wd_trainer = wd_Trainer(
                            tmp_model,
                            objective="regression",
                            verbose=0,
                            optimizers={
                                "deeptabular": torch.optim.Adam(
                                    tmp_model.deeptabular.parameters(),
                                    lr=params["lr"],
                                    weight_decay=params["weight_decay"],
                                )
                            },
                            num_workers=16,
                            device=self.trainer.device,
                        )

                    tmp_wd_trainer.fit(
                        X_train={"X_tab": X_tab_train, "target": y_train},
                        X_val={"X_tab": X_tab_val, "target": y_val},
                        n_epochs=self.trainer.bayes_epoch,
                        batch_size=int(params["batch_size"]),
                    )

                    pred = tmp_wd_trainer.predict(X_tab=X_tab_val)
                    res = Trainer._metric_sklearn(pred, y_val, self.trainer.loss)
                    return res

                with warnings.catch_warnings():
                    # To obtain clean progress bar.
                    warnings.simplefilter("ignore")
                    result = gp_minimize(
                        _widedeep_bayes_objective,
                        self.trainer.SPACE,
                        n_calls=self.trainer.n_calls,
                        callback=callback.call,
                        random_state=0,
                        x0=list(self.params.values()),
                    )
                params = {}
                for key, value in zip(self.params.keys(), result.x):
                    params[key] = value
                    if key in optimizer.defaults.keys():
                        optimizer.defaults[key] = value
                self.params = params
                self.model_params[name] = cp(params)
                callback.close()
                skopt.dump(result, self.trainer.project_root + "skopt.pt")
                self.params = self._get_model_params(
                    model_name=name, verbose=verbose
                )  # to announce the optimized params.

            wd_trainer.fit(
                X_train={"X_tab": X_tab_train, "target": y_train},
                X_val={"X_tab": X_tab_val, "target": y_val},
                n_epochs=total_epoch,
                batch_size=int(self.params["batch_size"]),
            )
            self.model[name] = wd_trainer

        # enable_tqdm()
        if dump_trainer:
            save_trainer(self.trainer)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        # SettingWithCopyWarning in TabPreprocessor.transform
        # i.e. df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        pd.set_option("mode.chained_assignment", "warn")
        X_df = self.tab_preprocessor.transform(df)
        pd.set_option("mode.chained_assignment", "raise")
        return self.model[model_name].predict(X_tab=X_df)

    def _get_model_names(self):
        return [
            "TabMlp",
            "TabResnet",
            "TabTransformer",
            "TabNet",
            "SAINT",
            "ContextAttentionMLP",
            "SelfAttentionMLP",
            "FTTransformer",
            "TabPerceiver",
            "TabFastFormer",
        ]


# class PytorchTabular(AbstractModel):
#     def __init__(self, trainer=None):
#         super(PytorchTabular, self).__init__(trainer)
#
#     def _get_program_name(self):
#         return "PytorchTabular"
#
#     def _train(
#         self,
#         verbose: bool = False,
#         debug_mode: bool = False,
#         dump_trainer=True,
#         warm_start=False,
#         **kwargs,
#     ):
#         print("\n-------------Run Pytorch-tabular-------------\n")
#         disable_tqdm()
#         import warnings
#
#         warnings.simplefilter(action="ignore", category=UserWarning)
#         from functools import partialmethod
#         from pytorch_tabular.config import ExperimentRunManager
#
#         ExperimentRunManager.__init__ = partialmethod(
#             ExperimentRunManager.__init__,
#             exp_version_manager=self.root + "exp_version_manager.yml",
#         )
#         from pytorch_tabular import TabularModel
#         from pytorch_tabular.models import (
#             CategoryEmbeddingModelConfig,
#             NodeConfig,
#             TabNetModelConfig,
#             TabTransformerConfig,
#             AutoIntConfig,
#             FTTransformerConfig,
#         )
#         from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
#
#         tabular_dataset, feature_names, label_name = self.trainer._get_tabular_dataset()
#
#         data_config = DataConfig(
#             target=label_name, continuous_cols=feature_names, num_workers=8
#         )
#
#         trainer_config = TrainerConfig(
#             max_epochs=self.trainer.static_params["epoch"] if not debug_mode else 10,
#             early_stopping_patience=self.trainer.static_params["patience"],
#         )
#         optimizer_config = OptimizerConfig()
#
#         model_configs = [
#             CategoryEmbeddingModelConfig(task="regression"),
#             # NodeConfig(task='regression'),
#             TabNetModelConfig(task="regression"),
#             TabTransformerConfig(task="regression"),
#             AutoIntConfig(task="regression"),
#             # FTTransformerConfig(task='regression')
#         ]
#
#         # dtype=int because pytorch-tabular can not convert np.int64 to Integer
#         # All commented lines in SPACEs cause error. Do not uncomment them.
#         SPACEs = {
#             "CategoryEmbeddingModel": {
#                 "SPACE": [
#                     Real(low=0, high=0.8, prior="uniform", name="dropout"),  # 0.5
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="embedding_dropout"
#                     ),  # 0.5
#                     Real(
#                         low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
#                     ),  # 0.001
#                 ],
#                 "defaults": [0.5, 0.5, 0.001],
#                 "class": CategoryEmbeddingModelConfig,
#             },
#             "NODEModel": {
#                 "SPACE": [
#                     Integer(
#                         low=2, high=6, prior="uniform", name="depth", dtype=int
#                     ),  # 6
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="embedding_dropout"
#                     ),  # 0.0
#                     Real(low=0, high=0.8, prior="uniform", name="input_dropout"),  # 0.0
#                     Real(
#                         low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
#                     ),  # 0.001
#                     Integer(
#                         low=128, high=512, prior="uniform", name="num_trees", dtype=int
#                     ),
#                 ],
#                 "defaults": [6, 0.0, 0.0, 0.001, 256],
#                 "class": NodeConfig,
#             },
#             "TabNetModel": {
#                 "SPACE": [
#                     Integer(
#                         low=4, high=64, prior="uniform", name="n_d", dtype=int
#                     ),  # 8
#                     Integer(
#                         low=4, high=64, prior="uniform", name="n_a", dtype=int
#                     ),  # 8
#                     Integer(
#                         low=3, high=10, prior="uniform", name="n_steps", dtype=int
#                     ),  # 3
#                     Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
#                     Integer(
#                         low=1, high=5, prior="uniform", name="n_independent", dtype=int
#                     ),  # 2
#                     Integer(
#                         low=1, high=5, prior="uniform", name="n_shared", dtype=int
#                     ),  # 2
#                     Real(
#                         low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
#                     ),  # 0.001
#                 ],
#                 "defaults": [8, 8, 3, 1.3, 2, 2, 0.001],
#                 "class": TabNetModelConfig,
#             },
#             "TabTransformerModel": {
#                 "SPACE": [
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="embedding_dropout"
#                     ),  # 0.1
#                     Real(low=0, high=0.8, prior="uniform", name="ff_dropout"),  # 0.1
#                     # Categorical([16, 32, 64, 128], name='input_embed_dim'),  # 32
#                     Real(
#                         low=0,
#                         high=0.5,
#                         prior="uniform",
#                         name="shared_embedding_fraction",
#                     ),  # 0.25
#                     # Categorical([2, 4, 8, 16], name='num_heads'),  # 8
#                     Integer(
#                         low=4,
#                         high=8,
#                         prior="uniform",
#                         name="num_attn_blocks",
#                         dtype=int,
#                     ),  # 6
#                     Real(low=0, high=0.8, prior="uniform", name="attn_dropout"),  # 0.1
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="add_norm_dropout"
#                     ),  # 0.1
#                     Integer(
#                         low=2,
#                         high=6,
#                         prior="uniform",
#                         name="ff_hidden_multiplier",
#                         dtype=int,
#                     ),  # 4
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="out_ff_dropout"
#                     ),  # 0.0
#                     Real(
#                         low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
#                     ),  # 0.001
#                 ],
#                 "defaults": [0.1, 0.1, 0.25, 6, 0.1, 0.1, 4, 0.0, 0.001],
#                 "class": TabTransformerConfig,
#             },
#             "AutoIntModel": {
#                 "SPACE": [
#                     Real(
#                         low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
#                     ),  # 0.001
#                     Real(low=0, high=0.8, prior="uniform", name="attn_dropouts"),  # 0.0
#                     # Categorical([16, 32, 64, 128], name='attn_embed_dim'),  # 32
#                     Real(low=0, high=0.8, prior="uniform", name="dropout"),  # 0.0
#                     # Categorical([8, 16, 32, 64], name='embedding_dim'),  # 16
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="embedding_dropout"
#                     ),  # 0.0
#                     Integer(
#                         low=1,
#                         high=4,
#                         prior="uniform",
#                         name="num_attn_blocks",
#                         dtype=int,
#                     ),  # 3
#                     # Integer(low=1, high=4, prior='uniform', name='num_heads', dtype=int),  # 2
#                 ],
#                 "defaults": [0.001, 0.0, 0.0, 0.0, 3],
#                 "class": AutoIntConfig,
#             },
#             "FTTransformerModel": {
#                 "SPACE": [
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="embedding_dropout"
#                     ),  # 0.1
#                     Real(
#                         low=0,
#                         high=0.5,
#                         prior="uniform",
#                         name="shared_embedding_fraction",
#                     ),  # 0.25
#                     Integer(
#                         low=4,
#                         high=8,
#                         prior="uniform",
#                         name="num_attn_blocks",
#                         dtype=int,
#                     ),  # 6
#                     Real(low=0, high=0.8, prior="uniform", name="attn_dropout"),  # 0.1
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="add_norm_dropout"
#                     ),  # 0.1
#                     Real(low=0, high=0.8, prior="uniform", name="ff_dropout"),  # 0.1
#                     Integer(
#                         low=2,
#                         high=6,
#                         prior="uniform",
#                         name="ff_hidden_multiplier",
#                         dtype=int,
#                     ),  # 4
#                     Real(
#                         low=0, high=0.8, prior="uniform", name="out_ff_dropout"
#                     ),  # 0.0
#                 ],
#                 "defaults": [0.1, 0.25, 6, 0.1, 0.1, 0.1, 4, 0.0],
#                 "class": FTTransformerConfig,
#             },
#         }
#
#         metrics = {"model": [], "mse": []}
#         models = {}
#         for model_config in model_configs:
#             model_name = model_config._model_name
#             print("Training", model_name)
#
#             SPACE = SPACEs[model_name]["SPACE"]
#             defaults = SPACEs[model_name]["defaults"]
#             config_class = SPACEs[model_name]["class"]
#             exceptions = []
#             global _pytorch_tabular_bayes_objective
#
#             @skopt.utils.use_named_args(SPACE)
#             def _pytorch_tabular_bayes_objective(**params):
#                 if verbose:
#                     print(params, end=" ")
#                 with HiddenPrints(disable_logging=True):
#                     tabular_model = TabularModel(
#                         data_config=data_config,
#                         model_config=config_class(task="regression", **params),
#                         optimizer_config=optimizer_config,
#                         trainer_config=TrainerConfig(
#                             max_epochs=self.trainer.bayes_epoch,
#                             early_stopping_patience=self.trainer.bayes_epoch,
#                         ),
#                     )
#                     tabular_model.config.checkpoints = None
#                     tabular_model.config["progress_bar_refresh_rate"] = 0
#                     try:
#                         tabular_model.fit(
#                             train=tabular_dataset.loc[
#                                 self.trainer.train_dataset.indices, :
#                             ],
#                             validation=tabular_dataset.loc[
#                                 self.trainer.val_dataset.indices, :
#                             ],
#                             loss=self.trainer.loss_fn,
#                         )
#                     except Exception as e:
#                         exceptions.append(e)
#                         return 1e3
#                     res = tabular_model.evaluate(
#                         tabular_dataset.loc[self.trainer.val_dataset.indices, :]
#                     )[0]["test_mean_squared_error"]
#                 if verbose:
#                     print(res)
#                 return res
#
#             if not debugger_is_active() and self.trainer.bayes_opt:
#                 # otherwise: AssertionError: can only test a child process
#                 result = gp_minimize(
#                     _pytorch_tabular_bayes_objective,
#                     SPACE,
#                     x0=defaults,
#                     n_calls=self.trainer.n_calls if not debug_mode else 11,
#                     random_state=0,
#                 )
#                 param_names = [x.name for x in SPACE]
#                 params = {}
#                 for key, value in zip(param_names, result.x):
#                     params[key] = value
#                 model_config = config_class(task="regression", **params)
#
#                 if len(exceptions) > 0 and verbose:
#                     print("Exceptions in bayes optimization:", exceptions)
#
#             with HiddenPrints(disable_logging=True if not verbose else False):
#                 tabular_model = TabularModel(
#                     data_config=data_config,
#                     model_config=model_config,
#                     optimizer_config=optimizer_config,
#                     trainer_config=trainer_config,
#                 )
#                 tabular_model.config.checkpoints_path = self.root
#                 tabular_model.config["progress_bar_refresh_rate"] = 0
#
#                 tabular_model.fit(
#                     train=tabular_dataset.loc[self.trainer.train_dataset.indices, :],
#                     validation=tabular_dataset.loc[self.trainer.val_dataset.indices, :],
#                     loss=self.trainer.loss_fn,
#                 )
#                 metrics["model"].append(tabular_model.config._model_name)
#                 metrics["mse"].append(
#                     tabular_model.evaluate(
#                         tabular_dataset.loc[self.trainer.test_dataset.indices, :]
#                     )[0]["test_mean_squared_error"]
#                 )
#                 models[tabular_model.config._model_name] = tabular_model
#
#         metrics = pd.DataFrame(metrics)
#         metrics["rmse"] = metrics["mse"] ** (1 / 2)
#         metrics.sort_values("rmse", inplace=True)
#         self.leaderboard = metrics
#         self.model = models
#
#         if verbose:
#             print(self.leaderboard)
#         self.leaderboard.to_csv(self.root + "leaderboard.csv")
#
#         enable_tqdm()
#         warnings.simplefilter(action="default", category=UserWarning)
#         if dump_trainer:
#             save_trainer(self.trainer)
#         print("\n-------------Pytorch-tabular End-------------\n")
#
#     def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
#         model = self.model[model_name]
#         target = model.config.target[0]
#         return np.array(model.predict(df)[f"{target}_prediction"])
#
#     def _get_model_names(self):
#         return list(self.leaderboard["model"])


class TabNet(AbstractModel):
    def __init__(self, trainer=None, program=None):
        super(TabNet, self).__init__(trainer, program=program)
        self.additional_params = None

    def _get_program_name(self):
        return "TabNet"

    def _get_additional_params(self, verbose=True):
        if self.additional_params is None:
            return [8, 8, 3, 1.3, 2, 2]
        else:
            if verbose:
                print(f"Previous additional params loaded: {self.additional_params}.")
            return list(self.additional_params.values())

    def _train(
        self,
        verbose: bool = False,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        # Basic components in _train():
        # 1. Prepare training, validation, and testing dataset.
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

        # 2. Setup a scikit-optimize search space and its initial values.
        SPACE = [
            Integer(low=4, high=64, prior="uniform", name="n_d", dtype=int),  # 8
            Integer(low=4, high=64, prior="uniform", name="n_a", dtype=int),  # 8
            Integer(low=3, high=10, prior="uniform", name="n_steps", dtype=int),  # 3
            Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
            Integer(
                low=1, high=5, prior="uniform", name="n_independent", dtype=int
            ),  # 2
            Integer(low=1, high=5, prior="uniform", name="n_shared", dtype=int),  # 2
        ] + self.trainer.SPACE
        param_names = [x.name for x in SPACE]

        def extract_params(keys, values):
            params = {}
            optim_params = {}
            batch_size = 32
            for key, value in zip(keys, values):
                if key in self.params.keys():
                    if key != "batch_size":
                        optim_params[key] = value
                else:
                    params[key] = value
                if key == "batch_size":
                    batch_size = int(value)
            return params, optim_params, batch_size

        # 3. Define a objective function that receives parameters, generates a new model, sets parameters, fits the
        # model, predicts on validation dataset, and returns a scalar metric.
        global _tabnet_bayes_objective

        @skopt.utils.use_named_args(SPACE)
        def _tabnet_bayes_objective(**params):
            # 3.1 Receive parameters to be set.
            params, optim_params, batch_size = extract_params(
                params.keys(), params.values()
            )

            with HiddenPrints(disable_logging=True):
                # 3.2 Generate a new model
                model = TabNetRegressor(verbose=0, optimizer_params=optim_params)
                # 3.3 Set parameters
                model.set_params(**params)
                # 3.4 Fits the model
                model.fit(
                    train_x,
                    train_y,
                    eval_set=eval_set,
                    max_epochs=self.trainer.bayes_epoch,
                    patience=self.trainer.bayes_epoch,
                    loss_fn=self.trainer.loss_fn,
                    eval_metric=[self.trainer.loss],
                    batch_size=batch_size,
                )
                # 3.5 Predicts on the validation dataset.
                res = self.trainer._metric_sklearn(
                    model.predict(val_x).reshape(-1, 1), val_y, self.trainer.loss
                )
            # 3.6 Returns a scalar metric.
            return res

        # 4. If trainer.bayes_opt is True, run bayesian hyperparameter searching.
        if not debugger_is_active() and self.trainer.bayes_opt:
            # debugger_is_active() otherwise: AssertionError: can only test a child process
            self.params = self._get_params(verbose=verbose)
            defaults = self._get_additional_params(verbose=verbose) + list(
                self.params.values()
            )
            callback = BayesCallback(
                tqdm(total=self.trainer.n_calls, disable=not verbose)
            )
            result = gp_minimize(
                _tabnet_bayes_objective,
                SPACE,
                x0=defaults,
                n_calls=self.trainer.n_calls if not debug_mode else 11,
                random_state=0,
                callback=callback.call,
            )
            params, optim_params, batch_size = extract_params(param_names, result.x)
            callback.close()
            skopt.dump(result, self.trainer.project_root + "skopt.pt")

            # Note: Set params for later usage, i.e. self._get_params().
            for name, value in zip(param_names, result.x):
                if name in self.params.keys():
                    self.params[name] = value
            self.additional_params = params
        # Note: Set initial chosen_params if bayes has not been run, otherwise optimized params are loaded.
        self.params = self._get_params(verbose=verbose)
        defaults = self._get_additional_params(verbose=verbose) + list(
            self.params.values()
        )
        params, optim_params, batch_size = extract_params(param_names, defaults)

        # 5. Generate a new model, set parameters based on chosen_params or results from bayesopt, and train the model.
        model = TabNetRegressor(
            verbose=20 if verbose else 0, optimizer_params=optim_params
        )
        model.set_params(**params)
        model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            max_epochs=self.trainer.static_params["epoch"],
            patience=self.trainer.static_params["patience"],
            loss_fn=self.trainer.loss_fn,
            eval_metric=[self.trainer.loss],
            batch_size=batch_size,
        )

        # Optional: Get some instant results.
        y_test_pred = model.predict(test_x).reshape(-1, 1)
        print(
            "MSE Loss:",
            self.trainer._metric_sklearn(y_test_pred, test_y, "mse"),
            "RMSE Loss:",
            self.trainer._metric_sklearn(y_test_pred, test_y, "rmse"),
        )
        # 6. Record the model
        self.model = model
        # Optional: Dump the trainer.
        if dump_trainer:
            save_trainer(self.trainer)

    def _predict(self, df: pd.DataFrame, model_name=None, derived_data=None, **kwargs):
        # Basic components in _predict():
        # Return a ndarray with shape (len(df), 1) of predictions by the model_name.
        return self.model.predict(
            df[self.trainer.feature_names].values.astype(np.float32)
        ).reshape(-1, 1)

    def _get_model_names(self):
        return ["TabNet"]


class TorchModel(AbstractModel):
    def __init__(self, trainer=None, program=None):
        super(TorchModel, self).__init__(trainer, program=program)

    def _new_model(self):
        raise NotImplementedError

    def _bayes(self, verbose=True):
        """
        Running Gaussian process bayesian optimization on hyperparameters. Configurations are given in the configfile.
        chosen_params will be optimized.
        """
        if not self.trainer.bayes_opt:
            return None

        self.params = self._get_params(verbose=verbose)

        # If def is not global, pickle will raise 'Can't get local attribute ...'
        # IT IS NOT SAFE, BUT I DID NOT FIND A BETTER SOLUTION
        global _trainer_bayes_objective, _trainerBayesCallback

        callback = BayesCallback(tqdm(total=self.trainer.n_calls, disable=not verbose))

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

        with warnings.catch_warnings():
            # To obtain clean progress bar.
            warnings.simplefilter("ignore")
            result = gp_minimize(
                _trainer_bayes_objective,
                self.trainer.SPACE,
                n_calls=self.trainer.n_calls,
                random_state=0,
                x0=list(self.params.values()),
                callback=callback.call,
            )
        callback.close()
        skopt.dump(result, self.trainer.project_root + "skopt.pt")

        params = {}
        for key, value in zip(self.params.keys(), result.x):
            params[key] = value

        self.params = params

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
            y_train_pred, y_train, _ = self._test_step(
                self.model, train_loader, self.trainer.loss_fn
            )
            y_val_pred, y_val, _ = self._test_step(
                self.model, val_loader, self.trainer.loss_fn
            )
        else:
            y_train_pred = y_train = None
            y_val_pred = y_val = None
        y_test_pred, y_test, _ = self._test_step(
            self.model, test_loader, self.trainer.loss_fn
        )

        predictions = {}
        predictions[self._get_model_names()[0]] = {
            "Training": (y_train_pred, y_train),
            "Validation": (y_val_pred, y_val),
            "Testing": (y_test_pred, y_test),
        }

        return predictions

    def _predict(
        self, input_df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        df = self.trainer.data_transform(input_df)
        X = torch.tensor(
            input_df[self.trainer.feature_names].values.astype(np.float32),
            dtype=torch.float32,
        ).to(self.trainer.device)
        D = [
            torch.tensor(value, dtype=torch.float32).to(self.trainer.device)
            for value in derived_data.values()
        ]
        y = torch.tensor(np.zeros((len(input_df), 1)), dtype=torch.float32).to(
            self.trainer.device
        )

        loader = Data.DataLoader(
            Data.TensorDataset(X, *D, y), batch_size=len(input_df), shuffle=False
        )

        pred, _, _ = self._test_step(self.model, loader, self.trainer.loss_fn)
        return pred

    def _model_train(
        self,
        model,
        verbose=True,
        verbose_per_epoch=20,
        warm_start=False,
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
            model.parameters(),
            lr=params["lr"] / 10 if warm_start else params["lr"],
            weight_decay=params["weight_decay"],
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
            train_loss = self._train_step(
                model, train_loader, optimizer, self.trainer.loss_fn
            )
            train_ls.append(train_loss)
            _, _, val_loss = self._test_step(model, val_loader, self.trainer.loss_fn)
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

    def _train(
        self,
        verbose_per_epoch=20,
        verbose: bool = True,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        if not warm_start or (warm_start and not self._trained):
            self.model = self._new_model()

        self._bayes(verbose=verbose)

        self.params = self._get_params(verbose=verbose)

        min_loss, self.train_ls, self.val_ls = self._model_train(
            model=self.model,
            verbose=verbose,
            verbose_per_epoch=verbose_per_epoch,
            warm_start=warm_start,
            **{**self.params, **self.trainer.static_params},
        )

        self.model.load_state_dict(torch.load(self.trainer.project_root + "fatigue.pt"))

        if verbose:
            print(f"Minimum loss: {min_loss:.5f}")

        test_loader = Data.DataLoader(
            self.trainer.test_dataset,
            batch_size=len(self.trainer.test_dataset),
            generator=torch.Generator().manual_seed(0),
        )

        _, _, mse = self._test_step(self.model, test_loader, torch.nn.MSELoss())
        rmse = np.sqrt(mse)
        self.metrics = {"mse": mse, "rmse": rmse}

        if verbose:
            print(f"Test MSE loss: {mse:.5f}, RMSE loss: {rmse:.5f}")

        if dump_trainer:
            save_trainer(self.trainer, verbose=verbose)


class ThisWork(TorchModel):
    def __init__(self, trainer=None, manual_activate=None, program=None):
        super(ThisWork, self).__init__(trainer, program=program)
        self.activated_sn = None
        self.manual_activate = manual_activate
        from src.core.sn_formulas import sn_mapping

        if self.manual_activate is not None:
            for sn in self.manual_activate:
                if sn not in sn_mapping.keys():
                    raise Exception(f"SN model {sn} is not implemented or activated.")

    def _get_program_name(self):
        return "ThisWork"

    def _new_model(self):
        from src.core.sn_formulas import sn_mapping

        if self.activated_sn is None:
            self.activated_sn = []
            for key, sn in sn_mapping.items():
                if sn.test_sn_vars(self.trainer) and (
                    self.manual_activate is None or key in self.manual_activate
                ):
                    self.activated_sn.append(sn(self.trainer))
            print(
                f"Activated SN models: {[sn.__class__.__name__ for sn in self.activated_sn]}"
            )
        set_torch_random(0)
        return ThisWorkNN(
            len(self.trainer.feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["ThisWork"]


class ThisWorkRidge(ThisWork):
    def __init__(self, trainer=None, manual_activate=None, program=None):
        super(ThisWorkRidge, self).__init__(
            trainer=trainer, manual_activate=manual_activate, program=program
        )

    def _get_program_name(self):
        return "ThisWorkRidge"

    def _new_model(self):
        from src.core.sn_formulas import sn_mapping

        if self.activated_sn is None:
            self.activated_sn = []
            for key, sn in sn_mapping.items():
                if sn.test_sn_vars(self.trainer) and (
                    self.manual_activate is None or key in self.manual_activate
                ):
                    self.activated_sn.append(sn(self.trainer))
            print(
                f"Activated SN models: {[sn.__class__.__name__ for sn in self.activated_sn]}"
            )
        set_torch_random(0)
        return ThisWorkRidgeNN(
            len(self.trainer.feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _train_step(self, model, train_loader, optimizer, loss_fn):
        model.train()
        avg_loss = 0
        for idx, tensors in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(*([data] + additional_tensors))
            self.ridge(model, yhat)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * len(y)

        avg_loss /= len(train_loader.dataset)
        return avg_loss

    def ridge(self, model, yhat, alpha=0.2):
        # https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
        X = model.preds
        y = yhat.view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        lhs = X.T @ X
        rhs = X.T @ y
        ridge = alpha * torch.eye(lhs.shape[0])
        w = torch.linalg.lstsq(rhs, lhs + ridge).solution
        from copy import copy

        model.component_weights = copy(w.T)
        # print(model.component_weights)


class ThisWorkPretrain(ThisWork):
    def __init__(self, trainer=None, manual_activate=None, program=None):
        super(ThisWorkPretrain, self).__init__(
            trainer=trainer, manual_activate=manual_activate, program=program
        )

    def train(self, *args, **kwargs):
        tmp_kwargs = cp(kwargs)
        verbose = "verbose" not in kwargs.keys() or kwargs["verbose"]
        if verbose:
            print(f"\n-------------Run {self.program}-------------\n")
        original_df = cp(self.trainer.df)
        train_indices = cp(self.trainer.train_indices)
        val_indices = cp(self.trainer.val_indices)
        test_indices = cp(self.trainer.test_indices)
        if not ("warm_start" in kwargs.keys() and kwargs["warm_start"]):
            if verbose:
                print("Pretraining...")
            mat_cnt = self.trainer.get_material_code(unique=True, partition="train")
            mat_cnt.sort_values(by=["Count"], ascending=False, inplace=True)
            selected = len(mat_cnt) // 10 + 1
            abundant_mat = mat_cnt["Material_Code"][:selected].values
            abundant_mat_indices = np.concatenate(
                [
                    self.trainer._select_by_material_code(
                        m_code=code, partition="train"
                    ).values
                    for code in abundant_mat
                ]
            )
            selected_df = (
                self.trainer.df.loc[abundant_mat_indices, :]
                .copy()
                .reset_index(drop=True)
            )
            self.trainer.set_data(
                selected_df,
                feature_names=self.trainer.feature_names,
                label_name=self.trainer.label_name,
                warm_start=True,
                verbose=verbose,
                all_training=True,
            )
            if verbose:
                print(
                    f"{selected} materials ({len(abundant_mat_indices)} records) are selected for pretraining."
                )
            tmp_kwargs["warm_start"] = False
            tmp_kwargs["verbose"] = verbose
            self._train(*args, **tmp_kwargs)

        if verbose:
            print("Finetunning...")
        self.trainer.set_data(
            original_df,
            feature_names=self.trainer.feature_names,
            label_name=self.trainer.label_name,
            warm_start=True,
            verbose=verbose,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
        tmp_kwargs["warm_start"] = True
        tmp_kwargs["verbose"] = verbose
        tmp_bayes_opt = cp(self.trainer.bayes_opt)
        self.trainer.bayes_opt = False
        self._train(*args, **tmp_kwargs)
        self.trainer.bayes_opt = tmp_bayes_opt

        if verbose:
            print(f"\n-------------{self.program} End-------------\n")

    def _get_program_name(self):
        return "ThisWorkPretrain"


class MLP(TorchModel):
    def __init__(self, trainer=None, program=None, layers=None):
        super(MLP, self).__init__(trainer, program=program)
        self.layers = layers

    def _get_program_name(self):
        return "MLP"

    def _new_model(self):
        set_torch_random(0)
        return NN(
            len(self.trainer.feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["MLP"]


class ModelAssembly(AbstractModel):
    def __init__(self, trainer=None, models=None, program=None):
        self.program = "ModelAssembly" if program is None else program
        super(ModelAssembly, self).__init__(trainer, program=self.program)
        self.models = (
            [TabNet(self.trainer), MLP(self.trainer)] if models is None else models
        )

    def _get_program_name(self):
        return self.program

    def fit(self, **kwargs):
        for submodel in self.models:
            submodel.fit(**kwargs)

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        return self.models[self._get_model_idx(model_name)].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def train(
        self,
        *args,
        **kwargs,
    ):
        print(f"\n-------------Run {self.program}-------------\n")
        for submodel in self.models:
            submodel.train(*args, **kwargs)
        print(f"\n-------------{self.program} End-------------\n")

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.models[self._get_model_idx(model_name)].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
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
                    f"{self.program} not trained, run {self.__class__.__name__}.train() first."
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
