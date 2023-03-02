import numpy as np
import pandas as pd
import torch.optim
from typing import Type
from src.trainer.trainer import Trainer, save_trainer
from skopt.space import Real, Integer, Categorical
import torch.utils.data as Data
from src.utils.utils import *
import skopt
from skopt import gp_minimize


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
            derived_data=derived_data,
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


class AutoGluon(AbstractModel):
    def __init__(self, trainer=None, program=None, model_subset=None):
        super(AutoGluon, self).__init__(
            trainer, program=program, model_subset=model_subset
        )

    def _get_program_name(self):
        return "AutoGluon"

    def _train(
        self,
        verbose: bool = False,
        model_subset: list = None,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        disable_tqdm()
        import warnings

        if model_subset is not None:
            warnings.warn(
                f"AutoGluon does not support training models separately, but a model_subset is passed to AutoGluon.",
                category=UserWarning,
            )
        warnings.simplefilter(action="ignore", category=UserWarning)
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import PipelineFeatureGenerator
        from autogluon.features.generators.category import CategoryFeatureGenerator
        from autogluon.features.generators.identity import IdentityFeatureGenerator
        from autogluon.common.features.feature_metadata import FeatureMetadata
        from autogluon.common.features.types import R_INT, R_FLOAT

        (
            tabular_dataset,
            cont_feature_names,
            cat_feature_names,
            label_name,
        ) = self.trainer.get_tabular_dataset()
        tabular_dataset = self.trainer.categories_inverse_transform(tabular_dataset)
        predictor = TabularPredictor(
            label=label_name[0], path=self.root, problem_type="regression"
        )
        feature_metadata = {}
        for feature in cont_feature_names:
            feature_metadata[feature] = "float"
        for feature in cat_feature_names:
            feature_metadata[feature] = "object"
        feature_generator = PipelineFeatureGenerator(
            generators=[
                [
                    IdentityFeatureGenerator(
                        infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT]),
                        feature_metadata_in=FeatureMetadata(feature_metadata),
                    ),
                    CategoryFeatureGenerator(
                        feature_metadata_in=FeatureMetadata(feature_metadata)
                    ),
                ]
            ]
        )
        with HiddenPrints(disable_logging=True if not verbose else False):
            predictor.fit(
                tabular_dataset.loc[self.trainer.train_indices, :],
                tuning_data=tabular_dataset.loc[self.trainer.val_indices, :],
                presets="best_quality"
                if not debug_mode
                else "medium_quality_faster_train",
                hyperparameter_tune_kwargs="bayesopt"
                if (not debug_mode) and self.trainer.bayes_opt
                else None,
                use_bag_holdout=True,
                verbosity=0 if not verbose else 2,
                feature_generator=feature_generator,
            )
        self.leaderboard = predictor.leaderboard(
            tabular_dataset.loc[self.trainer.test_indices, :], silent=True
        )
        self.leaderboard.to_csv(self.root + "leaderboard.csv")
        self.model = predictor
        enable_tqdm()
        warnings.simplefilter(action="default", category=UserWarning)
        if dump_trainer:
            save_trainer(self.trainer)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.model.predict(
            df[self.trainer.all_feature_names], model=model_name, **kwargs
        )

    def _get_model_names(self):
        return list(self.leaderboard["model"])


class WideDeep(AbstractModel):
    def __init__(self, trainer=None, program=None, model_subset=None):
        super(WideDeep, self).__init__(
            trainer, program=program, model_subset=model_subset
        )

    def _get_program_name(self):
        return "WideDeep"

    def _space(self, model_name):
        return self.trainer.SPACE

    def _initial_values(self, model_name):
        return self.trainer.chosen_params

    def _new_model(self, model_name, verbose, **kwargs):
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
        from pytorch_widedeep import Trainer as wd_Trainer
        from pytorch_widedeep.callbacks import Callback, EarlyStopping
        from typing import Optional, Dict

        cont_feature_names = self.trainer.cont_feature_names
        cat_feature_names = self.trainer.cat_feature_names

        args = dict(
            column_idx=self.tab_preprocessor.column_idx,
            continuous_cols=cont_feature_names,
            cat_embed_input=self.tab_preprocessor.cat_embed_input
            if len(cat_feature_names) != 0
            else None,
        )

        if model_name == "TabTransformer":
            args["embed_continuous"] = True if len(cat_feature_names) == 0 else False

        mapping = {
            "TabMlp": TabMlp,
            "TabResnet": TabResnet,
            "TabTransformer": TabTransformer,
            "TabNet": TabNet,
            "SAINT": SAINT,
            "ContextAttentionMLP": ContextAttentionMLP,
            "SelfAttentionMLP": SelfAttentionMLP,
            "FTTransformer": FTTransformer,
            "TabPerceiver": TabPerceiver,
            "TabFastFormer": TabFastFormer,
        }

        tab_model = mapping[model_name](**args)
        model = WideDeep(deeptabular=tab_model)

        optimizer = torch.optim.Adam(
            model.deeptabular.parameters(),
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )

        global _WideDeepCallback

        class _WideDeepCallback(Callback):
            def __init__(self):
                super(_WideDeepCallback, self).__init__()
                self.val_ls = []

            def on_epoch_end(
                callback_self,
                epoch: int,
                logs: Optional[Dict] = None,
                metric: Optional[float] = None,
            ):
                train_loss = logs["train_loss"]
                val_loss = logs["val_loss"]
                callback_self.val_ls.append(val_loss)
                if epoch % 20 == 0 and verbose:
                    print(
                        f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                        f"Min val loss: {np.min(callback_self.val_ls):.4f}"
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
            optimizers={"deeptabular": optimizer} if self.trainer.bayes_opt else None,
            num_workers=16,
            device=self.trainer.device,
        )
        return wd_trainer

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
        from pytorch_widedeep.preprocessing import TabPreprocessor

        cont_feature_names = self.trainer.cont_feature_names
        cat_feature_names = self.trainer.cat_feature_names
        tab_preprocessor = TabPreprocessor(
            continuous_cols=cont_feature_names,
            cat_embed_cols=cat_feature_names if len(cat_feature_names) != 0 else None,
        )
        X_tab_train = tab_preprocessor.fit_transform(X_train)
        X_tab_val = tab_preprocessor.transform(X_val)
        X_tab_test = tab_preprocessor.transform(X_test)
        self.tab_preprocessor = tab_preprocessor
        return (
            X_tab_train,
            D_train,
            y_train,
            X_tab_val,
            D_val,
            y_val,
            X_tab_test,
            D_test,
            y_test,
        )

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
        model.fit(
            X_train={"X_tab": X_train, "target": y_train},
            X_val={"X_tab": X_val, "target": y_val},
            n_epochs=epoch,
            batch_size=int(kwargs["batch_size"]),
        )

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        return model.predict(X_tab=X_test)

    def _data_preprocess(self, df, derived_data, model_name):
        # SettingWithCopyWarning in TabPreprocessor.transform
        # i.e. df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        pd.set_option("mode.chained_assignment", "warn")
        X_df = self.tab_preprocessor.transform(df)
        pd.set_option("mode.chained_assignment", "raise")
        return X_df, derived_data

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


class _WideDeepCallback:
    pass


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
#         tabular_dataset, cont_feature_names, label_name = self.trainer.get_tabular_dataset()
#
#         data_config = DataConfig(
#             target=label_name, continuous_cols=cont_feature_names, num_workers=8
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
#                                 self.trainer.train_indices, :
#                             ],
#                             validation=tabular_dataset.loc[
#                                 self.trainer.val_indices, :
#                             ],
#                             loss=self.trainer.loss_fn,
#                         )
#                     except Exception as e:
#                         exceptions.append(e)
#                         return 1e3
#                     res = tabular_model.evaluate(
#                         tabular_dataset.loc[self.trainer.val_indices, :]
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
#                     train=tabular_dataset.loc[self.trainer.train_indices, :],
#                     validation=tabular_dataset.loc[self.trainer.val_indices, :],
#                     loss=self.trainer.loss_fn,
#                 )
#                 metrics["model"].append(tabular_model.config._model_name)
#                 metrics["mse"].append(
#                     tabular_model.evaluate(
#                         tabular_dataset.loc[self.trainer.test_indices, :]
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
    def __init__(self, trainer=None, program=None, model_subset=None):
        super(TabNet, self).__init__(
            trainer, program=program, model_subset=model_subset
        )
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

    def _get_model_names(self):
        return ["TabNet"]

    def _new_model(self, model_name, verbose, **kwargs):
        from pytorch_tabnet.tab_model import TabNetRegressor

        def extract_params(**kwargs):
            params = {}
            optim_params = {}
            batch_size = 32
            for key, value in kwargs.items():
                if key in [
                    "n_d",
                    "n_a",
                    "n_steps",
                    "gamma",
                    "n_independent",
                    "n_shared",
                ]:
                    params[key] = value
                elif key == "batch_size":
                    batch_size = int(value)
                else:
                    optim_params[key] = value
            return params, optim_params, batch_size

        params, optim_params, batch_size = extract_params(**kwargs)

        model = TabNetRegressor(
            verbose=20 if verbose else 0, optimizer_params=optim_params
        )

        model.set_params(**params)
        return model

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
        cont_feature_names = self.trainer.cont_feature_names

        X_train = X_train[cont_feature_names].values.astype(np.float32)
        X_val = X_val[cont_feature_names].values.astype(np.float32)
        X_test = X_test[cont_feature_names].values.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

        return X_train, D_train, y_train, X_val, D_val, y_val, X_test, D_test, y_test

    def _data_preprocess(self, df, derived_data, model_name):
        return (
            df[self.trainer.cont_feature_names].values.astype(np.float32),
            derived_data,
        )

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
        eval_set = [(X_val, y_val)]

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            max_epochs=epoch,
            patience=self.trainer.bayes_epoch,
            loss_fn=self.trainer.loss_fn,
            eval_metric=[self.trainer.loss],
            batch_size=int(kwargs["batch_size"]),
        )

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        return model.predict(X_test).reshape(-1, 1)

    def _space(self, model_name):
        return [
            Integer(low=4, high=64, prior="uniform", name="n_d", dtype=int),  # 8
            Integer(low=4, high=64, prior="uniform", name="n_a", dtype=int),  # 8
            Integer(low=3, high=10, prior="uniform", name="n_steps", dtype=int),  # 3
            Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
            Integer(
                low=1, high=5, prior="uniform", name="n_independent", dtype=int
            ),  # 2
            Integer(low=1, high=5, prior="uniform", name="n_shared", dtype=int),  # 2
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        return {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "n_independent": 2,
            "n_shared": 2,
            "lr": self.trainer.chosen_params["lr"],
            "weight_decay": self.trainer.chosen_params["weight_decay"],
            "batch_size": self.trainer.chosen_params["batch_size"],
        }


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


class ThisWork(TorchModel):
    def __init__(
        self, trainer=None, manual_activate=None, program=None, model_subset=None
    ):
        super(ThisWork, self).__init__(
            trainer, program=program, model_subset=model_subset
        )
        self.activated_sn = None
        self.manual_activate = manual_activate
        from src.model.sn_formulas import sn_mapping

        if self.manual_activate is not None:
            for sn in self.manual_activate:
                if sn not in sn_mapping.keys():
                    raise Exception(f"SN model {sn} is not implemented or activated.")

    def _get_program_name(self):
        return "ThisWork"

    def _new_model(self, model_name, verbose, **kwargs):
        from src.model.sn_formulas import sn_mapping

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
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["ThisWork"]


class ThisWorkRidge(ThisWork):
    def _get_program_name(self):
        return "ThisWorkRidge"

    def _new_model(self, model_name, verbose, **kwargs):
        from src.model.sn_formulas import sn_mapping

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
            len(self.trainer.cont_feature_names),
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
                cont_feature_names=self.trainer.cont_feature_names,
                cat_feature_names=self.trainer.cat_feature_names,
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
            cont_feature_names=self.trainer.cont_feature_names,
            cat_feature_names=self.trainer.cat_feature_names,
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
    def __init__(self, trainer=None, program=None, layers=None, model_subset=None):
        super(MLP, self).__init__(trainer, program=program, model_subset=model_subset)
        self.layers = layers

    def _get_program_name(self):
        return "MLP"

    def _new_model(self, model_name, verbose, **kwargs):
        set_torch_random(0)
        return NN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["MLP"]


class RFE(TorchModel):
    def __init__(
        self,
        trainer: Trainer,
        torch_model: Type[TorchModel],
        model_subset=None,
        program=None,
        metric: str = "Validation RMSE",
        impor_method: str = "shap",
        cross_validation=5,
        min_features=1,
    ):
        self.metric = metric
        self.impor_method = impor_method
        self.cross_validation = cross_validation
        self.min_features = min_features

        self.internal_trainer = cp(trainer)
        self.torch_model = torch_model
        super(RFE, self).__init__(
            trainer=trainer, program=program, model_subset=model_subset
        )
        self.internal_trainer.project_root = self.root
        self.modelbase = self.torch_model(self.internal_trainer)
        self.internal_trainer.modelbases = []
        self.internal_trainer.modelbases_names = []
        self.internal_trainer.add_modelbases([self.modelbase])
        self.metrics = []
        self.features_eliminated = []
        self.selected_features = []
        self.impor_dicts = []

    def _get_program_name(self):
        return "RFE-" + self.torch_model.__name__

    def _get_model_names(self):
        return self.modelbase.get_model_names()

    def _new_model(self, model_name, verbose, **kwargs):
        return self.modelbase._new_model(model_name, verbose, **kwargs)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.modelbase._predict(df, model_name, derived_data, **kwargs)

    def _predict_all(self, **kwargs):
        return self.modelbase._predict_all(**kwargs)

    def _train(
        self,
        verbose: bool = True,
        model_subset: list = None,
        warm_start=False,
        **kwargs,
    ):
        if warm_start:
            self.modelbase._train(
                warm_start=warm_start,
                model_subset=model_subset,
                verbose=verbose,
                **kwargs,
            )
        else:
            self.run(verbose=verbose, model_subset=model_subset)
            self.modelbase._train(
                warm_start=warm_start,
                model_subset=model_subset,
                verbose=verbose,
                **kwargs,
            )

    def run(self, verbose=True, model_subset=None):
        rest_features = cp(self.trainer.all_feature_names)
        while len(rest_features) > self.min_features:
            if verbose:
                print(f"Using features: {rest_features}")
            self.internal_trainer.set_feature_names(rest_features)
            if self.cross_validation == 0:
                self.modelbase._train(
                    verbose=False, model_subset=model_subset, dump_trainer=False
                )
            leaderboard = self.internal_trainer.get_leaderboard(
                test_data_only=False,
                cross_validation=self.cross_validation,
                verbose=False,
                dump_trainer=False,
            )
            self.metrics.append(leaderboard.loc[0, self.metric])
            importance, names = self.internal_trainer.cal_feature_importance(
                program=self.modelbase.program,
                model_name=self.modelbase.get_model_names()[0],
                method=self.impor_method,
            )
            impor_dict = {"feature": [], "attr": []}
            for imp, name in zip(importance, names):
                if name in rest_features:
                    impor_dict["feature"].append(name)
                    impor_dict["attr"].append(imp)
            df = pd.DataFrame(impor_dict)
            df.sort_values(by="attr", inplace=True, ascending=False)
            df.reset_index(drop=True, inplace=True)
            rest_features = list(df["feature"])
            print(rest_features)
            self.features_eliminated.append(rest_features.pop(-1))
            self.impor_dicts.append(df)
            if verbose:
                print(f"Eliminated feature: {self.features_eliminated[-1]}")
                # print(f"Permutation importance:\n{df}")

        select_idx = self.metrics.index(np.min(self.metrics))
        self.selected_features = self.features_eliminated[select_idx:]
        self.internal_trainer.set_feature_names(self.selected_features)
        if verbose:
            print(f"Selected features: {self.selected_features}")
            print(f"Eliminated features: {self.features_eliminated[:select_idx]}")


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
            model_names += submodel.get_model_names()
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
            if model_name in submodel.get_model_names():
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
