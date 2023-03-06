from src.utils import *
from src.model import AbstractModel, TorchModel
from src.trainer import Trainer, save_trainer
from .nn_models import *
from copy import deepcopy as cp
from skopt.space import Real, Integer, Categorical


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
        if self.activated_sn is None:
            self.activated_sn = self._get_activated_sn()
        set_torch_random(0)
        return ThisWorkNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_activated_sn(self):
        from src.model.sn_formulas import sn_mapping

        activated_sn = []
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, type in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][type] == "Material"
        ]
        for key, sn in sn_mapping.items():
            if sn.test_sn_vars(
                self.trainer.cont_feature_names,
                list(self.trainer.derived_data.keys()),
            ) and (self.manual_activate is None or key in self.manual_activate):
                activated_sn.append(
                    sn(
                        cont_feature_names=self.trainer.cont_feature_names,
                        derived_feature_names=list(self.trainer.derived_data.keys()),
                        s_zero_slip=self.trainer.get_zero_slip(sn.get_sn_vars()[0]),
                        sn_coeff_vars_idx=sn_coeff_vars_idx,
                    )
                )
        print(f"Activated SN models: {[sn.__class__.__name__ for sn in activated_sn]}")
        return activated_sn

    def _get_model_names(self):
        return ["ThisWork"]


class ThisWorkRidge(ThisWork):
    def _get_program_name(self):
        return "ThisWorkRidge"

    def _new_model(self, model_name, verbose, **kwargs):
        if self.activated_sn is None:
            self.activated_sn = self._get_activated_sn()
        set_torch_random(0)
        return ThisWorkRidgeNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers,
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _train_step(self, model, train_loader, optimizer):
        model.train()
        avg_loss = 0
        for idx, tensors in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(*([data] + additional_tensors))
            self.ridge(model, yhat)
            loss = self._loss_fn(yhat, y, model)
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
        return MLPNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
        ).to(self.trainer.device)

    def _get_model_names(self):
        return ["MLP"]


class CatEmbedLSTM(TorchModel):
    def __init__(
        self,
        trainer=None,
        manual_activate_sn=None,
        program=None,
        layers=None,
        model_subset=None,
    ):
        super(CatEmbedLSTM, self).__init__(
            trainer, program=program, model_subset=model_subset
        )
        self.manual_activate_sn = manual_activate_sn
        self.layers = layers

    def _get_program_name(self):
        return "CatEmbedLSTM"

    def _new_model(self, model_name, verbose, **kwargs):
        set_torch_random(0)
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, type in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][type] == "Material"
        ]
        return CatEmbedLSTMNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
            manual_activate_sn=self.manual_activate_sn,
            sn_coeff_vars_idx=sn_coeff_vars_idx,
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            lstm_embedding_dim=kwargs["lstm_embedding_dim"],
            cat_embedding_dim=kwargs["cat_embedding_dim"],
            n_hidden=kwargs["n_hidden"],
        ).to(self.trainer.device)

    def _get_optimizer(self, model, warm_start, **kwargs):
        # return torch.optim.Adam(
        #     model.parameters(),
        #     lr=kwargs["lr"] / 10 if warm_start else kwargs["lr"],
        #     weight_decay=kwargs["weight_decay"],
        # )
        return torch.optim.SGD(
            model.parameters(),
            lr=kwargs["lr"] / 10 if warm_start else kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )

    def _get_model_names(self):
        return ["CatEmbedLSTM"]

    def _space(self, model_name):
        return [
            Integer(
                low=3, high=100, prior="uniform", name="lstm_embedding_dim", dtype=int
            ),
            Integer(
                low=3, high=100, prior="uniform", name="cat_embedding_dim", dtype=int
            ),
            Integer(low=3, high=10, prior="uniform", name="n_hidden", dtype=int),
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        return {
            "lstm_embedding_dim": 10,
            "cat_embedding_dim": 10,
            "n_hidden": 3,
            "lr": self.trainer.chosen_params["lr"],
            "weight_decay": self.trainer.chosen_params["weight_decay"],
            "batch_size": self.trainer.chosen_params["batch_size"],
        }


class RFE(TorchModel):
    def __init__(
        self,
        trainer: Trainer,
        modelbase: AbstractModel,
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

        internal_trainer = cp(trainer)
        internal_trainer.modelbases = []
        internal_trainer.modelbases_names = []
        self._model_names = modelbase.get_model_names()
        self.model_class = modelbase.__class__
        super(RFE, self).__init__(
            trainer=trainer, program=program, model_subset=model_subset
        )
        self.trainer_modelbase = {}

        internal_trainer.project_root = self.root

        for model_name in self.get_model_names():
            tmp_trainer = cp(internal_trainer)
            modelbase = self.model_class(tmp_trainer, model_subset=[model_name])
            tmp_trainer.add_modelbases([modelbase])
            self.trainer_modelbase[model_name] = (tmp_trainer, modelbase)
        self.metrics = {}
        self.features_eliminated = {}
        self.selected_features = {}
        self.impor_dicts = {}

    def _get_program_name(self):
        return "RFE-" + self.model_class.__name__

    def _get_model_names(self):
        return self._model_names

    def _new_model(self, model_name, verbose, **kwargs):
        return self.trainer_modelbase[model_name][1]._new_model(
            model_name, verbose, **kwargs
        )

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.trainer_modelbase[model_name][1]._predict(
            df, model_name, derived_data, **kwargs
        )

    def _predict_all(self, **kwargs):
        predictions = {}
        for name, (trainer, modelbase) in self.trainer_modelbase.items():
            predictions[name] = modelbase._predict_all(**kwargs)[name]
        return predictions

    def _train(
        self,
        verbose: bool = True,
        model_subset: list = None,
        warm_start=False,
        **kwargs,
    ):
        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if warm_start:
                self.trainer_modelbase[model_name][1]._train(
                    warm_start=warm_start,
                    model_subset=[model_name],
                    verbose=verbose,
                    **kwargs,
                )
            else:
                self.run(verbose=verbose, model_name=model_name)
                self.trainer_modelbase[model_name][1]._train(
                    warm_start=warm_start,
                    model_subset=[model_name],
                    verbose=verbose,
                    **kwargs,
                )

    def run(self, model_name, verbose=True):
        rest_features = cp(self.trainer.all_feature_names)
        trainer, modelbase = self.trainer_modelbase[model_name]
        metrics = []
        features_eliminated = []
        impor_dicts = []
        while len(rest_features) > self.min_features:
            if verbose:
                print(f"Using features: {rest_features}")
            trainer.set_feature_names(rest_features)
            if self.cross_validation == 0:
                modelbase._train(
                    verbose=False, model_subset=[model_name], dump_trainer=False
                )
            leaderboard = trainer.get_leaderboard(
                test_data_only=False,
                cross_validation=self.cross_validation,
                verbose=False,
                dump_trainer=False,
            )
            metrics.append(leaderboard.loc[0, self.metric])
            importance, names = trainer.cal_feature_importance(
                program=modelbase.program,
                model_name=model_name,
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
            features_eliminated.append(rest_features.pop(-1))
            impor_dicts.append(df)
            if verbose:
                print(f"Eliminated feature: {features_eliminated[-1]}")
                # print(f"Permutation importance:\n{df}")

        select_idx = metrics.index(np.min(metrics))
        selected_features = features_eliminated[select_idx:]
        trainer.set_feature_names(selected_features)
        self.metrics[model_name] = metrics
        self.features_eliminated[model_name] = features_eliminated
        self.impor_dicts[model_name] = impor_dicts
        self.selected_features[model_name] = selected_features
        if verbose:
            print(f"Selected features: {selected_features}")
            print(f"Eliminated features: {features_eliminated[:select_idx]}")


class ModelAssembly(AbstractModel):
    def __init__(self, trainer, models=None, program=None, model_subset=None):
        self.program = "ModelAssembly" if program is None else program
        super(ModelAssembly, self).__init__(
            trainer, program=self.program, model_subset=model_subset
        )
        self.models = {}
        if models is None:
            if model_subset is None:
                raise Exception(f"One of models and model_subset should be specified.")
            else:
                for model_name in model_subset:
                    self.models[model_name] = getattr(
                        sys.modules[__name__], model_name
                    )(trainer, model_subset=[model_name])
        else:
            for model in models:
                if len(model.get_model_names()) > 1:
                    raise Exception(
                        f"ModelAssembly is designed for modelbases with a single model."
                    )
                self.models[model.get_model_names()[0]] = model

    def _get_program_name(self):
        return self.program

    def fit(self, model_subset=None, **kwargs):
        for model_name in self.models.keys() if model_subset is None else model_subset:
            self.models[model_name].fit(**kwargs)

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        return self.models[model_name].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def train(
        self,
        *args,
        **kwargs,
    ):
        print(f"\n-------------Run {self.program}-------------\n")
        self._train(*args, **kwargs)
        print(f"\n-------------{self.program} End-------------\n")

    def _train(self, model_subset=None, *args, **kwargs):
        for model_name in self.models.keys() if model_subset is None else model_subset:
            self.models[model_name]._train(*args, **kwargs)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.models[model_name].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def _predict_all(self, **kwargs):
        self._check_train_status()
        predictions = {}
        for submodel in self.models.values():
            sub_predictions = submodel._predict_all(**kwargs)
            for key, value in sub_predictions.items():
                predictions[key] = value
        return predictions

    def _get_model_names(self):
        return list(self.models.keys())

    def _check_train_status(self):
        for submodel in self.models.values():
            try:
                submodel._check_train_status()
            except:
                raise Exception(
                    f"{self.program} not trained, run {self.__class__.__name__}.train() first."
                )
