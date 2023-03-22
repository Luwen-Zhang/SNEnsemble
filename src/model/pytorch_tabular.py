from src.utils import *
from src.model import AbstractModel
from skopt.space import Integer, Categorical, Real
import src


class PytorchTabular(AbstractModel):
    def _get_program_name(self):
        return "PytorchTabular"

    def _new_model(self, model_name, verbose, **kwargs):
        warnings.filterwarnings("ignore", message="Wandb")
        from functools import partialmethod
        from pytorch_tabular.config import ExperimentRunManager

        ExperimentRunManager.__init__ = partialmethod(
            ExperimentRunManager.__init__,
            exp_version_manager=self.root + "exp_version_manager.yml",
        )
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import (
            CategoryEmbeddingModelConfig,
            NodeConfig,
            TabNetModelConfig,
            TabTransformerConfig,
            AutoIntConfig,
            FTTransformerConfig,
            GatedAdditiveTreeEnsembleConfig,
        )
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        data_config = DataConfig(
            target=self.trainer.label_name,
            continuous_cols=self.trainer.cont_feature_names,
            categorical_cols=self.trainer.cat_feature_names,
            num_workers=6,
        )
        if not os.path.exists(os.path.join(self.root, "ckpts")):
            os.mkdir(os.path.join(self.root, "ckpts"))
        trainer_config = TrainerConfig(
            progress_bar="none",
            early_stopping="valid_mean_squared_error",
            early_stopping_patience=self.trainer.static_params["patience"],
            checkpoints="valid_mean_squared_error",
            checkpoints_path=os.path.join(self.root, "ckpts"),
            checkpoints_save_top_k=1,
            checkpoints_name=model_name,
            load_best=True,
            accelerator="cpu" if self.device == "cpu" else "auto",
        )
        optimizer_config = OptimizerConfig()

        model_configs = {
            "Category Embedding": CategoryEmbeddingModelConfig,
            "NODE": NodeConfig,
            "TabNet": TabNetModelConfig,
            "TabTransformer": TabTransformerConfig,
            "AutoInt": AutoIntConfig,
            "FTTransformer": FTTransformerConfig,
            "GATE": GatedAdditiveTreeEnsembleConfig,
        }
        with HiddenPrints():
            model_config = (
                model_configs[model_name](task="regression", **kwargs)
                if model_name != "NODE"
                else model_configs[model_name](
                    task="regression", embed_categorical=True, **kwargs
                )
            )
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )
        tabular_model.config["progress_bar_refresh_rate"] = 0
        return tabular_model

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
        all_feature_names = self.trainer.all_feature_names
        X_train = self.trainer.categories_inverse_transform(X_train[all_feature_names])
        X_val = self.trainer.categories_inverse_transform(X_val[all_feature_names])
        X_test = self.trainer.categories_inverse_transform(X_test[all_feature_names])
        return X_train, D_train, y_train, X_val, D_val, y_val, X_test, D_test, y_test

    def _data_preprocess(self, df, derived_data, model_name):
        all_feature_names = self.trainer.all_feature_names
        df = self.trainer.categories_inverse_transform(df[all_feature_names].copy())
        return df, derived_data

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
        tc = TqdmController()
        tc.disable_tqdm()
        warnings.simplefilter(action="ignore", category=UserWarning)
        label_name = self.trainer.label_name
        train_data = X_train.copy()
        train_data[label_name[0]] = y_train
        val_data = X_val.copy()
        val_data[label_name[0]] = y_val
        with HiddenPrints(
            disable_std=not verbose,
            disable_logging=not verbose,
        ):
            model.fit(
                train=train_data,
                validation=val_data,
                loss=self.trainer.loss_fn,
                max_epochs=epoch,
                callbacks=[PytorchTabularCallback(verbose=verbose, total_epoch=epoch)],
            )
        warnings.simplefilter(action="default", category=UserWarning)
        tc.enable_tqdm()

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        target = model.config.target[0]
        with HiddenPrints():
            # Two annoying warnings that cannot be suppressed:
            # 1. DeprecationWarning: Default for `include_input_features` will change from True to False in the next
            # release. Please set it explicitly.
            # 2. DeprecationWarning: "The `out_ff_layers`, `out_ff_activation`, `out_ff_dropoout`, and
            # `out_ff_initialization` arguments are deprecated and will be removed next release. Please use head and
            # head_config as an alternative.
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module="pytorch_tabular"
            )
            res = np.array(
                model.predict(X_test, include_input_features=False)[
                    f"{target}_prediction"
                ]
            ).reshape(-1, 1)
        return res

    def _get_model_names(self):
        return [
            "Category Embedding",
            "NODE",
            "TabNet",
            "TabTransformer",
            "AutoInt",
            "FTTransformer",
            "GATE",
        ]

    def _space(self, model_name):
        space_dict = {
            "Category Embedding": [
                Real(low=0, high=0.5, prior="uniform", name="dropout"),  # 0.5
                Real(low=0, high=0.5, prior="uniform", name="embedding_dropout"),  # 0.5
                Real(
                    low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                ),  # 0.001
            ],
            "NODE": [
                Integer(low=2, high=6, prior="uniform", name="depth", dtype=int),  # 6
                Real(low=0, high=0.3, prior="uniform", name="embedding_dropout"),  # 0.0
                Real(low=0, high=0.3, prior="uniform", name="input_dropout"),  # 0.0
                Real(
                    low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                ),  # 0.001
                Integer(
                    low=128, high=512, prior="uniform", name="num_trees", dtype=int
                ),
            ],
            "TabNet": [
                Integer(low=4, high=64, prior="uniform", name="n_d", dtype=int),  # 8
                Integer(low=4, high=64, prior="uniform", name="n_a", dtype=int),  # 8
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
            "TabTransformer": [
                Real(low=0, high=0.3, prior="uniform", name="embedding_dropout"),  # 0.1
                Real(low=0, high=0.3, prior="uniform", name="ff_dropout"),  # 0.1
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
                Real(low=0, high=0.3, prior="uniform", name="attn_dropout"),  # 0.1
                Real(low=0, high=0.3, prior="uniform", name="add_norm_dropout"),  # 0.1
                Integer(
                    low=2,
                    high=6,
                    prior="uniform",
                    name="ff_hidden_multiplier",
                    dtype=int,
                ),  # 4
                Real(
                    low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                ),  # 0.001
            ],
            "AutoInt": [
                Real(
                    low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"
                ),  # 0.001
                Real(low=0, high=0.3, prior="uniform", name="attn_dropouts"),  # 0.0
                # Categorical([16, 32, 64, 128], name='attn_embed_dim'),  # 32
                Real(low=0, high=0.3, prior="uniform", name="dropout"),  # 0.0
                # Categorical([8, 16, 32, 64], name='embedding_dim'),  # 16
                Real(low=0, high=0.3, prior="uniform", name="embedding_dropout"),  # 0.0
                Integer(
                    low=1,
                    high=4,
                    prior="uniform",
                    name="num_attn_blocks",
                    dtype=int,
                ),  # 3
                # Integer(low=1, high=4, prior='uniform', name='num_heads', dtype=int),  # 2
            ],
            "FTTransformer": [
                Real(low=0, high=0.3, prior="uniform", name="embedding_dropout"),  # 0.1
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
                Real(low=0, high=0.3, prior="uniform", name="attn_dropout"),  # 0.1
                Real(low=0, high=0.3, prior="uniform", name="add_norm_dropout"),  # 0.1
                Real(low=0, high=0.3, prior="uniform", name="ff_dropout"),  # 0.1
                Integer(
                    low=2,
                    high=6,
                    prior="uniform",
                    name="ff_hidden_multiplier",
                    dtype=int,
                ),  # 4
            ],
            "GATE": [
                Integer(low=2, high=10, prior="uniform", name="gflu_stages", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="gflu_dropout"),
                Integer(low=2, high=4, prior="uniform", name="tree_depth", dtype=int),
                Integer(low=10, high=20, prior="uniform", name="num_trees", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="tree_dropout"),
                Real(
                    low=0.0,
                    high=0.3,
                    prior="uniform",
                    name="tree_wise_attention_dropout",
                ),
                Real(low=0.0, high=0.3, prior="uniform", name="embedding_dropout"),
                Real(low=1e-5, high=0.1, prior="log-uniform", name="learning_rate"),
            ],
        }
        return space_dict[model_name]

    def _initial_values(self, model_name):
        params_dict = {
            "Category Embedding": {
                "dropout": 0.5,
                "embedding_dropout": 0.5,
                "learning_rate": 0.001,
            },
            "NODE": {
                "depth": 6,
                "embedding_dropout": 0.0,
                "input_dropout": 0.0,
                "learning_rate": 0.001,
                "num_trees": 256,
            },
            "TabNet": {
                "n_d": 8,
                "n_a": 8,
                "n_steps": 3,
                "gamma": 1.3,
                "n_independent": 2,
                "n_shared": 2,
                "learning_rate": 0.001,
            },
            "TabTransformer": {
                "embedding_dropout": 0.1,
                "ff_dropout": 0.1,
                "shared_embedding_fraction": 0.25,
                "num_attn_blocks": 6,
                "attn_dropout": 0.1,
                "add_norm_dropout": 0.1,
                "ff_hidden_multiplier": 4,
                "learning_rate": 0.001,
            },
            "AutoInt": {
                "learning_rate": 0.001,
                "attn_dropouts": 0.0,
                "dropout": 0.0,
                "embedding_dropout": 0.0,
                "num_attn_blocks": 3,
            },
            "FTTransformer": {
                "embedding_dropout": 0.1,
                "shared_embedding_fraction": 0.25,
                "num_attn_blocks": 6,
                "attn_dropout": 0.1,
                "add_norm_dropout": 0.1,
                "ff_dropout": 0.1,
                "ff_hidden_multiplier": 4,
            },
            "GATE": {
                "gflu_stages": 6,
                "gflu_dropout": 0.0,
                # `tree_depth` influences the memory usage a lot. `tree_depth`==10 with other default settings consumes
                # about 4 GiBs of ram.
                # When "tree_depth" larger than 4, and num_trees larger than 20 (approximately), performance on GPU
                # decreases dramatically.
                "tree_depth": 4,
                "num_trees": 20,
                "tree_dropout": 0.0,
                "tree_wise_attention_dropout": 0.0,
                "embedding_dropout": 0.1,
                "learning_rate": 1e-3,
            },
        }
        return params_dict[model_name]


import pytorch_lightning as pl
from pytorch_lightning import Callback


class PytorchTabularCallback(Callback):
    def __init__(self, verbose, total_epoch):
        super(PytorchTabularCallback, self).__init__()
        self.val_ls = []
        self.verbose = verbose
        self.total_epoch = total_epoch

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        train_loss = logs["train_mean_squared_error"].detach().cpu().numpy()
        val_loss = logs["valid_mean_squared_error"].detach().cpu().numpy()
        self.val_ls.append(val_loss)
        epoch = trainer.current_epoch
        if epoch % src.setting["verbose_per_epoch"] == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )
