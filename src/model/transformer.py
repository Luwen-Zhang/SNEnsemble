import src
from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel, AbstractNN
from ._transformer.models import *


class Transformer(TorchModel):
    def __init__(
        self,
        trainer,
        manual_activate_sn=None,
        layers=None,
        *args,
        **kwargs,
    ):
        super(Transformer, self).__init__(trainer, *args, **kwargs)
        self.manual_activate_sn = manual_activate_sn
        self.layers = layers

    def _get_program_name(self):
        return "Transformer"

    def _get_model_names(self):
        return [
            "FTTransformer",
            "TransformerLSTM",
            "TransformerSeq",
            "SNTransformer",
            "SNTransformerSeq",
            "SNTransformerAug",
            "SNTransformerLR",
            "SNTransformerLRKMeans",
        ]

    def _new_model(self, model_name, verbose, **kwargs):
        if model_name == "TransformerLSTM":
            return TransformerLSTMNN(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                layers=self.trainer.args["layers"]
                if self.layers is None
                else self.layers,
                trainer=self.trainer,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                seq_embedding_dim=kwargs["seq_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                n_hidden=kwargs["n_hidden"],
                lstm_layers=kwargs["lstm_layers"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                embed_dropout=kwargs["embed_dropout"],
                attn_dropout=kwargs["attn_dropout"],
            )
        elif model_name in [
            "FTTransformer",
            "SNTransformer",
            "SNTransformerAug",
            "SNTransformerLR",
            "SNTransformerLRKMeans",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                layers=self.trainer.args["layers"]
                if self.layers is None
                else self.layers,
                trainer=self.trainer,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                embedding_dim=kwargs["embedding_dim"],
                embed_dropout=kwargs["embed_dropout"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                attn_dropout=kwargs["attn_dropout"],
            )
        elif model_name in [
            "TransformerSeq",
            "SNTransformerSeq",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                layers=self.trainer.args["layers"]
                if self.layers is None
                else self.layers,
                trainer=self.trainer,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                seq_embedding_dim=kwargs["seq_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                embed_dropout=kwargs["embed_dropout"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                attn_dropout=kwargs["attn_dropout"],
                seq_attn_layers=kwargs["seq_attn_layers"],
                seq_attn_heads=kwargs["seq_attn_heads"],
                seq_attn_dropout=kwargs["seq_attn_dropout"],
            )

    def _space(self, model_name):
        if model_name == "TransformerLSTM":
            return [
                Categorical(categories=[2, 4, 8, 16, 32], name="seq_embedding_dim"),
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=1, high=30, prior="uniform", name="n_hidden", dtype=int),
                Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
                Integer(low=2, high=6, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "FTTransformer",
            "SNTransformer",
            "SNTransformerAug",
            "SNTransformerLR",
            "SNTransformerLRKMeans",
        ]:
            return [
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=6, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "TransformerSeq",
            "SNTransformerSeq",
        ]:
            return [
                # ``seq_embedding_dim`` should be able to divided by ``attn_heads``.
                Categorical(categories=[8, 16, 32], name="seq_embedding_dim"),
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=6, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
                Integer(
                    low=2, high=4, prior="uniform", name="seq_attn_layers", dtype=int
                ),
                Categorical(categories=[2, 4, 8], name="seq_attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="seq_attn_dropout"),
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        if model_name == "TransformerLSTM":
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "n_hidden": 10,
                "lstm_layers": 1,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in [
            "FTTransformer",
            "SNTransformer",
            "SNTransformerAug",
            "SNTransformerLR",
            "SNTransformerLRKMeans",
        ]:
            return {
                "embedding_dim": 32,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in [
            "TransformerSeq",
            "SNTransformerSeq",
        ]:
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "seq_attn_layers": 4,
                "seq_attn_heads": 8,
                "seq_attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }

    def _conditional_validity(self, model_name: str) -> bool:
        if (
            model_name
            in [
                "SNTransformerSeq",
                "SNTransformer",
                "SNTransformerAug",
                "SNTransformerLR",
                "SNTransformerLRKMeans",
            ]
            and "Relative Mean Stress" not in self.trainer.cont_feature_names
        ):
            return False
        return True

    def _bayes_eval(
        self,
        model,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
    ):
        """
        Evaluating the model for bayesian optimization iterations. If MaterialCycleSplitter or CycleSplitter
        is used, the average training and evaluation error is returned. Otherwise, evaluation error is returned.

        Returns
        -------
        result
            The evaluation of bayesian hyperparameter optimization.
        """
        y_val_pred = self._pred_single_model(model, X_val, D_val, verbose=False)
        res = metric_sklearn(y_val_pred, y_val, self.trainer.args["loss"])
        if self.trainer.args["data_splitter"] in ["CycleSplitter"]:
            y_train_pred = self._pred_single_model(
                model, X_train, D_train, verbose=False
            )
            res += metric_sklearn(y_train_pred, y_train, self.trainer.args["loss"])
            res /= 2
        return res

    def _early_stopping_eval(self, train_loss: float, val_loss: float) -> float:
        """
        Calculate the loss value (criteria) for early stopping. By default, if MaterialCycleSplitter or CycleSplitter
        is used, the average of ``train_loss`` and ``val_loss`` is returned. Otherwise, ``val_loss`` is returned.

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
        if self.trainer.args["data_splitter"] in ["CycleSplitter"]:
            return 0.5 * (train_loss + val_loss)
        else:
            return val_loss
