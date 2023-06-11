import warnings
from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel
from ._transformer.models_clustering import *
from ._transformer.models_with_seq import *
from ._transformer.models_basic import *


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
            "SNTransformerLRKMeans",
            "CategoryEmbedding",
            "CatEmbedSeq",
            "SNCatEmbedLRKMeansSeq",
            "SNCatEmbedLRGMM",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLRBMM",
            "SNCatEmbedLRPCABMM",
            "SNCatEmbedLR2LBMM",
            "SNCatEmbedLR2LPCABMM",
            "SNCatEmbedLRKMeans",
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LPCAKMeans",
        ]

    def _new_model(self, model_name, verbose, **kwargs):
        fix_kwargs = dict(
            n_inputs=len(self.trainer.cont_feature_names),
            n_outputs=len(self.trainer.label_name),
            layers=self.trainer.args["layers"] if self.layers is None else self.layers,
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            trainer=self.trainer,
        )
        if model_name == "TransformerLSTM":
            return TransformerLSTMNN(
                **fix_kwargs,
                attn_ff_dim=256,
                **kwargs,
            )
        elif model_name in [
            "FTTransformer",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                **kwargs,
            )
        elif model_name in [
            "TransformerSeq",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                attn_ff_dim=256,
                **kwargs,
            )
        elif model_name in ["CategoryEmbedding"]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                embedding_dim=3,
                **kwargs,
            )
        elif model_name in ["CatEmbedSeq"]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                attn_ff_dim=256,
                **kwargs,
            )
        elif model_name in [
            "SNCatEmbedLRKMeans",
            "SNCatEmbedLRGMM",
            "SNCatEmbedLRBMM",
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LBMM",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                embedding_dim=3,
                **kwargs,
            )
        elif model_name in [
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLRPCABMM",
            "SNCatEmbedLR2LPCAKMeans",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLR2LPCABMM",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name.replace('PCA', '')}NN")
            if "2L" not in model_name:
                feature_idx = cls.basic_clustering_features_idx(self.trainer)
            else:
                feature_idx = cls.top_clustering_features_idx(self.trainer)
            if len(feature_idx) > 1:
                pca = self.trainer.datamodule.pca(feature_idx=feature_idx)
                n_pca_dim = (
                    np.where(pca.explained_variance_ratio_.cumsum() < 0.9)[0][-1] + 1
                )
            else:
                n_pca_dim = 1
            return cls(
                **fix_kwargs,
                embedding_dim=3,
                n_pca_dim=n_pca_dim,
                **kwargs,
            )
        elif model_name in ["SNCatEmbedLRKMeansSeq"]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                attn_ff_dim=256,
                **kwargs,
            )
        elif model_name in [
            "SNTransformerLRKMeans",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name}NN")
            return cls(
                **fix_kwargs,
                **kwargs,
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
        ]:
            return [
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=6, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "SNTransformerLRKMeans",
        ]:
            return [
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=16, prior="uniform", name="n_clusters", dtype=int),
                Integer(low=2, high=6, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "TransformerSeq",
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
        elif model_name in ["CategoryEmbedding"]:
            return [
                # Integer(
                #     low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                # ),
                Real(low=0.0, high=0.5, prior="uniform", name="mlp_dropout"),
                Real(low=0.0, high=0.5, prior="uniform", name="embed_dropout"),
            ] + self.trainer.SPACE
        elif model_name in ["CatEmbedSeq"]:
            return [
                Integer(
                    low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                ),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="mlp_dropout"),
                Categorical(categories=[16, 32, 64], name="seq_embedding_dim"),
                Integer(
                    low=2, high=16, prior="uniform", name="seq_attn_layers", dtype=int
                ),
                Categorical(categories=[2, 4, 8, 16], name="seq_attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="seq_attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "SNCatEmbedLRKMeans",
            "SNCatEmbedLRGMM",
            "SNCatEmbedLRBMM",
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LBMM",
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLRPCABMM",
            "SNCatEmbedLR2LPCAKMeans",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLR2LPCABMM",
        ]:
            return [
                # Integer(
                #     low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                # ),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Integer(low=2, high=64, prior="uniform", name="n_clusters", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="mlp_dropout"),
            ] + self.trainer.SPACE
        elif model_name in ["SNCatEmbedLRKMeansSeq"]:
            return [
                Integer(
                    low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                ),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Integer(low=2, high=16, prior="uniform", name="n_clusters", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="mlp_dropout"),
                Categorical(categories=[16, 32, 64], name="seq_embedding_dim"),
                Integer(
                    low=2, high=16, prior="uniform", name="seq_attn_layers", dtype=int
                ),
                Categorical(categories=[2, 4, 8, 16], name="seq_attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="seq_attn_dropout"),
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        res = {}
        if model_name == "TransformerLSTM":
            res = {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "n_hidden": 10,
                "lstm_layers": 1,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.1,
            }
        elif model_name in [
            "FTTransformer",
        ]:
            res = {
                "embedding_dim": 32,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.1,
            }
        elif model_name in [
            "SNTransformerLRKMeans",
        ]:
            res = {
                "embedding_dim": 32,
                "n_clusters": 5,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.1,
            }
        elif model_name in [
            "TransformerSeq",
        ]:
            res = {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "attn_layers": 6,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.1,
                "seq_attn_layers": 4,
                "seq_attn_heads": 8,
                "seq_attn_dropout": 0.1,
            }
        elif model_name in [
            "CategoryEmbedding",
        ]:
            res = {
                # "embedding_dim": 3,
                "mlp_dropout": 0.0,
                "embed_dropout": 0.1,
            }
        elif model_name in [
            "CatEmbedSeq",
        ]:
            res = {
                "embedding_dim": 3,
                "embed_dropout": 0.1,
                "mlp_dropout": 0.0,
                "seq_embedding_dim": 16,
                "seq_attn_layers": 6,
                "seq_attn_heads": 8,
                "seq_attn_dropout": 0.1,
            }
        elif model_name in [
            "SNCatEmbedLRKMeans",
            "SNCatEmbedLRGMM",
            "SNCatEmbedLRBMM",
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LBMM",
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLRPCABMM",
            "SNCatEmbedLR2LPCAKMeans",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLR2LPCABMM",
        ]:
            res = {
                # "embedding_dim": 3,
                "embed_dropout": 0.1,
                "n_clusters": 16,
                "mlp_dropout": 0.0,
            }
        elif model_name in [
            "SNCatEmbedLRKMeansSeq",
        ]:
            res = {
                "embedding_dim": 3,
                "embed_dropout": 0.1,
                "n_clusters": 5,
                "mlp_dropout": 0.0,
                "seq_embedding_dim": 16,
                "seq_attn_layers": 6,
                "seq_attn_heads": 8,
                "seq_attn_dropout": 0.1,
            }
        res.update(self.trainer.chosen_params)
        return res

    def _conditional_validity(self, model_name: str) -> bool:
        if (
            model_name in ["SNTransformerLRKMeans"]
            and "Relative Mean Stress" not in self.trainer.cont_feature_names
        ):
            return False
        return True

    # def _bayes_eval(
    #     self,
    #     model,
    #     X_train,
    #     D_train,
    #     y_train,
    #     X_val,
    #     D_val,
    #     y_val,
    # ):
    #     """
    #     Evaluating the model for bayesian optimization iterations. If MaterialCycleSplitter or CycleSplitter
    #     is used, the average training and evaluation error is returned. Otherwise, evaluation error is returned.
    #
    #     Returns
    #     -------
    #     result
    #         The evaluation of bayesian hyperparameter optimization.
    #     """
    #     y_val_pred = self._pred_single_model(model, X_val, D_val, verbose=False)
    #     res = metric_sklearn(y_val_pred, y_val, self.trainer.args["loss"])
    #     if self.trainer.args["data_splitter"] in ["CycleSplitter"]:
    #         y_train_pred = self._pred_single_model(
    #             model, X_train, D_train, verbose=False
    #         )
    #         res += metric_sklearn(y_train_pred, y_train, self.trainer.args["loss"])
    #         res /= 2
    #     return res

    # def _early_stopping_eval(self, train_loss: float, val_loss: float) -> float:
    #     """
    #     Calculate the loss value (criteria) for early stopping. By default, if MaterialCycleSplitter or CycleSplitter
    #     is used, the average of ``train_loss`` and ``val_loss`` is returned. Otherwise, ``val_loss`` is returned.
    #
    #     Parameters
    #     ----------
    #     train_loss
    #         Training loss at the epoch.
    #     val_loss
    #         Validation loss at the epoch.
    #
    #     Returns
    #     -------
    #     result
    #         The early stopping evaluation.
    #     """
    #     if self.trainer.args["data_splitter"] in ["CycleSplitter"]:
    #         return 0.5 * (train_loss + val_loss)
    #     else:
    #         return val_loss
