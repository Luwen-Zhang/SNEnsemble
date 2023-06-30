import warnings
from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel
from ._transformer.models_clustering import *
from ._transformer.models_with_seq import *
from ._transformer.models_basic import *


class Transformer(TorchModel):
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
            "SNCategoryEmbedLR2LPCAKMeans",
            "SNCategoryEmbedLR2LPCAGMM",
            "SNCategoryEmbedLR2LPCABMM",
            "SNFTTransLR2LPCAKMeans",
            "SNFTTransLR2LPCAGMM",
            "SNFTTransLR2LPCABMM",
            "SNPyFTTransLRPCAKMeans",
            "SNPyFTTransLRPCAGMM",
            "SNPyFTTransLRPCABMM",
            "SNPyFTTransWrapLRPCAKMeans",
            "SNPyFTTransWrapLRPCAGMM",
            "SNPyFTTransWrapLRPCABMM",
            "SNPyFTTransLR2LPCAKMeans",
            "SNPyFTTransLR2LPCAGMM",
            "SNPyFTTransLR2LPCABMM",
            "SNPyFTTransWrapLR2LPCAKMeans",
            "SNPyFTTransWrapLR2LPCAGMM",
            "SNPyFTTransWrapLR2LPCABMM",
            "SNCategoryEmbedLRPCAKMeans",
            "SNCategoryEmbedLRPCAGMM",
            "SNCategoryEmbedLRPCABMM",
            "SNCategoryEmbedWrapLRPCAKMeans",
            "SNCategoryEmbedWrapLRPCAGMM",
            "SNCategoryEmbedWrapLRPCABMM",
            "SNCategoryEmbedWrapLR2LPCAKMeans",
            "SNCategoryEmbedWrapLR2LPCAGMM",
            "SNCategoryEmbedWrapLR2LPCABMM",
            "SNFTTransWrapLR2LPCAKMeans",
            "SNFTTransWrapLR2LPCAGMM",
            "SNFTTransWrapLR2LPCABMM",
            "SNTabTransLR2LPCAKMeans",
            "SNTabTransLR2LPCAGMM",
            "SNTabTransLR2LPCABMM",
            "SNTabTransWrapLR2LPCAKMeans",
            "SNTabTransWrapLR2LPCAGMM",
            "SNTabTransWrapLR2LPCABMM",
            "SNAutoIntLRPCAKMeans",
            "SNAutoIntLRPCAGMM",
            "SNAutoIntLRPCABMM",
            "SNAutoIntWrapLRPCAKMeans",
            "SNAutoIntWrapLRPCAGMM",
            "SNAutoIntWrapLRPCABMM",
        ]

    def _new_model(self, model_name, verbose, **kwargs):
        fix_kwargs = dict(
            n_inputs=len(self.datamodule.cont_feature_names),
            n_outputs=len(self.datamodule.label_name),
            layers=self.datamodule.args["layers"],
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            datamodule=self.datamodule,
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
                embedding_dim=3,
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
            "SNCategoryEmbedLR2LPCAKMeans",
            "SNCategoryEmbedLR2LPCAGMM",
            "SNCategoryEmbedLR2LPCABMM",
            "SNFTTransLR2LPCAKMeans",
            "SNFTTransLR2LPCAGMM",
            "SNFTTransLR2LPCABMM",
            "SNPyFTTransLRPCAKMeans",
            "SNPyFTTransLRPCAGMM",
            "SNPyFTTransLRPCABMM",
            "SNPyFTTransWrapLRPCAKMeans",
            "SNPyFTTransWrapLRPCAGMM",
            "SNPyFTTransWrapLRPCABMM",
            "SNPyFTTransLR2LPCAKMeans",
            "SNPyFTTransLR2LPCAGMM",
            "SNPyFTTransLR2LPCABMM",
            "SNPyFTTransWrapLR2LPCAKMeans",
            "SNPyFTTransWrapLR2LPCAGMM",
            "SNPyFTTransWrapLR2LPCABMM",
            "SNCategoryEmbedLRPCAKMeans",
            "SNCategoryEmbedLRPCAGMM",
            "SNCategoryEmbedLRPCABMM",
            "SNCategoryEmbedWrapLRPCAKMeans",
            "SNCategoryEmbedWrapLRPCAGMM",
            "SNCategoryEmbedWrapLRPCABMM",
            "SNCategoryEmbedWrapLR2LPCAKMeans",
            "SNCategoryEmbedWrapLR2LPCAGMM",
            "SNCategoryEmbedWrapLR2LPCABMM",
            "SNFTTransWrapLR2LPCAKMeans",
            "SNFTTransWrapLR2LPCAGMM",
            "SNFTTransWrapLR2LPCABMM",
            "SNTabTransLR2LPCAKMeans",
            "SNTabTransLR2LPCAGMM",
            "SNTabTransLR2LPCABMM",
            "SNTabTransWrapLR2LPCAKMeans",
            "SNTabTransWrapLR2LPCAGMM",
            "SNTabTransWrapLR2LPCABMM",
            "SNAutoIntLRPCAKMeans",
            "SNAutoIntLRPCAGMM",
            "SNAutoIntLRPCABMM",
            "SNAutoIntWrapLRPCAKMeans",
            "SNAutoIntWrapLRPCAGMM",
            "SNAutoIntWrapLRPCABMM",
        ]:
            cls = getattr(sys.modules[__name__], f"{model_name.replace('PCA', '')}NN")
            if "2L" not in model_name:
                feature_idx = cls.basic_clustering_features_idx(self.datamodule)
            else:
                feature_idx = cls.top_clustering_features_idx(self.datamodule)
            if len(feature_idx) > 2:
                pca = self.datamodule.pca(feature_idx=feature_idx)
                n_pca_dim = (
                    np.where(pca.explained_variance_ratio_.cumsum() < 0.9)[0][-1] + 1
                )
            else:
                n_pca_dim = len(feature_idx)
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
                # Integer(
                #     low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                # ),
                Real(low=0.0, high=0.5, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.5, prior="uniform", name="mlp_dropout"),
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
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLRPCABMM",
            "SNPyFTTransLRPCAKMeans",
            "SNPyFTTransLRPCAGMM",
            "SNPyFTTransLRPCABMM",
            "SNPyFTTransWrapLRPCAKMeans",
            "SNPyFTTransWrapLRPCAGMM",
            "SNPyFTTransWrapLRPCABMM",
            "SNAutoIntLRPCAKMeans",
            "SNAutoIntLRPCAGMM",
            "SNAutoIntLRPCABMM",
            "SNAutoIntWrapLRPCAKMeans",
            "SNAutoIntWrapLRPCAGMM",
            "SNAutoIntWrapLRPCABMM",
        ]:
            return [
                # Integer(
                #     low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                # ),
                Integer(low=1, high=64, prior="uniform", name="n_clusters", dtype=int),
            ] + self.trainer.SPACE
        elif model_name in [
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LBMM",
            "SNCatEmbedLR2LPCAKMeans",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLR2LPCABMM",
            "SNCategoryEmbedLRPCAKMeans",
            "SNCategoryEmbedLRPCAGMM",
            "SNCategoryEmbedLRPCABMM",
            "SNCategoryEmbedWrapLRPCAKMeans",
            "SNCategoryEmbedWrapLRPCAGMM",
            "SNCategoryEmbedWrapLRPCABMM",
            "SNCategoryEmbedLR2LPCAKMeans",
            "SNCategoryEmbedLR2LPCAGMM",
            "SNCategoryEmbedLR2LPCABMM",
            "SNPyFTTransLR2LPCAKMeans",
            "SNPyFTTransLR2LPCAGMM",
            "SNPyFTTransLR2LPCABMM",
            "SNPyFTTransWrapLR2LPCAKMeans",
            "SNPyFTTransWrapLR2LPCAGMM",
            "SNPyFTTransWrapLR2LPCABMM",
            "SNFTTransLR2LPCAKMeans",
            "SNFTTransLR2LPCAGMM",
            "SNFTTransLR2LPCABMM",
            "SNCategoryEmbedWrapLR2LPCAKMeans",
            "SNCategoryEmbedWrapLR2LPCAGMM",
            "SNCategoryEmbedWrapLR2LPCABMM",
            "SNFTTransWrapLR2LPCAKMeans",
            "SNFTTransWrapLR2LPCAGMM",
            "SNFTTransWrapLR2LPCABMM",
            "SNTabTransLR2LPCAKMeans",
            "SNTabTransLR2LPCAGMM",
            "SNTabTransLR2LPCABMM",
            "SNTabTransWrapLR2LPCAKMeans",
            "SNTabTransWrapLR2LPCAGMM",
            "SNTabTransWrapLR2LPCABMM",
        ]:
            return [
                # Integer(
                #     low=2, high=32, prior="uniform", name="embedding_dim", dtype=int
                # ),
                Integer(low=4, high=64, prior="uniform", name="n_clusters", dtype=int),
                Integer(
                    low=4,
                    high=32,
                    prior="uniform",
                    name="n_clusters_per_cluster",
                    dtype=int,
                ),
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
                # "embedding_dim": 3,
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
            "SNCatEmbedLRPCAKMeans",
            "SNCatEmbedLRPCAGMM",
            "SNCatEmbedLRPCABMM",
            "SNPyFTTransLRPCAKMeans",
            "SNPyFTTransLRPCAGMM",
            "SNPyFTTransLRPCABMM",
            "SNPyFTTransWrapLRPCAKMeans",
            "SNPyFTTransWrapLRPCAGMM",
            "SNPyFTTransWrapLRPCABMM",
            "SNAutoIntLRPCAKMeans",
            "SNAutoIntLRPCAGMM",
            "SNAutoIntLRPCABMM",
            "SNAutoIntWrapLRPCAKMeans",
            "SNAutoIntWrapLRPCAGMM",
            "SNAutoIntWrapLRPCABMM",
        ]:
            res = {
                # "embedding_dim": 3,
                "n_clusters": 16,
            }
        elif model_name in [
            "SNCatEmbedLR2LKMeans",
            "SNCatEmbedLR2LGMM",
            "SNCatEmbedLR2LBMM",
            "SNCatEmbedLR2LPCAKMeans",
            "SNCatEmbedLR2LPCAGMM",
            "SNCatEmbedLR2LPCABMM",
            "SNCategoryEmbedLR2LPCAKMeans",
            "SNCategoryEmbedLR2LPCAGMM",
            "SNCategoryEmbedLR2LPCABMM",
            "SNFTTransLR2LPCAKMeans",
            "SNFTTransLR2LPCAGMM",
            "SNFTTransLR2LPCABMM",
            "SNPyFTTransLR2LPCAKMeans",
            "SNPyFTTransLR2LPCAGMM",
            "SNPyFTTransLR2LPCABMM",
            "SNPyFTTransWrapLR2LPCAKMeans",
            "SNPyFTTransWrapLR2LPCAGMM",
            "SNPyFTTransWrapLR2LPCABMM",
            "SNCategoryEmbedLRPCAKMeans",
            "SNCategoryEmbedLRPCAGMM",
            "SNCategoryEmbedLRPCABMM",
            "SNCategoryEmbedWrapLRPCAKMeans",
            "SNCategoryEmbedWrapLRPCAGMM",
            "SNCategoryEmbedWrapLRPCABMM",
            "SNCategoryEmbedWrapLR2LPCAKMeans",
            "SNCategoryEmbedWrapLR2LPCAGMM",
            "SNCategoryEmbedWrapLR2LPCABMM",
            "SNFTTransWrapLR2LPCAKMeans",
            "SNFTTransWrapLR2LPCAGMM",
            "SNFTTransWrapLR2LPCABMM",
            "SNTabTransLR2LPCAKMeans",
            "SNTabTransLR2LPCAGMM",
            "SNTabTransLR2LPCABMM",
            "SNTabTransWrapLR2LPCAKMeans",
            "SNTabTransWrapLR2LPCAGMM",
            "SNTabTransWrapLR2LPCABMM",
        ]:
            res = {
                # "embedding_dim": 3,
                "n_clusters": 16,
                "n_clusters_per_cluster": 8,
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
            and "Relative Mean Stress" not in self.datamodule.cont_feature_names
        ):
            return False
        return True

    def required_models(self, model_name: str) -> Union[List[str], None]:
        if "SNCatEmbed" in model_name and "Seq" not in model_name:
            models = ["CategoryEmbedding"]
        elif "SNCategoryEmbed" in model_name:
            models = ["EXTERN_PytorchTabular_Category Embedding"]
        elif "SNFTTrans" in model_name:
            models = ["EXTERN_WideDeep_FTTransformer"]
        elif "SNTabTrans" in model_name:
            models = ["EXTERN_WideDeep_TabTransformer"]
        elif "SNAutoInt" in model_name:
            models = ["EXTERN_PytorchTabular_AutoInt"]
        elif "SNPyFTTrans" in model_name:
            models = ["EXTERN_PytorchTabular_FTTransformer"]
        else:
            models = None
        if models is not None:
            if "Wrap" in model_name:
                models = [x + "_WRAP" for x in models]
        return models

    def _prepare_custom_datamodule(self, model_name):
        from src.data import DataModule

        base = self.trainer.datamodule
        datamodule = DataModule(config=self.trainer.datamodule.args, initialize=False)
        datamodule.set_data_imputer("MeanImputer")
        datamodule.set_data_derivers(
            [("UnscaledDataDeriver", {"derived_name": "Unscaled"})]
        )
        datamodule.set_data_processors([("StandardScaler", {})])
        datamodule.set_data(
            base.df,
            cont_feature_names=base.cont_feature_names,
            cat_feature_names=base.cat_feature_names,
            label_name=base.label_name,
            train_indices=base.train_indices,
            val_indices=base.val_indices,
            test_indices=base.test_indices,
            verbose=False,
        )
        tmp_derived_data = base.derived_data.copy()
        tmp_derived_data.update(datamodule.derived_data)
        datamodule.derived_data = tmp_derived_data
        self.datamodule = datamodule
        return datamodule

    def _run_custom_data_module(self, df, derived_data, model_name):
        df, my_derived_data = self.datamodule.prepare_new_data(df, ignore_absence=True)
        derived_data = derived_data.copy()
        derived_data.update(my_derived_data)
        derived_data = self.datamodule.sort_derived_data(derived_data)
        return df, derived_data, self.datamodule

    # def _bayes_eval(
    #     self,
    #     model,
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    # ):
    #     """
    #     Evaluating the model for bayesian optimization iterations. The larger one of training and evaluation errors is
    #     returned.
    #
    #     Returns
    #     -------
    #     result
    #         The evaluation of bayesian hyperparameter optimization.
    #     """
    #     y_val_pred = self._pred_single_model(model, X_val, verbose=False)
    #     val_loss = metric_sklearn(y_val_pred, y_val, self.trainer.args["loss"])
    #     y_train_pred = self._pred_single_model(model, X_train, verbose=False)
    #     train_loss = metric_sklearn(y_train_pred, y_train, self.trainer.args["loss"])
    #     return max([train_loss, val_loss])
