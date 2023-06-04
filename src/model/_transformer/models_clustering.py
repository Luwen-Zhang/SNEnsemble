from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .models_with_seq import CatEmbedSeqNN
from ..base import AbstractNN
import numpy as np
from .clustering.singlelayer import KMeansSN, GMMSN, BMMSN
from .clustering.multilayer import TwolayerKMeansSN, TwolayerGMMSN, TwolayerBMMSN


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_outputs,
        trainer,
        clustering_features,
        clustering_sn_model,
        cont_cat_model,
        **kwargs
    ):
        super(AbstractClusteringModel, self).__init__(trainer, **kwargs)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_sn_model = clustering_sn_model
        self.cont_cat_model = cont_cat_model
        self.s_zero_slip = trainer.datamodule.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        naive_pred = self.cont_cat_model(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.clustering_sn_model(
            x[:, self.clustering_features], s_wo_bias, naive_pred
        )
        return x_out

    def loss_fn(self, y_true, y_pred, *data, **kwargs):
        self.naive_pred_loss = self.default_loss_fn(self._naive_pred, y_true)
        self.output_loss = self.default_loss_fn(y_pred, y_true)
        return self.output_loss

    def cal_backward_step(self, loss):
        self.manual_backward(self.output_loss, retain_graph=True)
        self.cont_cat_model.zero_grad()
        self.manual_backward(self.naive_pred_loss)
        self.optimizers().step()

    @staticmethod
    def basic_clustering_features_idx(trainer) -> np.ndarray:
        return np.concatenate(
            (
                trainer.datamodule.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )

    @staticmethod
    def top_clustering_features_idx(trainer):
        return AbstractClusteringModel.basic_clustering_features_idx(trainer)[:-1]


class SNTransformerLRKMeansNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            layers=layers,
            n_pca_dim=n_pca_dim,
        )
        super(SNTransformerLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=transformer,
            **kwargs,
        )


class SNCatEmbedLRKMeansNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            layers=layers,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLRKMeansSeqNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = CatEmbedSeqNN(
            n_inputs=n_inputs,
            n_outputs=1,
            layers=layers,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            layers=layers,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRKMeansSeqNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLRGMMNN(AbstractClusteringModel):
    def __init__(
        self, n_inputs, n_outputs, layers, trainer, n_clusters, n_pca_dim=None, **kwargs
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = GMMSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            layers=layers,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLRBMMNN(AbstractClusteringModel):
    def __init__(
        self, n_inputs, n_outputs, layers, trainer, n_clusters, n_pca_dim=None, **kwargs
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = BMMSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            layers=layers,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRBMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLR2LGMMNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = TwolayerGMMSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            n_clusters_per_cluster=5,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLR2LKMeansNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = TwolayerKMeansSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            n_clusters_per_cluster=5,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLR2LBMMNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = TwolayerBMMSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            layers=layers,
            n_clusters_per_cluster=5,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LBMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )
