from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .models_with_seq import CatEmbedSeqNN
from ..base import get_sequential, AbstractNN
import numpy as np
from .clustering.singlelayer import KMeansSN, GMMSN


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_outputs,
        trainer,
        clustering_features,
        clustering_sn_model,
        cont_cat_model,
    ):
        super(AbstractClusteringModel, self).__init__(trainer)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_sn_model = clustering_sn_model
        self.cont_cat_model = cont_cat_model
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.cont_cat_model(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.clustering_sn_model(
            x[:, self.clustering_features], s_wo_bias, naive_pred
        )
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNTransformerLRKMeansNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
        )
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        super(SNTransformerLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=transformer,
        )


class SNCatEmbedLRKMeansNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        super(SNCatEmbedLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
        )


class SNCatEmbedLRKMeansSeqNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
        )
        catembed = CatEmbedSeqNN(
            n_inputs=n_inputs,
            n_outputs=1,
            layers=layers,
            trainer=trainer,
            **kwargs,
        )
        super(SNCatEmbedLRKMeansSeqNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
        )


class SNCatEmbedLRGMMNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        sn = GMMSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        super(SNCatEmbedLRGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
        )
