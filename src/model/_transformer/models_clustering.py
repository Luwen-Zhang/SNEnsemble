from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .models_with_seq import CatEmbedSeqNN
from ..base import get_sequential, AbstractNN
import numpy as np
from .clustering.singlelayer import KMeansSN, GMMSN
from .clustering.multilayer import TwolayerKMeansSN, TwolayerGMMSN
import torch


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
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
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


class SNTransformerLRKMeansNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
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
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
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
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        catembed = CatEmbedSeqNN(
            n_inputs=n_inputs,
            n_outputs=1,
            layers=layers,
            trainer=trainer,
            **kwargs,
        )
        sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
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
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        sn = GMMSN(
            n_clusters=n_clusters, n_input=len(clustering_features), layers=layers
        )
        super(SNCatEmbedLRGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )


class SNCatEmbedLR2LGMMNN(AbstractClusteringModel):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        input_1_idx = list(np.arange(0, len(clustering_features) - 1))
        input_2_idx = list(
            np.arange(len(clustering_features) - 1, len(clustering_features))
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
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        clustering_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["R-value"]],
            )
        )
        input_1_idx = list(np.arange(0, len(clustering_features) - 1))
        input_2_idx = list(
            np.arange(len(clustering_features) - 1, len(clustering_features))
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
        )
        super(SNCatEmbedLR2LKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            **kwargs,
        )
