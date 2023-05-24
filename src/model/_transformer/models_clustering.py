from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .models_with_seq import CatEmbedSeqNN
from ..base import get_sequential, AbstractNN
import numpy as np


class SNTransformerLRKMeansNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerLRKMeansNN, self).__init__(trainer)
        from .sn_lr_kmeans import KMeansSN

        self.cluster_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        self.sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(self.cluster_features), layers=layers
        )
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.transformer(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(x[:, self.cluster_features], s_wo_bias, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNCatEmbedLRKMeansNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNCatEmbedLRKMeansNN, self).__init__(trainer)
        from .sn_lr_kmeans import KMeansSN

        self.cluster_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        self.sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(self.cluster_features), layers=layers
        )
        self.catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.catembed(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(x[:, self.cluster_features], s_wo_bias, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNCatEmbedLRKMeansSeqNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNCatEmbedLRKMeansSeqNN, self).__init__(trainer)
        from .sn_lr_kmeans import KMeansSN

        self.cluster_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        self.sn = KMeansSN(
            n_clusters=n_clusters, n_input=len(self.cluster_features), layers=layers
        )
        self.catembed = CatEmbedSeqNN(
            n_inputs=n_inputs,
            n_outputs=1,
            layers=layers,
            trainer=trainer,
            **kwargs,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.catembed(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(x[:, self.cluster_features], s_wo_bias, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNCatEmbedLRGMMNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, n_clusters, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNCatEmbedLRGMMNN, self).__init__(trainer)
        from .sn_lr_gmm import GMMSN

        self.cluster_features = np.concatenate(
            (
                trainer.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )
        self.sn = GMMSN(
            n_clusters=n_clusters, n_input=len(self.cluster_features), layers=layers
        )
        self.catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.catembed(x, derived_tensors)
        self._naive_pred = naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(x[:, self.cluster_features], s_wo_bias, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss
