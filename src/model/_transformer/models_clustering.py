from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .models_with_seq import CatEmbedSeqNN
from ..base import AbstractNN, get_linear
import numpy as np
from .clustering.singlelayer import KMeansSN, GMMSN, BMMSN
from .clustering.multilayer import TwolayerKMeansSN, TwolayerGMMSN, TwolayerBMMSN
import torch
from torch import nn
import warnings


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_outputs,
        trainer,
        clustering_features,
        clustering_sn_model,
        cont_cat_model,
        layers,
        hidden_rep_dim: int,
        ridge_penalty: float = 0.0,
        **kwargs,
    ):
        super(AbstractClusteringModel, self).__init__(trainer, **kwargs)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_sn_model = clustering_sn_model
        self.cont_cat_model = cont_cat_model
        if not hasattr(self.cont_cat_model, "hidden_representation") or not hasattr(
            self.cont_cat_model, "hidden_rep_dim"
        ):
            warnings.warn(
                f"The backbone should have an attribute called `hidden_representation` that records the "
                f"final output of the hidden layer, and `hidden_rep_dim` that records the dim of "
                f"`hidden_representation`. Now the output of the backbone is used instead."
            )
        self.s_zero_slip = trainer.datamodule.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")
        self.ridge_penalty = ridge_penalty
        self.cls_head = get_linear(
            n_inputs=hidden_rep_dim, n_outputs=n_outputs, nonlinearity="relu"
        )
        self.cls_head_normalize = nn.Sigmoid()
        self.cls_head_loss = nn.CrossEntropyLoss()
        if isinstance(self.cont_cat_model, nn.Module):
            self.set_requires_grad(self.cont_cat_model, requires_grad=False)

    def _forward(self, x, derived_tensors):
        # Prediction of deep learning models.
        if isinstance(self.cont_cat_model, nn.Module):
            dl_pred = self.cont_cat_model(x, derived_tensors)
        else:
            name = self.cont_cat_model.get_model_names()[0]
            full_name = f"EXTERN_{self.cont_cat_model.program}_{name}"
            ml_pred = self.cont_cat_model._pred_single_model(
                self.cont_cat_model.model[name],
                X_test=derived_tensors["data_required_models"][full_name],
                verbose=False,
            )
            dl_pred = torch.tensor(ml_pred, device=x.device)
        # Prediction of physical models
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        phy_pred = self.clustering_sn_model(x[:, self.clustering_features], s_wo_bias)
        # Projection from hidden output to deep learning weights
        hidden = getattr(self.cont_cat_model, "hidden_representation", dl_pred)
        dl_weight = self.cls_head_normalize(self.cls_head(hidden))
        # Weighted sum of prediction
        out = phy_pred + torch.mul(dl_weight, dl_pred - phy_pred)
        self.dl_pred = dl_pred
        self.phy_pred = phy_pred
        self.dl_weight = dl_weight
        return out

    def loss_fn(self, y_true, y_pred, *data, **kwargs):
        # Train the regression head
        self.dl_loss = self.default_loss_fn(self.dl_pred, y_true)
        # Train the classification head
        # If the error of dl predictions is lower, cls_label is 0
        cls_label = (
            torch.heaviside(
                torch.abs(self.dl_pred - y_true) - torch.abs(self.phy_pred - y_true),
                torch.tensor([0.0], device=y_true.device),
            )
            .flatten()
            .long()
        )
        self.cls_loss = self.cls_head_loss(
            torch.concat([self.dl_weight, 1 - self.dl_weight], dim=1), cls_label
        )
        self.output_loss = self.default_loss_fn(y_pred, y_true)
        # Train Least Square
        sum_weight = self.clustering_sn_model.nk[self.clustering_sn_model.x_cluster]
        self.lstsq_loss = torch.sum(
            torch.concat(
                [
                    torch.sum(
                        0.5 * (y_true.flatten() - sn.lstsq_output) ** 2 / sum_weight
                    ).unsqueeze(-1)
                    for sn in self.clustering_sn_model.sns
                ]
            )
        )
        # Train Ridge Regression
        ridge_weight = self.clustering_sn_model.running_sn_weight
        self.ridge_loss = torch.sum(
            0.5
            * (y_true.flatten() - self.clustering_sn_model.ridge_output) ** 2
            / sum_weight
        ) + torch.mul(
            torch.sum(torch.mul(ridge_weight, ridge_weight)), self.ridge_penalty
        )
        return self.output_loss

    def configure_optimizers(self):
        all_optimizer = torch.optim.Adam(
            list(self.cls_head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        regression_optimizer = torch.optim.Adam(
            list(self.clustering_sn_model.sns.parameters())
            + [self.clustering_sn_model.running_sn_weight],
            lr=0.8,
            weight_decay=0,
        )
        return [all_optimizer, regression_optimizer]

    def cal_backward_step(self, loss):
        optimizers = self.optimizers()
        all_optimizer = optimizers[0]
        regression_optimizer = optimizers[1]
        # The following commented zero_grad() operations are not necessary because `inputs`s are specified and no other
        # gradient is calculated.
        # 1st back-propagation: for deep learning weights.
        self.dl_weight.retain_grad()
        self.manual_backward(
            self.cls_loss,
            retain_graph=True,
            inputs=list(self.cls_head.parameters()),
        )
        # self.cont_cat_model.zero_grad()
        # self.clustering_sn_model.sns.zero_grad()
        # if self.clustering_sn_model.running_sn_weight.grad is not None:
        #     self.clustering_sn_model.running_sn_weight.grad.zero_()

        # 2nd back-propagation: for Ridge regression.
        self.manual_backward(
            self.ridge_loss,
            retain_graph=True,
            inputs=self.clustering_sn_model.running_sn_weight,
        )
        # self.cont_cat_model.zero_grad()
        # self.clustering_sn_model.sns.zero_grad()

        # 3rd back-propagation: for Least Square.
        self.manual_backward(
            self.lstsq_loss,
            inputs=list(self.clustering_sn_model.sns.parameters()),
            retain_graph=True,
        )
        # self.cont_cat_model.zero_grad()

        # 4th back-propagation: for deep learning backbones.
        # self.manual_backward(self.dl_loss)

        all_optimizer.step()
        regression_optimizer.step()

    @staticmethod
    def basic_clustering_features_idx(trainer) -> np.ndarray:
        return np.concatenate(
            (
                trainer.datamodule.get_feature_idx_by_type(typ="Material"),
                [trainer.cont_feature_names.index(x) for x in ["Frequency", "R-value"]],
            )
        )

    @staticmethod
    def top_clustering_features_idx(trainer):
        return AbstractClusteringModel.basic_clustering_features_idx(trainer)[:-2]


class SNTransformerLRKMeansNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        n_pca_dim: int = None,
        **kwargs,
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
            n_pca_dim=n_pca_dim,
        )
        super(SNTransformerLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=transformer,
            layers=layers,
            hidden_rep_dim=transformer.hidden_rep_dim,
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
        required_models,
        n_pca_dim: int = None,
        **kwargs,
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = required_models["CategoryEmbedding"]
        sn = KMeansSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
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
        **kwargs,
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
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRKMeansSeqNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
            **kwargs,
        )


class SNCatEmbedLRGMMNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        required_models,
        n_pca_dim=None,
        **kwargs,
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = required_models["CategoryEmbedding"]
        sn = GMMSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
            **kwargs,
        )


class SNCatEmbedLRBMMNN(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        n_clusters,
        required_models,
        n_pca_dim=None,
        **kwargs,
    ):
        clustering_features = self.basic_clustering_features_idx(trainer)
        catembed = required_models["CategoryEmbedding"]
        sn = BMMSN(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLRBMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
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
        n_clusters_per_cluster: int,
        required_models,
        n_pca_dim: int = None,
        **kwargs,
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = required_models["CategoryEmbedding"]
        sn = TwolayerGMMSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LGMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
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
        n_clusters_per_cluster: int,
        required_models,
        n_pca_dim: int = None,
        **kwargs,
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = required_models["CategoryEmbedding"]
        sn = TwolayerKMeansSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LKMeansNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
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
        n_clusters_per_cluster: int,
        required_models,
        n_pca_dim: int = None,
        **kwargs,
    ):
        clustering_features = list(self.basic_clustering_features_idx(trainer))
        top_level_clustering_features = self.top_clustering_features_idx(trainer)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        catembed = required_models["CategoryEmbedding"]
        sn = TwolayerBMMSN(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
        )
        super(SNCatEmbedLR2LBMMNN, self).__init__(
            n_outputs=n_outputs,
            trainer=trainer,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=catembed,
            layers=layers,
            hidden_rep_dim=catembed.hidden_rep_dim,
            **kwargs,
        )
