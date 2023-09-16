import tabensemb
from .models_with_seq import CatEmbedSeqNN
from tabensemb.model.base import AbstractNN, get_linear, get_sequential, AbstractModel
import numpy as np
from .clustering.singlelayer import KMeansPhy, GMMPhy, BMMPhy
from .clustering.multilayer import TwolayerKMeansPhy, TwolayerGMMPhy, TwolayerBMMPhy
from .gp.exact_gp import ExactGPModel
from .bayes_nn.bbp import MCDropout
import torch
from torch import nn
from tabensemb.model.sample import CategoryEmbeddingNN
from tabensemb.model.widedeep import WideDeepWrapper
from tabensemb.model.pytorch_tabular import PytorchTabularWrapper
from tabensemb.model.base import TorchModelWrapper
from copy import deepcopy as cp


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        clustering_features,
        clustering_phy_model,
        layers,
        l2_penalty: float = 0.0,
        l1_penalty: float = 0.0,
        **kwargs,
    ):
        super(AbstractClusteringModel, self).__init__(datamodule, **kwargs)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_phy_model = clustering_phy_model
        datamodule = cp(datamodule)
        datamodule.cont_feature_names += [
            f"phy_{i}" for i in self.clustering_phy_model.phys
        ]
        self.head = CategoryEmbeddingNN(
            layers=layers,
            datamodule=datamodule,
            embedding_dim=3,
            embed_extend_dim=False,
            **kwargs,
        )
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.cls_head_loss = nn.MSELoss()

    def _forward(self, x, derived_tensors):
        # Prediction of physical models
        _ = self.clustering_phy_model(x, self.clustering_features, derived_tensors)
        sns_preds = self.clustering_phy_model.weight_input
        x = torch.concat([x, sns_preds], dim=-1)
        out = self.head(x, derived_tensors)
        return out

    def loss_fn(self, y_pred, y_true, *data, **kwargs):
        # Train the regression head
        self.output_loss = self.default_loss_fn(y_pred, y_true)
        # Train Least Square
        sum_weight = self.clustering_phy_model.nk[self.clustering_phy_model.x_cluster]
        self.lstsq_loss = torch.sum(
            torch.concat(
                [
                    torch.sum(
                        0.5 * (y_true.flatten() - phy.lstsq_output) ** 2 / sum_weight
                    ).unsqueeze(-1)
                    for phy in self.clustering_phy_model.phys
                ]
            )
        )
        # Train weighted summation
        weight = self.clustering_phy_model.weight
        self.weight_loss = (
            torch.sum(
                0.5
                * (y_true.flatten() - self.clustering_phy_model.weight_output) ** 2
                / sum_weight
            )
            + torch.mul(torch.sum(torch.mul(weight, weight)), self.l2_penalty)
            + torch.mul(torch.sum(torch.abs(weight)), self.l1_penalty)
        )
        return self.output_loss

    def configure_optimizers(self):
        head_optimizer = torch.optim.Adam(
            list(self.head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        weight_optimizer = torch.optim.Adam(
            [self.clustering_phy_model.running_phy_weight],
            lr=0.8,
            weight_decay=0,
        )
        lstsq_optimizer = [
            phy.get_optimizer() for phy in self.clustering_phy_model.phys
        ]
        return [head_optimizer, weight_optimizer] + lstsq_optimizer

    def cal_backward_step(self, loss):
        optimizers = self.optimizers()
        head_optimizer = optimizers[0]
        weight_optimizer = optimizers[1]
        lstsq_optimizers = optimizers[2:]
        # The following commented zero_grad() operations are not necessary because `inputs`s are specified and no other
        # gradient is calculated.
        # 1st back-propagation: for deep learning weights.
        self.manual_backward(
            self.output_loss,
            retain_graph=True,
            inputs=[x for x in self.head.parameters() if x.requires_grad],
        )
        # self.cont_cat_model.zero_grad()
        # self.clustering_phy_model.phys.zero_grad()
        # if self.clustering_phy_model.running_phy_weight.grad is not None:
        #     self.clustering_phy_model.running_phy_weight.grad.zero_()

        # 2nd back-propagation: for weight.
        self.manual_backward(
            self.weight_loss,
            retain_graph=True,
            inputs=self.clustering_phy_model.running_phy_weight,
        )
        # self.cont_cat_model.zero_grad()
        # self.clustering_phy_model.phys.zero_grad()

        # 3rd back-propagation: for Least Square.
        self.manual_backward(
            self.lstsq_loss,
            inputs=list(self.clustering_phy_model.phys.parameters()),
            retain_graph=True,
        )
        # torch.nn.utils.clip_grad_value_(
        #     list(self.clustering_phy_model.phys.parameters()), 1
        # )
        # self.cont_cat_model.zero_grad()

        # 4th back-propagation: for deep learning backbones.
        # self.manual_backward(self.dl_loss)

        head_optimizer.step()
        for optimizer in lstsq_optimizers:
            optimizer.step()
        weight_optimizer.step()

    @staticmethod
    def basic_clustering_features_idx(datamodule) -> np.ndarray:
        return np.concatenate(
            (
                datamodule.get_feature_idx_by_type(
                    typ="Material", var_type="continuous"
                ),
                list(AbstractClusteringModel.top_clustering_features_idx(datamodule)),
            )
        ).astype(int)

    @staticmethod
    def top_clustering_features_idx(datamodule):
        top_clustering_features = [
            x for x in ["Frequency", "R-value"] if x in datamodule.cont_feature_names
        ]
        return np.array(
            [datamodule.cont_feature_names.index(x) for x in top_clustering_features]
        )


class Abstract1LClusteringModel(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        n_clusters,
        phy_class,
        n_pca_dim: int = None,
        phy_category: str = None,
        **kwargs,
    ):
        clustering_features = self.basic_clustering_features_idx(datamodule)
        phy = phy_class(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            n_pca_dim=n_pca_dim,
            datamodule=datamodule,
            phy_category=phy_category,
        )
        super(Abstract1LClusteringModel, self).__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            clustering_features=clustering_features,
            clustering_phy_model=phy,
            layers=layers,
            **kwargs,
        )


class Abstract2LClusteringModel(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        n_clusters,
        n_clusters_per_cluster: int,
        phy_class,
        n_pca_dim: int = None,
        phy_category: str = None,
        **kwargs,
    ):
        clustering_features = list(self.basic_clustering_features_idx(datamodule))
        top_level_clustering_features = self.top_clustering_features_idx(datamodule)
        input_1_idx = [
            list(clustering_features).index(x) for x in top_level_clustering_features
        ]
        input_2_idx = list(
            np.setdiff1d(np.arange(len(clustering_features)), input_1_idx)
        )
        phy = phy_class(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            datamodule=datamodule,
            phy_category=phy_category,
        )
        super(Abstract2LClusteringModel, self).__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            clustering_features=clustering_features,
            clustering_phy_model=phy,
            layers=layers,
            **kwargs,
        )
