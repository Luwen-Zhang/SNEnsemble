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
from tabensemb.model.widedeep import WideDeepWrapper
from tabensemb.model.pytorch_tabular import PytorchTabularWrapper
from tabensemb.model.base import TorchModelWrapper


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        clustering_features,
        clustering_phy_model,
        cont_cat_model,
        layers,
        ridge_penalty: float = 0.0,
        uncertainty: str = None,
        **kwargs,
    ):
        super(AbstractClusteringModel, self).__init__(datamodule, **kwargs)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_phy_model = clustering_phy_model
        self.cont_cat_model = cont_cat_model
        self.use_hidden_rep, hidden_rep_dim = self._test_required_model(
            n_inputs, self.cont_cat_model
        )
        self.uncertainty = uncertainty
        if uncertainty == "gp":
            self.gp = ExactGPModel(dynamic_input=True)
            import gpytorch

            gpytorch.settings.debug._set_state(tabensemb.setting["debug_mode"])
        else:
            self.gp = None
        if uncertainty == "bnn":
            self.cls_head = MCDropout(
                layers=[128, 64, 32],
                n_inputs=hidden_rep_dim,
                n_outputs=2,
                train_dropout=0.1,
                sample_dropout=0.1,
                type="homo",
                task="classification",
            )
        else:
            self.cls_head = nn.Sequential(
                get_sequential(
                    [128, 64, 32],
                    n_inputs=hidden_rep_dim,
                    n_outputs=1,
                    act_func=nn.ReLU,
                    dropout=0,
                    use_norm=False,
                ),
                nn.Sigmoid(),
            )
        self.ridge_penalty = ridge_penalty
        self.cls_head_loss = nn.BCELoss()
        if isinstance(self.cont_cat_model, nn.Module):
            self.set_requires_grad(self.cont_cat_model, requires_grad=False)

    def on_train_start(self) -> None:
        super(AbstractClusteringModel, self).on_train_start()
        if self.gp is not None:
            self.gp.on_train_start()

    def on_train_epoch_start(self) -> None:
        super(AbstractClusteringModel, self).on_train_epoch_start()
        if self.gp is not None:
            self.gp.on_epoch_start()

    def on_train_epoch_end(self) -> None:
        super(AbstractClusteringModel, self).on_train_epoch_end()
        if self.gp is not None:
            self.gp.on_epoch_end()

    def _forward(self, x, derived_tensors):
        # Prediction of deep learning models.
        dl_pred = self.call_required_model(self.cont_cat_model, x, derived_tensors)
        if self.use_hidden_rep:
            hidden = self.get_hidden_state(self.cont_cat_model, x, derived_tensors)
        else:
            hidden = torch.concat([x, dl_pred], dim=1)
        # Prediction of physical models
        phy_pred = self.clustering_phy_model(
            x, self.clustering_features, derived_tensors
        )
        # Projection from hidden output to deep learning weights
        dl_weight = self.cls_head(hidden)
        if self.uncertainty is not None:
            if self.uncertainty == "gp":
                mu, var = self.gp(hidden, dl_weight)
            else:
                mu, al_var, var = self.cls_head.predict(
                    hidden, n_samples=100, return_separate_var=True
                )
            std = torch.sqrt(var)
            uncertain_dl_weight = torch.clamp(dl_weight - std.view(-1, 1), min=1e-8)
            self.uncertain_dl_weight = uncertain_dl_weight
            self.mu = mu
            self.std = std
            # out = phy_pred + torch.mul(uncertain_dl_weight, dl_pred - phy_pred)
        out = phy_pred + torch.mul(dl_weight, dl_pred - phy_pred)
        self.dl_pred = dl_pred
        self.phy_pred = phy_pred
        self.dl_weight = dl_weight
        return out

    def loss_fn(self, y_pred, y_true, *data, **kwargs):
        # Train the regression head
        self.dl_loss = self.default_loss_fn(self.dl_pred, y_true)
        # Train the classification head
        # If the error of dl predictions is lower, cls_label is 0
        cls_label = torch.heaviside(
            torch.abs(self.dl_pred - y_true) - torch.abs(self.phy_pred - y_true),
            torch.tensor([0.0], device=y_true.device),
        )
        self.cls_loss = self.cls_head_loss(self.dl_weight, 1 - cls_label)
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
        # Train Ridge Regression
        ridge_weight = self.clustering_phy_model.running_phy_weight
        self.ridge_loss = torch.sum(
            0.5
            * (y_true.flatten() - self.clustering_phy_model.ridge_output) ** 2
            / sum_weight
        ) + torch.mul(
            torch.sum(torch.mul(ridge_weight, ridge_weight)), self.ridge_penalty
        )
        if self.gp is not None:
            self.gp_loss = self.gp.loss
        return self.output_loss

    def configure_optimizers(self):
        cls_optimizer = torch.optim.Adam(
            list(self.cls_head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        ridge_optimizer = torch.optim.Adam(
            [self.clustering_phy_model.running_phy_weight],
            lr=0.8,
            weight_decay=0,
        )
        lstsq_optimizer = [
            phy.get_optimizer() for phy in self.clustering_phy_model.phys
        ]
        if self.gp is not None:
            gp_optimizer = self.gp._get_optimizer(**self.gp.kwargs)
            return [cls_optimizer, ridge_optimizer, gp_optimizer] + lstsq_optimizer
        else:
            return [cls_optimizer, ridge_optimizer] + lstsq_optimizer

    def cal_backward_step(self, loss):
        optimizers = self.optimizers()
        if self.gp is not None:
            cls_optimizer = optimizers[0]
            ridge_optimizer = optimizers[1]
            gp_optimizer = optimizers[2]
            lstsq_optimizers = optimizers[3:]
        else:
            cls_optimizer = optimizers[0]
            ridge_optimizer = optimizers[1]
            lstsq_optimizers = optimizers[2:]
            gp_optimizer = None
        # The following commented zero_grad() operations are not necessary because `inputs`s are specified and no other
        # gradient is calculated.
        # 1st back-propagation: for deep learning weights.
        self.dl_weight.retain_grad()
        self.manual_backward(
            self.cls_loss,
            retain_graph=True,
            inputs=[x for x in self.cls_head.parameters() if x.requires_grad],
        )
        # self.cont_cat_model.zero_grad()
        # self.clustering_phy_model.phys.zero_grad()
        # if self.clustering_phy_model.running_phy_weight.grad is not None:
        #     self.clustering_phy_model.running_phy_weight.grad.zero_()

        # 2nd back-propagation: for Ridge regression.
        self.manual_backward(
            self.ridge_loss,
            retain_graph=True,
            inputs=self.clustering_phy_model.running_phy_weight,
        )
        if self.gp is not None:
            if self.gp.optim_hp:
                self.manual_backward(
                    self.gp_loss, retain_graph=True, inputs=list(self.gp.parameters())
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

        cls_optimizer.step()
        for optimizer in lstsq_optimizers:
            optimizer.step()
        ridge_optimizer.step()
        if self.gp is not None:
            if self.gp.optim_hp:
                gp_optimizer.step()

    @staticmethod
    def basic_clustering_features_idx(datamodule) -> np.ndarray:
        return np.concatenate(
            (
                datamodule.get_feature_idx_by_type(typ="Material"),
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
        cont_cat_model,
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
            cont_cat_model=cont_cat_model,
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
        cont_cat_model,
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
            cont_cat_model=cont_cat_model,
            layers=layers,
            **kwargs,
        )
