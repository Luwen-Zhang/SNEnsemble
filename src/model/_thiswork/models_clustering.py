from tabensemb.model.base import AbstractNN, get_sequential
import numpy as np
import torch
from torch import nn


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        datamodule,
        clustering_phy_model,
        cont_cat_model,
        dropout=0.0,
        **kwargs,
    ):
        super(AbstractClusteringModel, self).__init__(datamodule, **kwargs)
        if self.n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_phy_model = clustering_phy_model
        self.cont_cat_model = cont_cat_model
        self.use_hidden_rep, hidden_rep_dim = self._test_required_model(
            self.n_inputs, self.cont_cat_model
        )
        self.cls_head = nn.Sequential(
            get_sequential(
                [128, 64, 32],
                n_inputs=hidden_rep_dim,
                n_outputs=1,
                act_func=nn.ReLU,
                dropout=dropout,
                use_norm=True,
                norm_type="batch",
            ),
            nn.Sigmoid(),
        )
        self.cls_head_loss = nn.BCELoss()
        if isinstance(self.cont_cat_model, nn.Module):
            self.set_requires_grad(self.cont_cat_model, requires_grad=False)
        if isinstance(self.clustering_phy_model, nn.Module):
            self.set_requires_grad(self.clustering_phy_model, requires_grad=False)

    def _forward(self, x, derived_tensors):
        # Prediction of deep learning models.
        dl_pred = self.call_required_model(self.cont_cat_model, x, derived_tensors)
        if self.use_hidden_rep:
            hidden = self.get_hidden_state(self.cont_cat_model, x, derived_tensors)
        else:
            hidden = torch.concat([x, dl_pred], dim=1)
        # Prediction of physical models
        phy_modelname = f"PHYSICS_{'NoPCA' if not hasattr(self.clustering_phy_model, 'pca') else 'PCA'}_{self.clustering_phy_model.clustering.__class__.__name__.split('PCA')[-1]}"
        phy_pred = self.call_required_model(
            self.clustering_phy_model,
            x,
            derived_tensors,
            model_name=phy_modelname,
        )
        # Projection from hidden output to deep learning weights
        dl_weight = self.cls_head(hidden)
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
        return self.output_loss

    def configure_optimizers(self):
        cls_optimizer = torch.optim.Adam(
            list(self.cls_head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return cls_optimizer

    def cal_backward_step(self, loss):
        cls_optimizer = self.optimizers()
        self.dl_weight.retain_grad()
        self.manual_backward(
            self.cls_loss,
            retain_graph=True,
            inputs=[x for x in self.cls_head.parameters() if x.requires_grad],
        )
        cls_optimizer.step()
