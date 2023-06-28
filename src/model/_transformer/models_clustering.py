from .models_with_seq import CatEmbedSeqNN
from ..base import AbstractNN, get_linear, get_sequential, AbstractModel
import numpy as np
from .clustering.singlelayer import KMeansSN, GMMSN, BMMSN
from .clustering.multilayer import TwolayerKMeansSN, TwolayerGMMSN, TwolayerBMMSN
import torch
from torch import nn
from ..widedeep import WideDeepWrapper
from ..pytorch_tabular import PytorchTabularWrapper
from ..base import TorchModelWrapper


class AbstractClusteringModel(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        clustering_features,
        clustering_sn_model,
        cont_cat_model,
        layers,
        ridge_penalty: float = 0.0,
        **kwargs,
    ):
        super(AbstractClusteringModel, self).__init__(datamodule, **kwargs)
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        self.clustering_features = clustering_features
        self.clustering_sn_model = clustering_sn_model
        self.cont_cat_model = cont_cat_model
        self.use_hidden_rep, hidden_rep_dim = self._test_required_model(
            n_inputs, self.cont_cat_model
        )
        if not self.use_hidden_rep:
            self.cls_head = get_sequential(
                [128, 64, 32],
                n_inputs=hidden_rep_dim,
                n_outputs=n_outputs,
                act_func=nn.ReLU,
                dropout=0,
                use_norm=False,
            )
        else:
            self.cls_head = get_linear(
                n_inputs=hidden_rep_dim, n_outputs=n_outputs, nonlinearity="relu"
            )
        self.ridge_penalty = ridge_penalty
        self.cls_head_normalize = nn.Sigmoid()
        self.cls_head_loss = nn.CrossEntropyLoss()
        if isinstance(self.cont_cat_model, nn.Module):
            self.set_requires_grad(self.cont_cat_model, requires_grad=False)

    def _forward(self, x, derived_tensors):
        # Prediction of deep learning models.
        dl_pred = self.call_required_model(self.cont_cat_model, x, derived_tensors)
        if self.use_hidden_rep:
            hidden = self.get_hidden_state(self.cont_cat_model, x, derived_tensors)
        else:
            hidden = torch.concat([x, dl_pred], dim=1)
        # Prediction of physical models
        phy_pred = self.clustering_sn_model(x, self.clustering_features)
        # Projection from hidden output to deep learning weights
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
    def basic_clustering_features_idx(datamodule) -> np.ndarray:
        return np.concatenate(
            (
                datamodule.get_feature_idx_by_type(typ="Material"),
                [
                    datamodule.cont_feature_names.index(x)
                    for x in ["Frequency", "R-value"]
                ],
            )
        ).astype(int)

    @staticmethod
    def top_clustering_features_idx(datamodule):
        return AbstractClusteringModel.basic_clustering_features_idx(datamodule)[:-2]


class Abstract1LClusteringModel(AbstractClusteringModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        n_clusters,
        sn_class,
        cont_cat_model,
        n_pca_dim: int = None,
        **kwargs,
    ):
        clustering_features = self.basic_clustering_features_idx(datamodule)
        sn = sn_class(
            n_clusters=n_clusters,
            n_input=len(clustering_features),
            n_pca_dim=n_pca_dim,
            datamodule=datamodule,
        )
        super(Abstract1LClusteringModel, self).__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
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
        sn_class,
        cont_cat_model,
        n_pca_dim: int = None,
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
        sn = sn_class(
            n_clusters=n_clusters,
            n_input_1=len(input_1_idx),
            n_input_2=len(input_2_idx),
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            datamodule=datamodule,
        )
        super(Abstract2LClusteringModel, self).__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            clustering_features=clustering_features,
            clustering_sn_model=sn,
            cont_cat_model=cont_cat_model,
            layers=layers,
            **kwargs,
        )


class SNAutoIntLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt"]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNAutoIntLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt"]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNAutoIntLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt"]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNAutoIntWrapLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt_WRAP"]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNAutoIntWrapLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt_WRAP"]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNAutoIntWrapLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_AutoInt_WRAP"]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNFTTransLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer"]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNFTTransLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer"]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNFTTransLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer"]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNPyFTTransLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer"]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNPyFTTransLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer"]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNPyFTTransLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer"]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNPyFTTransWrapLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer_WRAP"]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNPyFTTransWrapLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer_WRAP"]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNPyFTTransWrapLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_FTTransformer_WRAP"]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNTabTransLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer"]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNTabTransLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer"]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNTabTransLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer"]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedWrapLRKMeansNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(sn_class=KMeansSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedWrapLRGMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(sn_class=GMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedWrapLRBMMNN(Abstract1LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(sn_class=BMMSN, cont_cat_model=cont_cat_model, **kwargs)


class SNCategoryEmbedLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_PytorchTabular_Category Embedding"]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNFTTransWrapLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNFTTransWrapLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNFTTransWrapLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_FTTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedWrapLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedWrapLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNCategoryEmbedWrapLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models[
            "EXTERN_PytorchTabular_Category Embedding_WRAP"
        ]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNTabTransWrapLR2LGMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerGMMSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNTabTransWrapLR2LKMeansNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerKMeansSN, cont_cat_model=cont_cat_model, **kwargs
        )


class SNTabTransWrapLR2LBMMNN(Abstract2LClusteringModel):
    def __init__(self, required_models, **kwargs):
        cont_cat_model = required_models["EXTERN_WideDeep_TabTransformer_WRAP"]
        super().__init__(
            sn_class=TwolayerBMMSN, cont_cat_model=cont_cat_model, **kwargs
        )
