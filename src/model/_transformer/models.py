from .cont_cat_embedding import Embedding, Embedding1d
from .fttransformer import FTTransformer
from .lstm import LSTM
from .loss import BiasLoss, ConsGradLoss, StressGradLoss
from .seq_fttransformer import SeqFTTransformer
from .sn import SN
from ..base import get_sequential, AbstractNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import numpy as np


class FTTransformerNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        trainer,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        embed_dropout=0.1,
        attn_layers=4,
        attn_heads=8,
        attn_dropout=0.1,
        **kwargs,
    ):
        super(FTTransformerNN, self).__init__(trainer)

        self.embed = Embedding(
            embedding_dim,
            n_inputs,
            embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = FTTransformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + n_inputs,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=embedding_dim * 4,
            ff_layers=[],
            dropout=attn_dropout,
            n_outputs=n_outputs,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        return x_trans


class TransformerLSTMNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        seq_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        attn_layers=4,
        attn_heads=8,
        embed_dropout=0.1,
        attn_ff_dim=256,
        attn_dropout=0.1,
    ):
        super(TransformerLSTMNN, self).__init__(trainer)
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
            cat_num_unique=cat_num_unique,
            embedding_dim=embedding_dim,
            embed_dropout=embed_dropout,
            attn_layers=attn_layers,
            attn_heads=attn_heads,
            attn_ff_dim=attn_ff_dim,
            attn_dropout=attn_dropout,
        )

        self.lstm = LSTM(
            n_hidden,
            seq_embedding_dim,
            lstm_layers,
            run="Number of Layers" in self.derived_feature_names,
        )

        if self.lstm.run or n_outputs != 1:
            self.w = get_sequential(
                layers,
                1 * n_outputs + int(self.lstm.run),
                n_outputs,
                nn.ReLU,
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x, derived_tensors)
        all_res = [x_trans]

        x_lstm = self.lstm(x, derived_tensors)
        if self.lstm.run:
            all_res += [x_lstm]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class TransformerSeqNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        seq_embedding_dim=16,
        embed_dropout=0.1,
        attn_layers=4,
        attn_heads=8,
        attn_ff_dim=256,
        attn_dropout=0.1,
        seq_attn_layers=4,
        seq_attn_heads=8,
        seq_attn_dropout=0.1,
    ):
        super(TransformerSeqNN, self).__init__(trainer)
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
            cat_num_unique=cat_num_unique,
            embedding_dim=embedding_dim,
            embed_dropout=embed_dropout,
            attn_layers=attn_layers,
            attn_heads=attn_heads,
            attn_ff_dim=attn_ff_dim,
            attn_dropout=attn_dropout,
        )

        self.seq_transformer = SeqFTTransformer(
            n_inputs=None,
            attn_heads=seq_attn_heads,
            attn_layers=seq_attn_layers,
            embedding_dim=seq_embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=layers,
            dropout=seq_attn_dropout,
            n_outputs=n_outputs,
            run="Lay-up Sequence" in self.derived_feature_names
            and "Number of Layers" in self.derived_feature_names,
        )

        if self.seq_transformer.run or n_outputs != 1:
            self.w = get_sequential(
                layers,
                (1 + int(self.seq_transformer.run)) * n_outputs,
                n_outputs,
                nn.ReLU,
                norm_type="layer",
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x, derived_tensors)
        all_res = [x_trans]

        x_seq = self.seq_transformer(x, derived_tensors)
        if self.seq_transformer.run:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class SNTransformerNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerNN, self).__init__(trainer)
        self.sn = SN()
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1,
            n_outputs=sum(self.sn.n_coeff_ls) + len(self.sn.n_coeff_ls),
            act_func=nn.ReLU,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        self.s_original.requires_grad_()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.transformer(x, derived_tensors)
        self._naive_pred = naive_pred
        grad_s = torch.autograd.grad(
            outputs=naive_pred,
            inputs=self.s_original,
            grad_outputs=torch.ones_like(naive_pred),
            retain_graph=True,
            create_graph=False,  # True to compute higher order derivatives, and is more expensive.
        )[0].view(-1, 1)
        coeffs_proj = self.coeff_head(naive_pred) + naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(s_wo_bias, coeffs_proj, grad_s, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNTransformerSeqNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerSeqNN, self).__init__(trainer)
        self.sn = SN()
        self.transformer = TransformerSeqNN(
            n_inputs=n_inputs,
            n_outputs=1,
            layers=layers,
            trainer=trainer,
            **kwargs,
        )
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1,
            n_outputs=sum(self.sn.n_coeff_ls) + len(self.sn.n_coeff_ls),
            act_func=nn.ReLU,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        self.s_original.requires_grad_()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.transformer(x, derived_tensors)
        self._naive_pred = naive_pred
        grad_s = torch.autograd.grad(
            outputs=naive_pred,
            inputs=self.s_original,
            grad_outputs=torch.ones_like(naive_pred),
            retain_graph=True,
            create_graph=False,  # True to compute higher order derivatives, and is more expensive.
        )[0].view(-1, 1)
        coeffs_proj = self.coeff_head(naive_pred) + naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(s_wo_bias, coeffs_proj, grad_s, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNTransformerAugNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerAugNN, self).__init__(trainer)
        self.sn = SN()
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1,
            n_outputs=sum(self.sn.n_coeff_ls) + len(self.sn.n_coeff_ls),
            act_func=nn.ReLU,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_one_slip = trainer.get_var_change("Relative Maximum Stress", value=1)
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")
        self.stress_loss = StressGradLoss()

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        self.s_original.requires_grad_()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.transformer(x, derived_tensors)
        self._naive_pred = naive_pred
        grad_s = torch.autograd.grad(
            outputs=naive_pred,
            inputs=self.s_original,
            grad_outputs=torch.ones_like(naive_pred),
            retain_graph=True,
            create_graph=False,  # True to compute higher order derivatives, and is more expensive.
        )[0].view(-1, 1)
        coeffs_proj = self.coeff_head(naive_pred) + naive_pred
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(s_wo_bias, coeffs_proj, grad_s, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        s_aug = torch.zeros_like(self.s_original).uniform_(
            self.s_zero_slip, self.s_one_slip
        )
        s_aug.requires_grad_()
        data_aug = [x.clone() for x in data]
        data_aug[0][:, self.s_idx] = s_aug
        y_pred_aug = model(*data_aug)
        loss = self.stress_loss(y_pred=y_pred_aug, s=s_aug, base_loss=loss)
        return loss


class SNTransformerLRNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerLRNN, self).__init__(trainer)
        from .sn_lr import SN as lrSN

        self.sn = lrSN()
        self.transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=1,
            trainer=trainer,
            **kwargs,
        )
        self.n_coeff_ls = self.sn.n_coeff_ls
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1,
            n_outputs=sum(self.n_coeff_ls) + len(self.n_coeff_ls),
            act_func=nn.ReLU,
        )
        self.s_zero_slip = trainer.get_zero_slip("Relative Maximum Stress")
        self.s_idx = self.cont_feature_names.index("Relative Maximum Stress")

    def _forward(self, x, derived_tensors):
        self.s_original = x[:, self.s_idx].clone()
        x[:, self.s_idx] = self.s_original  # enable gradient wrt a single column
        naive_pred = self.transformer(x, derived_tensors)
        self._naive_pred = naive_pred
        coeffs_proj = self.coeff_head(naive_pred) + naive_pred
        sn_coeffs, sn_weights = coeffs_proj.split(
            [sum(self.n_coeff_ls), len(self.n_coeff_ls)], dim=1
        )
        sn_coeffs = sn_coeffs.split(self.n_coeff_ls, dim=1)
        s_wo_bias = x[:, self.s_idx] - self.s_zero_slip
        x_out = self.sn(s_wo_bias, sn_coeffs, sn_weights, naive_pred)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = (loss + self.default_loss_fn(self._naive_pred, y_true)) / 2
        return loss


class SNTransformerLRKMeansNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
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
            n_clusters=5, n_input=len(self.cluster_features), layers=layers
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


class CategoryEmbeddingNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        trainer,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        embed_dropout=0.1,
        mlp_dropout=0.0,
        **kwargs,
    ):
        super(CategoryEmbeddingNN, self).__init__(trainer)

        self.embed = Embedding1d(
            embedding_dim,
            embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.linear = get_sequential(
            [128, 64, 32],
            n_inputs=n_inputs
            + len(cat_num_unique) * embedding_dim * self.embed.run_cat,
            n_outputs=n_outputs,
            act_func=nn.ReLU,
            dropout=mlp_dropout,
            use_norm=False,
        )

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x_embed = self.embed(x, derived_tensors)
        output = self.linear(x_embed)
        return output


class SNCatEmbedLRKMeansNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
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
            n_clusters=5, n_input=len(self.cluster_features), layers=layers
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
