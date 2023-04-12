from .cont_cat_embedding import Embedding
from .fttransformer import FTTransformer
from .fasttransformer import FastFormer
from .lstm import LSTM
from .loss import BiasLoss, ConsGrad
from .seq_fastformer import SeqFastFormer
from .seq_fttransformer import SeqFTTransformer
from .sn import SN
from ..base import get_sequential, AbstractNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


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
        attn_ff_dim=256,
        attn_dropout=0.1,
        use_torch_transformer=False,
        flatten_transformer=True,
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
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=[],
            dropout=attn_dropout,
            n_outputs=n_outputs,
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
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
        flatten_transformer=True,
        embed_dropout=0.1,
        attn_ff_dim=256,
        attn_dropout=0.1,
        use_torch_transformer=False,
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
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
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
        if x_lstm is not None:
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
        use_torch_transformer=False,
        flatten_transformer=True,
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
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
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
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x, derived_tensors)
        all_res = [x_trans]

        x_seq = self.seq_transformer(x, derived_tensors)
        if x_seq is not None:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class BiasTransformerSeqNN(TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = BiasLoss(self.training, loss, w)
        return loss


class ConsGradTransformerSeqNN(TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class BiasConsGradTransformerSeqNN(TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = BiasLoss(self.training, loss, w)
        loss = ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class CatEmbedLSTMNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        embed_continuous=False,
        cat_num_unique: List[int] = None,
        embedding_dim=10,
        lstm_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        embed_dropout=0.1,
    ):
        super(CatEmbedLSTMNN, self).__init__(trainer)

        self.embed = Embedding(
            embedding_dim=embedding_dim,
            n_inputs=n_inputs,
            embed_dropout=embed_dropout,
            cat_num_unique=cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
            embed_cont=embed_continuous,
            cont_encoder_layers=layers,
        )
        self.embed_encoder = get_sequential(
            layers,
            n_inputs=embedding_dim,
            n_outputs=1,
            act_func=nn.ReLU,
            dropout=embed_dropout,
            norm_type="layer",
        )
        self.lstm = LSTM(
            n_hidden,
            lstm_embedding_dim,
            lstm_layers,
            run="Number of Layers" in self.derived_feature_names,
        )

        self.w = get_sequential(
            layers,
            n_inputs + int(self.embed.run_cat) * self.n_cat + int(self.lstm.run),
            n_outputs,
            nn.ReLU,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        if type(x_embed) == tuple:
            # x_cont is encoded, x_cat is embedded.
            x_cont, x_cat = x_embed
            x_cat_encode = self.embed_encoder(x_cat).squeeze(2)
            x_embed_encode = torch.cat([x_cont, x_cat_encode], dim=1)
        elif x_embed.ndim == 3:
            # x_cont and x_cat (if exists) are embedded.
            x_embed_encode = self.embed_encoder(x_embed).squeeze(2)
        else:
            # x_cont is encoded, x_cat does not exists.
            x_embed_encode = x_embed
        all_res = [x_embed_encode]

        x_lstm = self.lstm(x, derived_tensors)
        if x_lstm is not None:
            all_res += [x_lstm]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class BiasCatEmbedLSTMNN(CatEmbedLSTMNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = BiasLoss(self.training, loss, w)
        return loss


class FastFormerNN(AbstractNN):
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
        attn_ff_dim=256,
        attn_dropout=0.1,
        flatten_transformer=True,
    ):
        super(FastFormerNN, self).__init__(trainer)

        self.embed = Embedding(
            embedding_dim,
            n_inputs,
            embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = FastFormer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=[],
            dropout=attn_dropout,
            n_outputs=n_outputs,
            flatten_transformer=flatten_transformer,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        return x_trans


class FastFormerSeqNN(AbstractNN):
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
        flatten_transformer=True,
    ):
        super(FastFormerSeqNN, self).__init__(trainer)
        self.transformer = FastFormerNN(
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
            flatten_transformer=flatten_transformer,
        )
        self.seq_transformer = SeqFastFormer(
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
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x, derived_tensors)
        all_res = [x_trans]

        x_seq = self.seq_transformer(x, derived_tensors)
        if x_seq is not None:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class ConsGradFastFormerSeqNN(FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class BiasFastFormerSeqNN(FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = BiasLoss(self.training, loss, w)
        return loss


class BiasConsGradFastFormerSeqNN(FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = BiasLoss(self.training, loss, w)
        loss = ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
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
        s = x[:, self.s_idx] - self.s_zero_slip
        coeffs = self.transformer(x, derived_tensors)
        self._coeffs = coeffs
        coeffs_proj = self.coeff_head(coeffs) + coeffs
        x_out = self.sn(s, coeffs_proj)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        if self.training:
            loss = (loss + self.default_loss_fn(self._coeffs, y_true)) / 2
        return loss


class SNTransformerAddGradSeqNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer, **kwargs):
        if n_outputs != 1:
            raise Exception("n_outputs > 1 is not supported.")
        super(SNTransformerAddGradSeqNN, self).__init__(trainer)
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
        s = x[:, self.s_idx] - self.s_zero_slip
        coeffs = self.transformer(x, derived_tensors)
        self._coeffs = coeffs
        grad = torch.autograd.grad(
            outputs=coeffs,
            inputs=x,
            grad_outputs=torch.ones_like(coeffs),
            retain_graph=True,
            create_graph=False,  # True to compute higher order derivatives, and is more expensive.
        )[0]
        grad_s = grad[:, self.s_idx]
        approx_b = torch.mul(F.relu(grad_s), s)
        coeffs_proj = self.coeff_head(coeffs) + coeffs
        coeffs_proj[:, 0] += grad_s
        coeffs_proj[:, 2] += grad_s
        coeffs_proj[:, 1] += approx_b
        coeffs_proj[:, 3] += approx_b

        x_out = self.sn(s, coeffs_proj)
        return x_out

    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        if self.training:
            loss = (loss + self.default_loss_fn(self._coeffs, y_true)) / 2
        return loss
