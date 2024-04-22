from tabensemb.model.base import get_sequential
from typing import List, Union
import torch
from torch import nn
from tabensemb.model.base import AbstractNN
import numpy as np


class AbstractLayupModel(AbstractNN):
    def __init__(
        self,
        layers,
        datamodule,
        seq_model: Union[AbstractNN, nn.Module],
        cont_cat_model: AbstractNN = None,
        **kwargs,
    ):
        super(AbstractLayupModel, self).__init__(datamodule, **kwargs)
        if not (
            "Lay-up Sequence" in self.derived_feature_names
            and "Number of Layers" in self.derived_feature_names
        ):
            raise Exception("Add LayUpSequenceDeriver to data_derivers")
        self.seq_model = seq_model
        self.cont_cat_model = cont_cat_model
        self.use_hidden_rep, hidden_rep_dim = self._test_required_model(
            self.n_inputs, self.cont_cat_model
        )
        self.hidden_rep_dim = hidden_rep_dim + self.seq_model.hidden_rep_dim
        self.w = get_sequential(
            layers=layers,
            n_inputs=self.hidden_rep_dim,
            n_outputs=self.n_outputs,
            act_func=nn.ReLU,
            dropout=self.hparams.dropout,
        )

    def _forward(self, x, derived_tensors):
        x_contcat = self.call_required_model(self.cont_cat_model, x, derived_tensors)
        if self.use_hidden_rep:
            hidden = self.get_hidden_state(self.cont_cat_model, x, derived_tensors)
        else:
            hidden = torch.concat([x, x_contcat], dim=1)

        seq_hidden = self.get_hidden_state(
            self.seq_model, x, derived_tensors, model_name="TransformerLayup"
        )

        output = torch.concat([hidden, seq_hidden], dim=1)
        self.hidden_representation = output
        output = self.w(output) + x_contcat
        return output


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(AbstractNN):
    def __init__(
        self,
        attn_heads,
        attn_layers,
        embedding_dim,
        ff_dim,
        ff_layers,
        dropout,
        n_outputs,
        **kwargs,
    ):
        super(Transformer, self).__init__(**kwargs)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attn_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5,  # the default value of nn.LayerNorm
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=attn_layers
        )

        self.transformer_head = get_sequential(
            ff_layers,
            embedding_dim,
            n_outputs,
            nn.ReLU,
            norm_type="layer",
            dropout=0,
        )
        self.hidden_rep_dim = embedding_dim
        self.hidden_representation = None

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x)
        x_trans = x_trans.mean(1)
        self.hidden_representation = x_trans
        x_trans = self.transformer_head(x_trans)
        return x_trans


class TransformerLayup(Transformer):
    def __init__(self, **kwargs):
        AbstractNN.__init__(self, **kwargs)
        embedding_dim = self.hparams.seq_embedding_dim
        dropout = self.hparams.seq_attn_dropout
        super(TransformerLayup, self).__init__(
            embedding_dim=embedding_dim,
            dropout=dropout,
            ff_dim=self.hparams.attn_ff_dim,
            ff_layers=self.hparams.layers,
            n_outputs=self.n_outputs,
            attn_heads=self.hparams.seq_attn_heads,
            attn_layers=self.hparams.seq_attn_layers,
            **kwargs,
        )
        self.embedding = nn.Embedding(
            num_embeddings=191, embedding_dim=embedding_dim, padding_idx=190
        )
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)

    def _forward(self, x, derived_tensors):
        device = torch.device(x.get_device()) if x.get_device() != -1 else "cpu"
        seq = derived_tensors["Lay-up Sequence"].long()
        lens = derived_tensors["Number of Layers"].long()
        max_len = seq.size(1)
        # for the definition of padding_mask, see nn.MultiheadAttention.forward
        padding_mask = (
            torch.arange(max_len, device=device).expand(len(lens), max_len) >= lens
        )
        x_seq = self.embedding(seq.long() + 90)
        x_pos = self.pos_encoding(x_seq)
        x_trans = self.transformer(x_pos, src_key_padding_mask=padding_mask)
        x_trans = x_trans.mean(1)
        self.hidden_representation = x_trans
        x_trans = self.transformer_head(x_trans)
        return x_trans
