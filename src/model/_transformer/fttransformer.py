from torch import nn
from typing import *
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from ..base import get_sequential


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The size of the last dimension must be even.")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_heads,
        ff_dim,
        dropout,
        activation,
    ):
        super(TransformerBlock, self).__init__()
        if ff_dim % 2 != 0:
            raise Exception(f"transformer_ff_dim should be an even number.")
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        is_reglu = activation == ReGLU
        self.f = nn.Sequential(
            OrderedDict(
                [
                    ("first_linear", nn.Linear(embed_dim, ff_dim)),
                    ("activation", activation()),
                    ("dropout", nn.Dropout(dropout)),
                    (
                        "second_linear",
                        nn.Linear(
                            ff_dim // 2 if is_reglu else ff_dim,
                            embed_dim,
                        ),
                    ),
                ]
            )
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        normed_x = self.attn_norm(x)
        x_attn, _ = self.attn(
            normed_x, normed_x, normed_x, key_padding_mask=key_padding_mask
        )
        x_attn = x + self.dropout(x_attn)
        return x + self.dropout(self.f(self.f_norm(x_attn)))


class TransformerEncoder(nn.Module):
    def __init__(self, attn_layers, embed_dim, attn_heads, ff_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.Sequential()
        for i in range(attn_layers):
            self.transformer.add_module(
                f"block_{i}",
                TransformerBlock(
                    embed_dim=embed_dim,
                    attn_heads=attn_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=ReGLU,
                ),
            )

    def forward(self, x):
        return self.transformer(x)


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_inputs,
        attn_heads,
        attn_layers,
        embedding_dim,
        ff_dim,
        ff_layers,
        dropout,
        n_outputs,
        use_torch_transformer,
        flatten_transformer,
        **kwargs,
    ):
        super(FTTransformer, self).__init__()
        # Indeed, the implementation of TransformerBlock is almost the same as torch.nn.TransformerEncoderLayer, except
        # that the activation function in FT-Transformer is ReGLU instead of ReLU or GeLU in torch implementation.
        # The performance of these two implementations can be verified after several epochs by changing
        # ``use_torch_transformer`` and setting the activation of TransformerBlock to nn.GELU.
        # In our scenario, ReGLU performs much better, which is why we implement our own version of transformer, just
        # like FT-Transformer and WideDeep do.
        # Also, dropout in MultiheadAttention improves performance.
        self.flatten_transformer = flatten_transformer
        if use_torch_transformer:
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
        else:
            self.transformer = TransformerEncoder(
                attn_layers=attn_layers,
                embed_dim=embedding_dim,
                attn_heads=attn_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        self.transformer_head = get_sequential(
            ff_layers,
            n_inputs * embedding_dim if self.flatten_transformer else embedding_dim,
            n_outputs,
            nn.Identity if len(ff_layers) == 0 else nn.ReLU,
            use_norm=False if len(ff_layers) == 0 else True,
            dropout=0,
        )

    def forward(self, x, derived_tensors):
        x_trans = self.transformer(x)
        x_trans = x_trans.flatten(1) if self.flatten_transformer else x_trans.mean(1)
        x_trans = self.transformer_head(x_trans)
        return x_trans
