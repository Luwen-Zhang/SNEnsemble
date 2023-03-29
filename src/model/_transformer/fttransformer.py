from torch import nn
from typing import *
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


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
