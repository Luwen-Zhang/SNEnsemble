import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The size of the last dimension must be even.")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, attn_heads):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.f = nn.Sequential(
            OrderedDict(
                [
                    ("first_linear", nn.Linear(embed_dim, 256)),
                    ("activation", ReGLU()),
                    ("dropout", nn.Dropout(0.1)),
                    ("second_linear", nn.Linear(256 // 2, embed_dim)),
                ]
            )
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        normed_x = self.attn_norm(x)
        x_attn, _ = self.attn(normed_x, normed_x, normed_x)
        x_attn = x + self.dropout(x_attn)
        return x + self.dropout(self.f(self.f_norm(x_attn)))
