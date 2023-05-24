from torch import nn, einsum
from typing import *
import torch
import einops
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from src.model.base import get_sequential


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The size of the last dimension must be even.")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class PositionWiseFeedForward(nn.Module):
    # Reference:
    # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feed_forward.py

    # Pytorch-Tabular uses this implementation in the transformer block.
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)


class FeedForward(nn.Module):
    # This is the original ff used in the official implementation
    # https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py
    # and also in Pytorch-Widedeep

    def __init__(self, activation, embed_dim, ff_dim, dropout):
        super(FeedForward, self).__init__()
        is_reglu = activation == ReGLU
        self.f = nn.Sequential(
            OrderedDict(
                [
                    (
                        "first_linear",
                        nn.Linear(embed_dim, ff_dim * 2 if is_reglu else ff_dim),
                    ),
                    ("activation", activation()),
                    ("dropout", nn.Dropout(dropout)),
                    (
                        "second_linear",
                        nn.Linear(ff_dim, embed_dim),
                    ),
                ]
            )
        )

    def forward(self, x):
        return self.f(x)


class AppendCLSToken(nn.Module):
    # Reference: pytorch_tabular https://github.com/manujosephv/pytorch_tabular
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        nn.init.normal_(self.weight, std=1 / np.sqrt(d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


class LinearAttention(nn.Module):
    # Reference: pytorch_tabular https://github.com/manujosephv/pytorch_tabular
    # pytorch_tabular version of linear attention

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: int = 0.1,
        keep_attn: bool = True,
    ):
        super().__init__()
        assert (
            input_dim % num_heads == 0
        ), "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim**-0.5
        self.keep_attn = keep_attn

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        sim = torch.mul(einsum("b h i d, b h j d -> b h i j", q, k), self.scale)
        if key_padding_mask is not None:
            # https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py#L204
            # https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/functional.py#L5141
            mask = torch.zeros_like(key_padding_mask, dtype=q.dtype, device=q.device)
            mask.masked_fill_(key_padding_mask, float("-inf"))
            sim = sim + mask[:, None, :, None]
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if self.keep_attn:
            self.attn_weights = attn
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_heads,
        ff_dim,
        dropout,
        activation,
        linear_attn=True,
    ):
        super(TransformerBlock, self).__init__()
        if ff_dim % 2 != 0:
            raise Exception(f"transformer_ff_dim should be an even number.")
        self.linear_attn = linear_attn
        if linear_attn:
            self.attn = LinearAttention(
                input_dim=embed_dim,
                num_heads=attn_heads,
                head_dim=embed_dim,
                dropout=dropout,
                keep_attn=False,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
                bias=False,
            )
        if activation == nn.GELU:
            self.f = PositionWiseFeedForward(
                d_model=embed_dim,
                d_ff=ff_dim,
                dropout=dropout,
                activation=activation(),
                is_gated=True,
                bias_gate=False,
                bias1=False,
                bias2=False,
            )
        else:
            self.f = FeedForward(
                activation=activation,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        if self.linear_attn:
            x_attn = self.attn(x, key_padding_mask=key_padding_mask)
        else:
            x_attn, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x_attn = self.attn_norm(x + self.dropout(x_attn))
        return self.f_norm(x + self.dropout(self.f(x_attn)))


class TransformerEncoder(nn.Module):
    def __init__(self, attn_layers, embed_dim, attn_heads, ff_dim, dropout):
        super(TransformerEncoder, self).__init__()
        attn_ls = []
        for i in range(attn_layers):
            attn_ls.append(
                TransformerBlock(
                    embed_dim=embed_dim,
                    attn_heads=attn_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=nn.GELU,
                ),
            )
        self.transformer = nn.ModuleList(attn_ls)

    def forward(self, x, src_key_padding_mask=None):
        for attn_layer in self.transformer:
            x = attn_layer(x, key_padding_mask=src_key_padding_mask)
        return x


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
        use_torch_transformer=False,
        cls_token=True,
        force_mean=False,
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
        self.cls_token = cls_token
        # cls_token is used in pytorch_tabular but not in pytorch_widedeep.
        self.force_mean = force_mean
        if cls_token:
            self.add_cls = AppendCLSToken(d_token=embedding_dim)
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
        # The head in pytorch_tabular is nn.Linear, but in pytorch_widedeep it is MLP.
        if len(ff_layers) == 0:
            self.transformer_head = nn.Linear(
                in_features=n_inputs * embedding_dim
                if not self.cls_token
                else embedding_dim,
                out_features=n_outputs,
            )
        else:
            self.transformer_head = get_sequential(
                ff_layers,
                n_inputs * embedding_dim
                if not self.cls_token and not force_mean
                else embedding_dim,
                n_outputs,
                nn.ReLU,
                norm_type="layer",
                dropout=0,
            )

    def forward(self, x, derived_tensors):
        if self.cls_token:
            x = self.add_cls(x)
        x_trans = self.transformer(x)
        if self.cls_token:
            x_trans = x_trans[:, -1, :]
        elif self.force_mean:
            x_trans = x_trans.mean(1)
        else:
            x_trans = x_trans.flatten(1)
        x_trans = self.transformer_head(x_trans)
        return x_trans
