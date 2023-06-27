from typing import List, Dict
from ..base import get_sequential, get_linear, AbstractNN
from .common.cont_cat_embedding import Embedding, Embedding1d
from .common.fttransformer import FTTransformer
import torch
from torch import nn


class CategoryEmbeddingNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        super(CategoryEmbeddingNN, self).__init__(datamodule, **kwargs)

        run_cat = "categorical" in self.derived_feature_names

        self.linear = get_sequential(
            [128, 64],
            n_inputs=n_inputs
            + len(cat_num_unique) * self.hparams.embedding_dim * run_cat,
            n_outputs=32,
            act_func=nn.ReLU,
            dropout=self.hparams.mlp_dropout,
            use_norm=False,
            out_activate=True,
            out_norm_dropout=True,
        )

        self.embed = Embedding1d(
            self.hparams.embedding_dim,
            self.hparams.embed_dropout,
            cat_num_unique,
            n_inputs,
            run_cat=run_cat,
        )

        self.head = get_linear(
            n_inputs=32,
            n_outputs=n_outputs,
            nonlinearity="relu",
        )
        self.hidden_rep_dim = 32
        self.hidden_representation = None

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x_embed = self.embed(x, derived_tensors)
        output = self.linear(x_embed)
        self.hidden_representation = output
        output = self.head(output)
        return output


class FTTransformerNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        datamodule,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        super(FTTransformerNN, self).__init__(datamodule, **kwargs)

        self.embed = Embedding(
            self.hparams.embedding_dim,
            n_inputs,
            self.hparams.embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = FTTransformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + n_inputs,
            attn_heads=self.hparams.attn_heads,
            attn_layers=self.hparams.attn_layers,
            embedding_dim=self.hparams.embedding_dim,
            ff_dim=self.hparams.embedding_dim * 4,
            ff_layers=[],
            dropout=self.hparams.attn_dropout,
            n_outputs=n_outputs,
        )
        self.hidden_rep_dim = self.embed_transformer.hidden_rep_dim
        self.hidden_representation = None

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        self.hidden_representation = self.embed_transformer.hidden_representation
        return x_trans
