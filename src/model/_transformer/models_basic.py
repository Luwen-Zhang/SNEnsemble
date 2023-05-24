from typing import List, Dict
from ..base import get_sequential, AbstractNN
from .common.cont_cat_embedding import Embedding, Embedding1d
from .common.fttransformer import FTTransformer
import torch
from torch import nn


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
