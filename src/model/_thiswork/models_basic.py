from typing import List, Dict
from tabensemb.model.base import get_sequential, get_linear, AbstractNN
from .common.cont_cat_embedding import Embedding, Embedding1d
from .common.fttransformer import FTTransformer
import torch
from torch import nn


class FTTransformerNN(AbstractNN):
    def __init__(self, datamodule, **kwargs):
        super(FTTransformerNN, self).__init__(datamodule, **kwargs)

        self.embed = Embedding(
            self.hparams.embedding_dim,
            self.n_inputs,
            self.hparams.embed_dropout,
            self.cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = FTTransformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_inputs,
            attn_heads=self.hparams.attn_heads,
            attn_layers=self.hparams.attn_layers,
            embedding_dim=self.hparams.embedding_dim,
            ff_dim=self.hparams.embedding_dim * 4,
            ff_layers=[],
            dropout=self.hparams.attn_dropout,
            n_outputs=self.n_outputs,
        )
        self.hidden_rep_dim = self.embed_transformer.hidden_rep_dim
        self.hidden_representation = None

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        self.hidden_representation = self.embed_transformer.hidden_representation
        return x_trans
