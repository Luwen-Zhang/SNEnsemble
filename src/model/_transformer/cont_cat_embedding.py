import torch
from torch import nn
import numpy as np
from ..base import get_sequential


class Embedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_inputs,
        embed_dropout,
        cat_num_unique,
        run_cat,
        embed_cont=True,
        cont_encoder_layers=None,
    ):
        super(Embedding, self).__init__()
        # Module: Continuous embedding
        self.embed_cont = embed_cont
        if embed_cont:
            self.embedding_dim = embedding_dim
            self.cont_norm = nn.BatchNorm1d(n_inputs)
            self.cont_embed_weight = nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
            )
            self.cont_embed_bias = nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
            )
            self.cont_dropout = nn.Dropout(embed_dropout)
        else:
            self.cont_encoder = get_sequential(
                cont_encoder_layers,
                n_inputs,
                n_inputs,
                nn.ReLU,
            )

        # Module: Categorical embedding
        if run_cat:
            # See pytorch_widedeep.models.tabular.embeddings_layers.SameSizeCatEmbeddings
            self.cat_embeds = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=num_unique + 1,
                        embedding_dim=embedding_dim,
                        padding_idx=0,
                    )
                    for num_unique in cat_num_unique
                ]
            )
            self.cat_dropout = nn.Dropout(embed_dropout)
            self.run_cat = True
        else:
            self.run_cat = False

    def forward(self, x, derived_tensors):
        if self.embed_cont:
            x_cont = self.cont_embed_weight.unsqueeze(0) * self.cont_norm(x).unsqueeze(
                2
            ) + self.cont_embed_bias.unsqueeze(0)
            x_cont = self.cont_dropout(x_cont)
        else:
            x_cont = self.cont_encoder(x)
        if self.run_cat:
            cat = derived_tensors["categorical"].long()
            x_cat_embeds = [
                self.cat_embeds[i](cat[:, i]).unsqueeze(1) for i in range(cat.size(1))
            ]
            x_cat = torch.cat(x_cat_embeds, 1)
            x_cat = self.cat_dropout(x_cat)
            if self.embed_cont:
                x_res = torch.cat([x_cont, x_cat], dim=1)
            else:
                x_res = (x_cont, x_cat)
        else:
            x_res = x_cont
        return x_res
