from ..base import get_sequential, AbstractNN
from typing import List
from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .lstm import LSTM
from .seq_fttransformer import SeqFTTransformer
import torch
from torch import nn


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
        embed_dropout=0.1,
        attn_ff_dim=256,
        attn_dropout=0.1,
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
        if self.lstm.run:
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
            use_torch_transformer=True,
            cls_token=False,
            force_mean=True,
        )

        if self.seq_transformer.run or n_outputs != 1:
            self.w = get_sequential(
                layers,
                (1 + int(self.seq_transformer.run)) * n_outputs,
                n_outputs,
                nn.ReLU,
                norm_type="layer",
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_trans = self.transformer(x, derived_tensors)
        all_res = [x_trans]

        x_seq = self.seq_transformer(x, derived_tensors)
        if self.seq_transformer.run:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class CatEmbedSeqNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        embed_dropout=0.1,
        mlp_dropout=0.0,
        seq_embedding_dim=16,
        attn_ff_dim=256,
        seq_attn_layers=4,
        seq_attn_heads=8,
        seq_attn_dropout=0.1,
    ):
        super(CatEmbedSeqNN, self).__init__(trainer)
        self.catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
            cat_num_unique=cat_num_unique,
            embedding_dim=embedding_dim,
            embed_dropout=embed_dropout,
            mlp_dropout=mlp_dropout,
        )

        self.seq_transformer = SeqFTTransformer(
            n_inputs=None,  # not needed
            attn_heads=seq_attn_heads,
            attn_layers=seq_attn_layers,
            embedding_dim=seq_embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=layers,
            dropout=seq_attn_dropout,
            n_outputs=n_outputs,
            run="Lay-up Sequence" in self.derived_feature_names
            and "Number of Layers" in self.derived_feature_names,
            use_torch_transformer=True,
            cls_token=False,
            force_mean=True,
        )

        if self.seq_transformer.run or n_outputs != 1:
            self.w = get_sequential(
                layers,
                (1 + int(self.seq_transformer.run)) * n_outputs,
                n_outputs,
                nn.ReLU,
                norm_type="layer",
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_catembed = self.catembed(x, derived_tensors)
        all_res = [x_catembed]

        x_seq = self.seq_transformer(x, derived_tensors)
        if self.seq_transformer.run:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output
