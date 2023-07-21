from tabensemb.model.base import get_sequential, get_linear
from typing import List, Union
from tabensemb.model.sample import CategoryEmbeddingNN
from .models_basic import FTTransformerNN
from .common.lstm import LSTM
from .common.seq_fttransformer import SeqFTTransformer
import torch
from torch import nn
from tabensemb.model.base import AbstractNN


class AbstractSeqModel(AbstractNN):
    def __init__(
        self,
        n_outputs,
        layers,
        datamodule,
        cont_cat_model: AbstractNN,
        seq_model: Union[AbstractNN, nn.Module],
        **kwargs,
    ):
        super(AbstractSeqModel, self).__init__(datamodule, **kwargs)
        self.cont_cat_model = cont_cat_model
        self.seq_model = seq_model
        self.hidden_rep_dim = (
            self.cont_cat_model.hidden_rep_dim
            + int(self.seq_model.run) * self.seq_model.hidden_rep_dim
        )
        self.w = get_linear(self.hidden_rep_dim, n_outputs, "relu")

    def _forward(self, x, derived_tensors):
        x_contcat = self.cont_cat_model(x, derived_tensors)
        all_res = [self.cont_cat_model.hidden_representation]

        x_seq = self.seq_model(x, derived_tensors)
        if self.seq_model.run:
            all_res += [self.seq_model.hidden_representation]

        output = torch.concat(all_res, dim=1)
        self.hidden_representation = output
        output = self.w(output)
        return output


class TransformerLSTMNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, datamodule, **kwargs)
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            cat_num_unique=cat_num_unique,
            embedding_dim=self.hparams.embedding_dim,
            embed_dropout=self.hparams.embed_dropout,
            attn_layers=self.hparams.attn_layers,
            attn_heads=self.hparams.attn_heads,
            attn_ff_dim=self.hparams.attn_ff_dim,
            attn_dropout=self.hparams.attn_dropout,
        )

        lstm = LSTM(
            self.hparams.n_hidden,
            self.hparams.seq_embedding_dim,
            self.hparams.lstm_layers,
            run="Number of Layers" in self.derived_feature_names,
        )
        super(TransformerLSTMNN, self).__init__(
            n_outputs=n_outputs,
            layers=layers,
            datamodule=datamodule,
            cont_cat_model=transformer,
            seq_model=lstm,
        )


class TransformerSeqNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, datamodule, **kwargs)
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            cat_num_unique=cat_num_unique,
            embedding_dim=self.hparams.embedding_dim,
            embed_dropout=self.hparams.embed_dropout,
            attn_layers=self.hparams.attn_layers,
            attn_heads=self.hparams.attn_heads,
            attn_ff_dim=self.hparams.attn_ff_dim,
            attn_dropout=self.hparams.attn_dropout,
        )

        seq_transformer = SeqFTTransformer(
            n_inputs=None,
            attn_heads=self.hparams.seq_attn_heads,
            attn_layers=self.hparams.seq_attn_layers,
            embedding_dim=self.hparams.seq_embedding_dim,
            ff_dim=self.hparams.attn_ff_dim,
            ff_layers=layers,
            dropout=self.hparams.seq_attn_dropout,
            n_outputs=n_outputs,
            run="Lay-up Sequence" in self.derived_feature_names
            and "Number of Layers" in self.derived_feature_names,
            use_torch_transformer=True,
            cls_token=False,
            force_mean=True,
        )

        super(TransformerSeqNN, self).__init__(
            n_outputs=n_outputs,
            layers=layers,
            datamodule=datamodule,
            cont_cat_model=transformer,
            seq_model=seq_transformer,
        )


class CatEmbedSeqNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        datamodule,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, datamodule, **kwargs)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            datamodule=datamodule,
            cat_num_unique=cat_num_unique,
            embedding_dim=self.hparams.embedding_dim,
            embed_dropout=self.hparams.embed_dropout,
            mlp_dropout=self.hparams.mlp_dropout,
        )
        seq_transformer = SeqFTTransformer(
            n_inputs=None,  # not needed
            attn_heads=self.hparams.seq_attn_heads,
            attn_layers=self.hparams.seq_attn_layers,
            embedding_dim=self.hparams.seq_embedding_dim,
            ff_dim=self.hparams.attn_ff_dim,
            ff_layers=layers,
            dropout=self.hparams.seq_attn_dropout,
            n_outputs=n_outputs,
            run="Lay-up Sequence" in self.derived_feature_names
            and "Number of Layers" in self.derived_feature_names,
            use_torch_transformer=True,
            cls_token=False,
            force_mean=True,
        )
        super(CatEmbedSeqNN, self).__init__(
            n_outputs=n_outputs,
            layers=layers,
            datamodule=datamodule,
            cont_cat_model=catembed,
            seq_model=seq_transformer,
        )
