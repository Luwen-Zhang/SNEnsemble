from ..base import get_sequential
from typing import List, Union
from .models_basic import CategoryEmbeddingNN, FTTransformerNN
from .common.lstm import LSTM
from .common.seq_fttransformer import SeqFTTransformer
import torch
from torch import nn
from ..base import AbstractNN


class AbstractSeqModel(AbstractNN):
    def __init__(
        self,
        n_outputs,
        layers,
        trainer,
        cont_cat_model: AbstractNN,
        seq_model: Union[AbstractNN, nn.Module],
        **kwargs,
    ):
        super(AbstractSeqModel, self).__init__(trainer, **kwargs)
        self.cont_cat_model = cont_cat_model
        self.seq_model = seq_model
        if self.seq_model.run or n_outputs != 1:
            self.w = get_sequential(
                layers,
                (1 + int(self.seq_model.run)) * n_outputs,
                n_outputs,
                nn.ReLU,
                norm_type="layer",
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_contcat = self.cont_cat_model(x, derived_tensors)
        all_res = [x_contcat]

        x_seq = self.seq_model(x, derived_tensors)
        if self.seq_model.run:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output) + x_contcat
        return output


class TransformerLSTMNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, trainer, **kwargs)
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
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
            trainer=trainer,
            cont_cat_model=transformer,
            seq_model=lstm,
        )


class TransformerSeqNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, trainer, **kwargs)
        transformer = FTTransformerNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
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
            trainer=trainer,
            cont_cat_model=transformer,
            seq_model=seq_transformer,
        )


class CatEmbedSeqNN(AbstractSeqModel):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        cat_num_unique: List[int] = None,
        **kwargs,
    ):
        AbstractNN.__init__(self, trainer, **kwargs)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
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
            trainer=trainer,
            cont_cat_model=catembed,
            seq_model=seq_transformer,
        )
