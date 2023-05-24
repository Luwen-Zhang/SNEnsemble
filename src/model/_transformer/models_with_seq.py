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
    ):
        super(AbstractSeqModel, self).__init__(trainer)
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
        output = self.w(output)
        return output


class TransformerLSTMNN(AbstractSeqModel):
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
        AbstractNN.__init__(self, trainer)
        transformer = FTTransformerNN(
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

        lstm = LSTM(
            n_hidden,
            seq_embedding_dim,
            lstm_layers,
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
        AbstractNN.__init__(self, trainer)
        transformer = FTTransformerNN(
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

        seq_transformer = SeqFTTransformer(
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
        embedding_dim=32,
        embed_dropout=0.1,
        mlp_dropout=0.0,
        seq_embedding_dim=16,
        attn_ff_dim=256,
        seq_attn_layers=4,
        seq_attn_heads=8,
        seq_attn_dropout=0.1,
    ):
        AbstractNN.__init__(self, trainer)
        catembed = CategoryEmbeddingNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            trainer=trainer,
            cat_num_unique=cat_num_unique,
            embedding_dim=embedding_dim,
            embed_dropout=embed_dropout,
            mlp_dropout=mlp_dropout,
        )
        seq_transformer = SeqFTTransformer(
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
        super(CatEmbedSeqNN, self).__init__(
            n_outputs=n_outputs,
            layers=layers,
            trainer=trainer,
            cont_cat_model=catembed,
            seq_model=seq_transformer,
        )
