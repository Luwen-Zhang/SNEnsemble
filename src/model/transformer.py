import src
from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel, AbstractNN
from .base import get_sequential
import torch.nn as nn
from typing import *
from ._transformer.fttransformer import PositionalEncoding, TransformerEncoder
from ._transformer.fasttransformer import FastformerEncoder


class Transformer(TorchModel):
    def __init__(
        self,
        trainer,
        manual_activate_sn=None,
        layers=None,
        *args,
        **kwargs,
    ):
        super(Transformer, self).__init__(trainer, *args, **kwargs)
        self.manual_activate_sn = manual_activate_sn
        self.layers = layers

    def _get_program_name(self):
        return "Transformer"

    def _get_model_names(self):
        return [
            "FastFormer",
            "FastFormerSeq",
            "BiasFastFormerSeq",
            "ConsGradFastFormerSeq",
            "BiasConsGradFastFormerSeq",
            "FTTransformer",
            "TransformerLSTM",
            "TransformerSeq",
            "BiasTransformerSeq",
            "ConsGradTransformerSeq",
            "BiasConsGradTransformerSeq",
            "CatEmbedLSTM",
            "BiasCatEmbedLSTM",
        ]

    def _new_model(self, model_name, verbose, **kwargs):
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, t in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][t] == "Material"
        ]
        if model_name == "TransformerLSTM":
            return _TransformerLSTMNN(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.args["layers"] if self.layers is None else self.layers,
                trainer=self.trainer,
                manual_activate_sn=self.manual_activate_sn,
                sn_coeff_vars_idx=sn_coeff_vars_idx,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                seq_embedding_dim=kwargs["seq_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                n_hidden=kwargs["n_hidden"],
                lstm_layers=kwargs["lstm_layers"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                embed_dropout=kwargs["embed_dropout"],
                attn_dropout=kwargs["attn_dropout"],
            )
        elif model_name in [
            "FTTransformer",
            "FastFormer",
        ]:
            cls = getattr(sys.modules[__name__], f"_{model_name}NN")
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                trainer=self.trainer,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                embedding_dim=kwargs["embedding_dim"],
                embed_dropout=kwargs["embed_dropout"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                attn_dropout=kwargs["attn_dropout"],
            )
        elif model_name in [
            "FastFormerSeq",
            "BiasFastFormerSeq",
            "ConsGradFastFormerSeq",
            "BiasConsGradFastFormerSeq",
            "TransformerSeq",
            "BiasTransformerSeq",
            "ConsGradTransformerSeq",
            "BiasConsGradTransformerSeq",
        ]:
            cls = getattr(sys.modules[__name__], f"_{model_name}NN")
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.args["layers"] if self.layers is None else self.layers,
                trainer=self.trainer,
                manual_activate_sn=self.manual_activate_sn,
                sn_coeff_vars_idx=sn_coeff_vars_idx,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                seq_embedding_dim=kwargs["seq_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                embed_dropout=kwargs["embed_dropout"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                attn_dropout=kwargs["attn_dropout"],
                seq_attn_layers=kwargs["seq_attn_layers"],
                seq_attn_heads=kwargs["seq_attn_heads"],
                seq_attn_dropout=kwargs["seq_attn_dropout"],
            )
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            cls = getattr(sys.modules[__name__], f"_{model_name}NN")
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.args["layers"] if self.layers is None else self.layers,
                trainer=self.trainer,
                manual_activate_sn=self.manual_activate_sn,
                sn_coeff_vars_idx=sn_coeff_vars_idx,
                embed_continuous=True,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                lstm_embedding_dim=kwargs["lstm_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                n_hidden=kwargs["n_hidden"],
                lstm_layers=kwargs["lstm_layers"],
                embed_dropout=kwargs["embed_dropout"],
            )

    def _space(self, model_name):
        if model_name == "TransformerLSTM":
            return [
                Categorical(categories=[2, 4, 8, 16, 32], name="seq_embedding_dim"),
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=1, high=30, prior="uniform", name="n_hidden", dtype=int),
                Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
                Integer(low=2, high=4, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "FTTransformer",
            "FastFormer",
        ]:
            return [
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=4, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in [
            "FastFormerSeq",
            "BiasFastFormerSeq",
            "ConsGradFastFormerSeq",
            "BiasConsGradFastFormerSeq",
            "TransformerSeq",
            "BiasTransformerSeq",
            "ConsGradTransformerSeq",
            "BiasConsGradTransformerSeq",
        ]:
            return [
                # `seq_embedding_dim` should be able to divided by `attn_heads`.
                Categorical(categories=[8, 16, 32], name="seq_embedding_dim"),
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=2, high=4, prior="uniform", name="attn_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="attn_dropout"),
                Integer(
                    low=2, high=4, prior="uniform", name="seq_attn_layers", dtype=int
                ),
                Categorical(categories=[2, 4, 8], name="seq_attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="seq_attn_dropout"),
            ] + self.trainer.SPACE
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            return [
                Categorical(categories=[2, 4, 8, 16, 32], name="lstm_embedding_dim"),
                Categorical(categories=[8, 16, 32], name="embedding_dim"),
                Integer(low=1, high=30, prior="uniform", name="n_hidden", dtype=int),
                Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        if model_name == "TransformerLSTM":
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "n_hidden": 10,
                "lstm_layers": 1,
                "attn_layers": 4,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in [
            "FTTransformer",
            "FastFormer",
        ]:
            return {
                "embedding_dim": 32,
                "attn_layers": 4,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in [
            "FastFormerSeq",
            "BiasFastFormerSeq",
            "ConsGradFastFormerSeq",
            "BiasConsGradFastFormerSeq",
            "TransformerSeq",
            "BiasTransformerSeq",
            "ConsGradTransformerSeq",
            "BiasConsGradTransformerSeq",
        ]:
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 32,
                "attn_layers": 4,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "attn_dropout": 0.2,
                "seq_attn_layers": 4,
                "seq_attn_heads": 8,
                "seq_attn_dropout": 0.2,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            return {
                "lstm_embedding_dim": 16,  # bayes-opt: 1000
                "embedding_dim": 32,  # bayes-opt: 1
                "n_hidden": 10,  # bayes-opt: 1
                "lstm_layers": 1,  # bayes-opt: 1
                "embed_dropout": 0.1,
                "lr": 0.003,  # bayes-opt: 0.0218894
                "weight_decay": 0.002,  # bayes-opt: 0.05
                "batch_size": 1024,  # bayes-opt: 32
            }

    def _conditional_validity(self, model_name: str) -> bool:
        if model_name in ["ConsGradTransformerSeq"] and not src.check_grad_in_loss():
            return False
        return True


class _FTTransformerNN(AbstractNN):
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
        attn_ff_dim=256,
        attn_dropout=0.1,
        use_torch_transformer=False,
        flatten_transformer=True,
    ):
        super(_FTTransformerNN, self).__init__(trainer)
        self.n_cont = n_inputs
        self.n_outputs = n_outputs
        self.n_cat = len(cat_num_unique) if cat_num_unique is not None else 0

        self.embed = _Embedding(
            embedding_dim,
            n_inputs,
            embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = _FTTransformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=[],
            dropout=attn_dropout,
            n_outputs=n_outputs,
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        return x_trans


class _TransformerLSTMNN(_FTTransformerNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
        cat_num_unique: List[int] = None,
        embedding_dim=32,
        seq_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        attn_layers=4,
        attn_heads=8,
        flatten_transformer=True,
        embed_dropout=0.1,
        attn_ff_dim=256,
        attn_dropout=0.1,
        use_torch_transformer=False,
    ):
        super(_TransformerLSTMNN, self).__init__(
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
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
        )

        self.lstm = _LSTM(
            n_hidden,
            seq_embedding_dim,
            lstm_layers,
            run="Number of Layers" in self.derived_feature_names,
        )
        self.sn = _SN(trainer, manual_activate_sn, sn_coeff_vars_idx)

        self.run_any = self.sn.run or self.lstm.run
        if self.run_any:
            self.w = get_sequential(
                layers,
                1 + int(self.sn.run) + int(self.lstm.run),
                n_outputs,
                nn.ReLU,
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.transformer(x_embed, derived_tensors)
        all_res = [x_trans]

        x_sn = self.sn(x, derived_tensors)
        if x_sn is not None:
            all_res += [x_sn]

        x_lstm = self.lstm(x, derived_tensors)
        if x_lstm is not None:
            all_res += [x_lstm]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class _TransformerSeqNN(_FTTransformerNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
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
        use_torch_transformer=False,
        flatten_transformer=True,
    ):
        super(_TransformerSeqNN, self).__init__(
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
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
        )

        self.seq_transformer = _SeqFTTransformer(
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
        )
        self.sn = _SN(trainer, manual_activate_sn, sn_coeff_vars_idx)

        self.run_any = self.sn.run or self.seq_transformer.run
        if self.run_any:
            self.w = get_sequential(
                layers,
                1 + int(self.sn.run) + int(self.seq_transformer.run),
                n_outputs,
                nn.ReLU,
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        all_res = [x_trans]

        x_sn = self.sn(x, derived_tensors)
        if x_sn is not None:
            all_res += [x_sn]

        x_seq = self.seq_transformer(x, derived_tensors)
        if x_seq is not None:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class _BiasTransformerSeqNN(_TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = _BiasLoss(self.training, loss, w)
        return loss


class _ConsGradTransformerSeqNN(_TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = _ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class _BiasConsGradTransformerSeqNN(_TransformerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = _BiasLoss(self.training, loss, w)
        loss = _ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class _CatEmbedLSTMNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
        embed_continuous=False,
        cat_num_unique: List[int] = None,
        embedding_dim=10,
        lstm_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        embed_dropout=0.1,
    ):
        super(_CatEmbedLSTMNN, self).__init__(trainer)
        self.n_cont = n_inputs
        self.n_outputs = n_outputs
        self.n_cat = len(cat_num_unique) if cat_num_unique is not None else 0

        self.embed = _Embedding(
            embedding_dim=embedding_dim,
            n_inputs=n_inputs,
            embed_dropout=embed_dropout,
            cat_num_unique=cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
            embed_cont=embed_continuous,
            cont_encoder_layers=layers,
        )
        self.embed_encoder = get_sequential(
            layers,
            n_inputs=embedding_dim,
            n_outputs=n_outputs,
            act_func=nn.ReLU,
            dropout=embed_dropout,
            norm_type="layer",
        )
        self.lstm = _LSTM(
            n_hidden,
            lstm_embedding_dim,
            lstm_layers,
            run="Number of Layers" in self.derived_feature_names,
        )
        self.sn = _SN(trainer, manual_activate_sn, sn_coeff_vars_idx)

        self.run_any = self.sn.run or self.lstm.run
        self.w = get_sequential(
            layers,
            n_inputs
            + int(self.embed.run_cat) * self.n_cat
            + int(self.sn.run)
            + int(self.lstm.run),
            n_outputs,
            nn.ReLU,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        if type(x_embed) == tuple:
            # x_cont is encoded, x_cat is embedded.
            x_cont, x_cat = x_embed
            x_cat_encode = self.embed_encoder(x_cat).squeeze(2)
            x_embed_encode = torch.cat([x_cont, x_cat_encode], dim=1)
        elif x_embed.ndim == 3:
            # x_cont and x_cat (if exists) are embedded.
            x_embed_encode = self.embed_encoder(x_embed).squeeze(2)
        else:
            # x_cont is encoded, x_cat does not exists.
            x_embed_encode = x_embed
        all_res = [x_embed_encode]

        x_sn = self.sn(x, derived_tensors)
        if x_sn is not None:
            all_res += [x_sn]

        x_lstm = self.lstm(x, derived_tensors)
        if x_lstm is not None:
            all_res += [x_lstm]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class _BiasCatEmbedLSTMNN(_CatEmbedLSTMNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = _BiasLoss(self.training, loss, w)
        return loss


class _FastFormerNN(AbstractNN):
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
        attn_ff_dim=256,
        attn_dropout=0.1,
        flatten_transformer=True,
    ):
        super(_FastFormerNN, self).__init__(trainer)
        self.n_cont = n_inputs
        self.n_outputs = n_outputs
        self.n_cat = len(cat_num_unique) if cat_num_unique is not None else 0

        self.embed = _Embedding(
            embedding_dim,
            n_inputs,
            embed_dropout,
            cat_num_unique,
            run_cat="categorical" in self.derived_feature_names,
        )
        self.embed_transformer = _FastFormer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=attn_ff_dim,
            ff_layers=[],
            dropout=attn_dropout,
            n_outputs=n_outputs,
            flatten_transformer=flatten_transformer,
        )

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        return x_trans


class _FastFormerSeqNN(_FastFormerNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
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
        flatten_transformer=True,
    ):
        super(_FastFormerSeqNN, self).__init__(
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
            flatten_transformer=flatten_transformer,
        )
        self.seq_transformer = _SeqFastFormer(
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
        )
        self.sn = _SN(trainer, manual_activate_sn, sn_coeff_vars_idx)

        self.run_any = self.sn.run or self.seq_transformer.run
        if self.run_any:
            self.w = get_sequential(
                layers,
                1 + int(self.sn.run) + int(self.seq_transformer.run),
                n_outputs,
                nn.ReLU,
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_embed = self.embed(x, derived_tensors)
        x_trans = self.embed_transformer(x_embed, derived_tensors)
        all_res = [x_trans]

        x_sn = self.sn(x, derived_tensors)
        if x_sn is not None:
            all_res += [x_sn]

        x_seq = self.seq_transformer(x, derived_tensors)
        if x_seq is not None:
            all_res += [x_seq]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class _ConsGradFastFormerSeqNN(_FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        loss = _ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class _BiasFastFormerSeqNN(_FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = _BiasLoss(self.training, loss, w)
        return loss


class _BiasConsGradFastFormerSeqNN(_FastFormerSeqNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        loss = self.default_loss_fn(y_pred, y_true)
        where_weight = self.derived_feature_names.index("Sample Weight")
        w = data[1 + where_weight]
        loss = _BiasLoss(self.training, loss, w)
        loss = _ConsGrad(
            self.training,
            base_loss=loss,
            y_pred=y_pred,
            n_cont=self.n_cont,
            cont_feature_names=self.cont_feature_names,
            *data,
        )
        return loss


class _SN(nn.Module):
    def __init__(self, trainer, manual_activate_sn, sn_coeff_vars_idx):
        super(_SN, self).__init__()
        from ._transformer.sn_formulas import sn_mapping

        activated_sn = []
        for key, sn in sn_mapping.items():
            if (
                sn.test_sn_vars(
                    trainer.cont_feature_names, list(trainer.derived_data.keys())
                )
                and (manual_activate_sn is None or key in manual_activate_sn)
                and sn.activated()
            ):
                activated_sn.append(
                    sn(
                        cont_feature_names=trainer.cont_feature_names,
                        derived_feature_names=list(trainer.derived_data.keys()),
                        s_zero_slip=trainer.get_zero_slip(sn.get_sn_vars()[0]),
                        sn_coeff_vars_idx=sn_coeff_vars_idx,
                    )
                )
        if len(activated_sn) > 0 and self._check_activate():
            print(
                f"Activated SN models: {[sn.__class__.__name__ for sn in activated_sn]}"
            )
            self.sn_coeff_vars_idx = sn_coeff_vars_idx
            self.activated_sn = nn.ModuleList(activated_sn)
            self.sn_component_weights = get_sequential(
                [16, 64, 128, 64, 16],
                len(sn_coeff_vars_idx),
                len(self.activated_sn),
                nn.ReLU,
            )
            self.run = True
        else:
            self.run = False

    def forward(self, x, derived_tensors):
        if self.run:
            x_sn = torch.concat(
                [sn(x, derived_tensors) for sn in self.activated_sn],
                dim=1,
            )
            x_sn = torch.mul(
                x_sn,
                nn.functional.normalize(
                    torch.abs(self.sn_component_weights(x[:, self.sn_coeff_vars_idx])),
                    p=1,
                    dim=1,
                ),
            )
            x_sn = torch.sum(x_sn, dim=1).view(-1, 1)
            return x_sn
        else:
            return None

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"SN module is manually deactivated.")
            return False

    def _manual_activate(self):
        return False


class _Embedding(nn.Module):
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
        super(_Embedding, self).__init__()
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


class _LSTM(nn.Module):
    def __init__(self, n_hidden, embedding_dim, layers, run):
        super(_LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm_embedding_dim = embedding_dim
        self.lstm_layers = layers
        if run and self._check_activate():
            self.seq_lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=n_hidden,
                num_layers=layers,
                batch_first=True,
            )
            # The input degree would be in range [-90, 100] (where 100 is the padding value). It will be transformed to
            # [0, 190] by adding 100, so the number of categories (vocab) will be 191
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=embedding_dim
            )
            self.run = True
        else:
            self.run = False

    def forward(self, x, derived_tensors):
        if self.run:
            seq = derived_tensors["Lay-up Sequence"]
            lens = derived_tensors["Number of Layers"]
            device = "cpu" if seq.get_device() == -1 else seq.get_device()
            h_0 = torch.zeros(
                self.lstm_layers, seq.size(0), self.n_hidden, device=device
            )
            c_0 = torch.zeros(
                self.lstm_layers, seq.size(0), self.n_hidden, device=device
            )

            seq_embed = self.embedding(seq.long() + 90)
            seq_packed = nn.utils.rnn.pack_padded_sequence(
                seq_embed,
                torch.flatten(lens.cpu()),
                batch_first=True,
                enforce_sorted=False,
            )
            # We don't need all hidden states for all hidden LSTM cell (which is the first returned value), but only
            # the last hidden state.
            _, (h_t, _) = self.seq_lstm(seq_packed, (h_0, c_0))
            return torch.mean(h_t, dim=[0, 2]).view(-1, 1)
        else:
            return None

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"LSTM module is manually deactivated.")
            return False

    def _manual_activate(self):
        return False


class _FastFormer(nn.Module):
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
        **kwargs,
    ):
        super(_FastFormer, self).__init__()
        self.transformer = FastformerEncoder(
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embed_dim=embedding_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            max_position_embeddings=256,
        )

        self.transformer_head = get_sequential(
            ff_layers,
            embedding_dim,
            n_outputs,
            nn.Identity if len(ff_layers) == 0 else nn.ReLU,
            use_norm=False if len(ff_layers) == 0 else True,
            dropout=0,
        )

    def forward(self, x, derived_tensors):
        x_trans = self.transformer(x)
        x_trans = self.transformer_head(x_trans)
        return x_trans


class _SeqFastFormer(_FastFormer):
    def __init__(
        self,
        embedding_dim,
        dropout,
        run,
        *args,
        **kwargs,
    ):
        if run and self._check_activate():
            super(_SeqFastFormer, self).__init__(
                embedding_dim=embedding_dim,
                dropout=dropout,
                *args,
                **kwargs,
            )
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=embedding_dim
            )
            self.pos_encoding = PositionalEncoding(
                d_model=embedding_dim, dropout=dropout
            )
            self.run = True
        else:
            super(_FastFormer, self).__init__()
            self.run = False

    def forward(self, x, derived_tensors):
        if self.run:
            seq = derived_tensors["Lay-up Sequence"]
            lens = derived_tensors["Number of Layers"]
            max_len = seq.size(1)
            device = "cpu" if seq.get_device() == -1 else seq.get_device()
            # for the definition of padding_mask, see nn.MultiheadAttention.forward
            padding_mask = (
                torch.arange(max_len, device=device).expand(len(lens), max_len) >= lens
            )
            x = self.embedding(seq.long() + 90)
            x_pos = self.pos_encoding(x)
            x_trans = self.transformer(x_pos, src_key_padding_mask=padding_mask)
            x_trans = self.transformer_head(x_trans)
            return x_trans
        else:
            return None

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"SeqFastFormer module is manually deactivated.")
            return False

    def _manual_activate(self):
        return False


class _FTTransformer(nn.Module):
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
        use_torch_transformer,
        flatten_transformer,
        **kwargs,
    ):
        super(_FTTransformer, self).__init__()
        # Indeed, the implementation of TransformerBlock is almost the same as torch.nn.TransformerEncoderLayer, except
        # that the activation function in FT-Transformer is ReGLU instead of ReLU or GeLU in torch implementation.
        # The performance of these two implementations can be verified after several epochs by changing
        # `use_torch_transformer` and setting the activation of TransformerBlock to nn.GELU.
        # In our scenario, ReGLU performs much better, which is why we implement our own version of transformer, just
        # like FT-Transformer and WideDeep do.
        # Also, dropout in MultiheadAttention improves performance.
        self.flatten_transformer = flatten_transformer
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
        self.transformer_head = get_sequential(
            ff_layers,
            n_inputs * embedding_dim if self.flatten_transformer else embedding_dim,
            n_outputs,
            nn.Identity if len(ff_layers) == 0 else nn.ReLU,
            use_norm=False if len(ff_layers) == 0 else True,
            dropout=0,
        )

    def forward(self, x, derived_tensors):
        x_trans = self.transformer(x)
        x_trans = x_trans.flatten(1) if self.flatten_transformer else x_trans.mean(1)
        x_trans = self.transformer_head(x_trans)
        return x_trans


class _SeqFTTransformer(_FTTransformer):
    def __init__(
        self,
        embedding_dim,
        dropout,
        run,
        use_torch_transformer=None,
        flatten_transformer=None,
        *args,
        **kwargs,
    ):
        if run and self._check_activate():
            # flatten_transformer=False because the length of the padded sequence might be unknown.
            super(_SeqFTTransformer, self).__init__(
                embedding_dim=embedding_dim,
                dropout=dropout,
                use_torch_transformer=True,
                flatten_transformer=False,
                *args,
                **kwargs,
            )
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=embedding_dim
            )
            self.pos_encoding = PositionalEncoding(
                d_model=embedding_dim, dropout=dropout
            )
            self.run = True
        else:
            super(_FTTransformer, self).__init__()
            self.run = False

    def forward(self, x, derived_tensors):
        if self.run:
            seq = derived_tensors["Lay-up Sequence"]
            lens = derived_tensors["Number of Layers"]
            max_len = seq.size(1)
            device = "cpu" if seq.get_device() == -1 else seq.get_device()
            # for the definition of padding_mask, see nn.MultiheadAttention.forward
            padding_mask = (
                torch.arange(max_len, device=device).expand(len(lens), max_len) >= lens
            )
            x = self.embedding(seq.long() + 90)
            x_pos = self.pos_encoding(x)
            x_trans = self.transformer(x_pos, src_key_padding_mask=padding_mask)
            x_trans = x_trans.mean(1)
            x_trans = self.transformer_head(x_trans)
            return x_trans
        else:
            return None

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"SeqTransformer module is manually deactivated.")
            return False

    def _manual_activate(self):
        return False


def _BiasLoss(training, base_loss: torch.Tensor, w: torch.Tensor):
    if not training:
        return base_loss
    return (base_loss * w).mean()


def _ConsGrad(training, *data, **kwargs):
    base_loss: torch.Tensor = kwargs["base_loss"]
    y_pred: torch.Tensor = kwargs["y_pred"]
    n_cont: int = kwargs["n_cont"]
    cont_feature_names: List[str] = kwargs["cont_feature_names"]
    implemented_features = ["Relative Mean Stress"]
    if not training:
        return base_loss
    feature_idx_mapping = {
        x: cont_feature_names.index(x)
        for x in implemented_features
        if x in cont_feature_names
    }
    grad = torch.autograd.grad(
        outputs=y_pred,
        inputs=data[0],
        grad_outputs=torch.ones_like(y_pred),
        retain_graph=True,
        create_graph=False,  # True to compute higher order derivatives, and is more expensive.
    )[0]
    feature_loss = torch.zeros((n_cont,))
    for feature, idx in feature_idx_mapping.items():
        grad_feature = grad[:, idx]
        if feature == "Relative Mean Stress":
            feature_loss[idx] = torch.mean(nn.ReLU()(grad_feature) ** 2)
        else:
            raise Exception(
                f"Operation on the gradient of feature {feature} is not implemented."
            )

    base_loss = base_loss + torch.mul(torch.sum(feature_loss), 1e3)
    return base_loss
