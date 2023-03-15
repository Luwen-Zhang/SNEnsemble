from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel, AbstractNN
from .base import get_sequential
import torch.nn as nn
from typing import *
from torch import Tensor
import torch.nn.functional as F


class TransformerLSTM(TorchModel):
    def __init__(
        self,
        trainer=None,
        manual_activate_sn=None,
        program=None,
        layers=None,
        model_subset=None,
        **kwargs,
    ):
        super(TransformerLSTM, self).__init__(
            trainer, program=program, model_subset=model_subset, **kwargs
        )
        self.manual_activate_sn = manual_activate_sn
        self.layers = layers

    def _get_program_name(self):
        return "TransformerLSTM"

    def _get_model_names(self):
        return ["TransformerLSTM", "TransformerSeq", "CatEmbedLSTM", "BiasCatEmbedLSTM"]

    def _new_model(self, model_name, verbose, **kwargs):
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, t in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][t] == "Material"
        ]
        if model_name == "TransformerLSTM":
            return TransformerLSTMNN(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.layers if self.layers is None else self.layers,
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
                transformer_dropout=kwargs["transformer_dropout"],
            )
        elif model_name == "TransformerSeq":
            return TransformerSeqNN(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.layers if self.layers is None else self.layers,
                trainer=self.trainer,
                manual_activate_sn=self.manual_activate_sn,
                sn_coeff_vars_idx=sn_coeff_vars_idx,
                cat_num_unique=[
                    len(x) for x in self.trainer.cat_feature_mapping.values()
                ],
                seq_embedding_dim=kwargs["seq_embedding_dim"],
                embedding_dim=kwargs["embedding_dim"],
                attn_layers=kwargs["attn_layers"],
                attn_heads=kwargs["attn_heads"],
                embed_dropout=kwargs["embed_dropout"],
                transformer_dropout=kwargs["transformer_dropout"],
            )
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            if model_name == "CatEmbedLSTM":
                cls = CatEmbedLSTMNN
            else:
                cls = BiasCatEmbedLSTMNN
            return cls(
                len(self.trainer.cont_feature_names),
                len(self.trainer.label_name),
                self.trainer.layers if self.layers is None else self.layers,
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
                Categorical(
                    categories=[2, 4, 8, 16, 32, 64, 128], name="seq_embedding_dim"
                ),
                Categorical(categories=[8, 16, 32, 64], name="embedding_dim"),
                Integer(low=1, high=100, prior="uniform", name="n_hidden", dtype=int),
                Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
                Categorical(categories=[2, 4, 8], name="attn_layers"),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="transformer_dropout"),
            ] + self.trainer.SPACE
        elif model_name == "TransformerSeq":
            return [
                Categorical(
                    categories=[2, 4, 8, 16, 32, 64, 128], name="seq_embedding_dim"
                ),
                Categorical(categories=[8, 16, 32, 64], name="embedding_dim"),
                Categorical(categories=[2, 4, 8], name="attn_layers"),
                Categorical(categories=[2, 4, 8], name="attn_heads"),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
                Real(low=0.0, high=0.3, prior="uniform", name="transformer_dropout"),
            ] + self.trainer.SPACE
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            return [
                Categorical(
                    categories=[2, 4, 8, 16, 32, 64, 128], name="lstm_embedding_dim"
                ),
                Categorical(categories=[8, 16, 32, 64], name="embedding_dim"),
                Integer(low=1, high=100, prior="uniform", name="n_hidden", dtype=int),
                Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
                Real(low=0.0, high=0.3, prior="uniform", name="embed_dropout"),
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        if model_name == "TransformerLSTM":
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 64,
                "n_hidden": 10,
                "lstm_layers": 1,
                "attn_layers": 4,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "transformer_dropout": 0.1,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name == "TransformerSeq":
            return {
                "seq_embedding_dim": 16,
                "embedding_dim": 64,
                "attn_layers": 4,
                "attn_heads": 8,
                "embed_dropout": 0.1,
                "transformer_dropout": 0.1,
                "lr": 0.003,
                "weight_decay": 0.002,
                "batch_size": 1024,
            }
        elif model_name in ["CatEmbedLSTM", "BiasCatEmbedLSTM"]:
            return {
                "lstm_embedding_dim": 16,  # bayes-opt: 1000
                "embedding_dim": 64,  # bayes-opt: 1
                "n_hidden": 10,  # bayes-opt: 1
                "lstm_layers": 1,  # bayes-opt: 1
                "embed_dropout": 0.1,
                "lr": 0.003,  # bayes-opt: 0.0218894
                "weight_decay": 0.002,  # bayes-opt: 0.05
                "batch_size": 1024,  # bayes-opt: 32
            }


class TransformerLSTMNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
        cat_num_unique: List[int] = None,
        embedding_dim=64,
        seq_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        attn_layers=4,
        attn_heads=8,
        flatten_transformer=True,
        embed_dropout=0.1,
        transformer_ff_dim=256,
        transformer_dropout=0.1,
        use_torch_transformer=False,
    ):
        super(TransformerLSTMNN, self).__init__(trainer)
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
        self.transformer = _Transformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=transformer_ff_dim,
            ff_layers=[],
            dropout=transformer_dropout,
            n_outputs=n_outputs,
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


class TransformerSeqNN(AbstractNN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        layers,
        trainer,
        manual_activate_sn=None,
        sn_coeff_vars_idx=None,
        cat_num_unique: List[int] = None,
        embedding_dim=64,
        seq_embedding_dim=10,
        attn_layers=4,
        attn_heads=8,
        flatten_transformer=True,
        embed_dropout=0.1,
        transformer_ff_dim=256,
        transformer_dropout=0.1,
        use_torch_transformer=False,
    ):
        super(TransformerSeqNN, self).__init__(trainer)
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
        self.embed_transformer = _Transformer(
            n_inputs=int(self.embed.run_cat) * self.n_cat + self.n_cont,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=embedding_dim,
            ff_dim=transformer_ff_dim,
            ff_layers=[],
            dropout=transformer_dropout,
            n_outputs=n_outputs,
            use_torch_transformer=use_torch_transformer,
            flatten_transformer=flatten_transformer,
        )
        self.seq_transformer = _SeqTransformer(
            n_inputs=None,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            embedding_dim=seq_embedding_dim,
            ff_dim=transformer_ff_dim,
            ff_layers=layers,
            dropout=transformer_dropout,
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


class CatEmbedLSTMNN(AbstractNN):
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
        super(CatEmbedLSTMNN, self).__init__(trainer)
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


class BiasCatEmbedLSTMNN(CatEmbedLSTMNN):
    def loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        base_loss = self.default_loss_fn(y_pred, y_true)
        if not self.training:
            return base_loss
        else:
            where_weight = self.derived_feature_names.index("Sample Weight")
            w = data[1 + where_weight]
            return (base_loss * w).mean()


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The size of the last dimension must be even.")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_heads,
        ff_dim,
        dropout,
        activation,
    ):
        super(TransformerBlock, self).__init__()
        if ff_dim % 2 != 0:
            raise Exception(f"transformer_ff_dim should be an even number.")
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        is_reglu = activation == ReGLU
        self.f = nn.Sequential(
            OrderedDict(
                [
                    ("first_linear", nn.Linear(embed_dim, ff_dim)),
                    ("activation", activation()),
                    ("dropout", nn.Dropout(dropout)),
                    (
                        "second_linear",
                        nn.Linear(
                            ff_dim // 2 if is_reglu else ff_dim,
                            embed_dim,
                        ),
                    ),
                ]
            )
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        normed_x = self.attn_norm(x)
        x_attn, _ = self.attn(
            normed_x, normed_x, normed_x, key_padding_mask=key_padding_mask
        )
        x_attn = x + self.dropout(x_attn)
        return x + self.dropout(self.f(self.f_norm(x_attn)))


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


class _SN(nn.Module):
    def __init__(self, trainer, manual_activate_sn, sn_coeff_vars_idx):
        super(_SN, self).__init__()
        from src.model._thiswork_sn_formulas import sn_mapping

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
        return True


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
            h_0 = torch.zeros(self.lstm_layers, seq.size(0), self.n_hidden)
            c_0 = torch.zeros(self.lstm_layers, seq.size(0), self.n_hidden)

            seq_embed = self.embedding(seq.long() + 90)
            seq_packed = nn.utils.rnn.pack_padded_sequence(
                seq_embed,
                torch.flatten(lens),
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
        return True


class _Transformer(nn.Module):
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
        super(_Transformer, self).__init__()
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
            self.transformer = nn.Sequential()
            for i in range(attn_layers):
                self.transformer.add_module(
                    f"block_{i}",
                    TransformerBlock(
                        embed_dim=embedding_dim,
                        attn_heads=attn_heads,
                        ff_dim=ff_dim,
                        dropout=dropout,
                        activation=ReGLU,
                    ),
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


class _SeqTransformer(_Transformer):
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
            super(_SeqTransformer, self).__init__(
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
            super(_Transformer, self).__init__()
            self.run = False

    def forward(self, x, derived_tensors):
        if self.run:
            seq = derived_tensors["Lay-up Sequence"]
            lens = derived_tensors["Number of Layers"]
            max_len = seq.size(1)
            # for the definition of padding_mask, see nn.MultiheadAttention.forward
            padding_mask = torch.arange(max_len).expand(len(lens), max_len) >= lens
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
        return True
