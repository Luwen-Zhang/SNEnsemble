from src.utils import *
from skopt.space import Integer, Categorical
from .catembed_lstm import CatEmbedLSTM
from src.model import AbstractNN
from .base import get_sequential
import torch.nn as nn
from typing import *
from torch import Tensor
import torch.nn.functional as F


class TransformerLSTM(CatEmbedLSTM):
    def _get_program_name(self):
        return "TransformerLSTM"

    def _new_model(self, model_name, verbose, **kwargs):
        set_torch_random(0)
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, type in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][type] == "Material"
        ]
        return TransformerLSTMNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
            manual_activate_sn=self.manual_activate_sn,
            sn_coeff_vars_idx=sn_coeff_vars_idx,
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            lstm_embedding_dim=kwargs["lstm_embedding_dim"],
            embedding_dim=kwargs["embedding_dim"],
            n_hidden=kwargs["n_hidden"],
            lstm_layers=kwargs["lstm_layers"],
            attn_layers=kwargs["attn_layers"],
            attn_heads=kwargs["attn_heads"],
        ).to(self.trainer.device)

    def _space(self, model_name):
        return [
            Categorical(
                categories=[2, 4, 8, 16, 32, 64, 128], name="lstm_embedding_dim"
            ),
            Categorical(categories=[8, 16, 32, 64], name="embedding_dim"),
            Integer(low=1, high=100, prior="uniform", name="n_hidden", dtype=int),
            Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
            Categorical(categories=[2, 4, 8], name="attn_layers"),
            Categorical(categories=[2, 4, 8], name="attn_heads"),
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        return {
            "lstm_embedding_dim": 16,
            "embedding_dim": 64,
            "n_hidden": 10,
            "lstm_layers": 1,
            "attn_layers": 4,
            "attn_heads": 8,
            "lr": 0.003,
            "weight_decay": 0.002,
            "batch_size": 1024,
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
        lstm_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
        attn_layers=4,
        attn_heads=8,
    ):
        super(TransformerLSTMNN, self).__init__(trainer)
        self.n_cont = n_inputs
        self.n_outputs = n_outputs
        self.n_cat = len(cat_num_unique) if cat_num_unique is not None else 0
        self.last_dim = []

        # Module: Continuous embedding
        self.embedding_dim = embedding_dim
        self.cont_norm = nn.BatchNorm1d(n_inputs)
        self.cont_embed_weight = nn.init.kaiming_uniform_(
            nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
        )
        self.cont_embed_bias = nn.init.kaiming_uniform_(
            nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
        )
        self.cont_dropout = nn.Dropout(0.1)

        # Module: Categorical embedding
        if "categorical" in self.derived_feature_names:
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
            self.cat_dropout = nn.Dropout(0.1)
            self.run_cat = True
        else:
            self.run_cat = False

        # Module: Transformer
        self.transformer = nn.Sequential()

        for i in range(attn_layers):
            self.transformer.add_module(
                f"block_{i}",
                TransformerBlock(embed_dim=embedding_dim, attn_heads=attn_heads),
            )
        self.transformer_head = get_sequential(
            layers, embedding_dim, n_outputs, nn.ReLU, norm_type="layer"
        )
        self.last_dim.append(1)

        # Module: Sequence encoding by LSTM
        self.n_hidden = n_hidden
        self.lstm_embedding_dim = lstm_embedding_dim
        self.lstm_layers = lstm_layers
        if "Number of Layers" in self.derived_feature_names:
            self.seq_lstm = nn.LSTM(
                input_size=lstm_embedding_dim,
                hidden_size=n_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            # The input degree would be in range [-90, 100] (where 100 is the padding value). It will be transformed to
            # [0, 190] by adding 100, so the number of categories (vocab) will be 191
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=lstm_embedding_dim
            )
            self.last_dim.append(1)
            self.run_lstm = True
        else:
            self.run_lstm = False

        # Module: SN models
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
        if len(activated_sn) > 0:
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
            self.last_dim.append(1)
            self.run_sn = True
        else:
            self.run_sn = False

        self.run_any = self.run_sn or self.run_lstm
        if self.run_any:
            self.w = get_sequential(
                layers,
                sum(self.last_dim),
                n_outputs,
                nn.ReLU,
            )
        else:
            self.w = nn.Identity()

    def _forward(self, x, derived_tensors):
        x_cont = self.cont_embed_weight.unsqueeze(0) * self.cont_norm(x).unsqueeze(
            2
        ) + self.cont_embed_bias.unsqueeze(0)
        x_cont = self.cont_dropout(x_cont)
        if self.run_cat:
            cat = derived_tensors["categorical"].long()
            x_cat_embeds = [
                self.cat_embeds[i](cat[:, i]).unsqueeze(1) for i in range(cat.size(1))
            ]
            x_cat = torch.cat(x_cat_embeds, 1)
            x_cat = self.cat_dropout(x_cat)
            x_trans = torch.cat([x_cont, x_cat], dim=1)
        else:
            x_trans = x_cont
        x_trans = self.transformer(x_trans)
        x_trans = torch.mean(x_trans, dim=1)
        x_trans = self.transformer_head(x_trans)
        all_res = [x_trans]

        if self.run_sn:
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
            all_res += [x_sn]

        if self.run_lstm:
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
            all_res += [torch.mean(h_t, dim=[0, 2]).view(-1, 1)]

        output = torch.concat(all_res, dim=1)
        output = self.w(output)
        return output


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The size of the last dimension must be even.")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, attn_heads):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.f = nn.Sequential(
            OrderedDict(
                [
                    ("first_linear", nn.Linear(embed_dim, 256)),
                    ("activation", ReGLU()),
                    ("dropout", nn.Dropout(0.1)),
                    ("second_linear", nn.Linear(256 // 2, embed_dim)),
                ]
            )
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        normed_x = self.attn_norm(x)
        x_attn, _ = self.attn(normed_x, normed_x, normed_x)
        x_attn = x + self.dropout(x_attn)
        return x + self.dropout(self.f(self.f_norm(x_attn)))
