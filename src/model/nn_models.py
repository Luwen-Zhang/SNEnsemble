from src.utils import *
from src.model import AbstractNN
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


class MLPNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer):
        super(MLPNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs
        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.nets = nn.ModuleList(
            [
                get_sequential(layers, dims[-1], 1, nn.ReLU)
                for dims in self.derived_feature_dims
            ]
        )
        self.weight = get_sequential([32], len(self.nets) + 1, num_outputs, nn.ReLU)

    def _forward(self, x, derived_tensors):
        if len(derived_tensors) > 0:
            x = [self.net(x)] + [
                net(y.to(torch.float32))
                for net, y in zip(self.nets, derived_tensors.values())
            ]
            x = torch.concat(x, dim=1)
            output = self.weight(x)
        else:
            output = self.net(x)

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
    ):
        super(CatEmbedLSTMNN, self).__init__(trainer)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.last_dim = []
        # Module 1: SN models
        from src.model.sn_formulas import sn_mapping

        activated_sn = []
        for key, sn in sn_mapping.items():
            if sn.test_sn_vars(
                trainer.cont_feature_names, list(trainer.derived_data.keys())
            ) and (manual_activate_sn is None or key in manual_activate_sn):
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

        # Module 2: Continuous embedding
        self.embedding_dim = embedding_dim
        self.cont_norm = nn.BatchNorm1d(n_inputs)
        if (
            embed_continuous
        ):  # pytorch_widedeep.models.tabular.embeddings_layers.ContEmbeddings
            self.cont_embed_weight = nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
            )
            self.cont_embed_bias = nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_inputs, embedding_dim)), a=np.sqrt(5)
            )
            self.cont_encoder = get_sequential(
                [], embedding_dim, 1, nn.ReLU, norm_type="layer"
            )
            self.last_dim.append(n_inputs)
            self.cont_dropout = nn.Dropout(0.1)
            self.run_cont = True
        else:
            self.last_dim.append(n_inputs)
            self.cont_encoder = get_sequential(layers, n_inputs, n_inputs, nn.ReLU)
            self.run_cont = False

        # Module 3: Categorical embedding
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
            self.cat_encoder = get_sequential(
                [], embedding_dim, 1, nn.ReLU, norm_type="layer"
            )
            self.last_dim.append(len(cat_num_unique))
            self.run_cat = True
        else:
            self.run_cat = False

        # Module 4: Sequence encoding by LSTM
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

        self.run_any = self.run_sn or self.run_lstm or self.run_cat
        self.w = get_sequential(
            layers if self.run_any else [],
            sum(self.last_dim),
            n_outputs,
            nn.ReLU,
        )

    def _forward(self, x, derived_tensors):
        all_res = []

        if self.run_cont:
            x_cont = self.cont_embed_weight.unsqueeze(0) * self.cont_norm(x).unsqueeze(
                2
            ) + self.cont_embed_bias.unsqueeze(0)
            x_cont = self.cont_dropout(x_cont)
            x_cont = self.cont_encoder(x_cont).squeeze(-1)
            all_res += [x_cont]
        else:
            all_res += [self.cont_encoder(self.cont_norm(x))]

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

        if self.run_cat:
            cat = derived_tensors["categorical"].long()
            x_cat_embeds = [
                self.cat_embeds[i](cat[:, i]).unsqueeze(1) for i in range(cat.size(1))
            ]
            x_cat = torch.cat(x_cat_embeds, 1)
            x_cat = self.cat_dropout(x_cat)
            x_cat = self.cat_encoder(x_cat).squeeze(-1)
            all_res += [x_cat]

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
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
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
        from src.utils.model.attention import TransformerBlock

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
        from src.model.sn_formulas import sn_mapping

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


class ThisWorkNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn, trainer):
        super(ThisWorkNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.sn_coeff_vars_idx = activated_sn[0].sn_coeff_vars_idx
        self.component_weights = get_sequential(
            [16, 64, 128, 64, 16],
            len(self.sn_coeff_vars_idx),
            len(self.activated_sn),
            nn.ReLU,
        )

    def _forward(self, x, derived_tensors):
        preds = torch.concat(
            [sn(x, derived_tensors) for sn in self.activated_sn],
            dim=1,
        )

        output = torch.mul(
            preds,
            nn.functional.normalize(
                torch.abs(self.component_weights(x[:, self.sn_coeff_vars_idx])),
                p=1,
                dim=1,
            ),
        )  # element wise multiplication
        output = torch.sum(output, dim=1).view(-1, 1)
        return output


class ThisWorkRidgeNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn, trainer):
        super(ThisWorkRidgeNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.sn_coeff_vars_idx = activated_sn[0].sn_coeff_vars_idx
        self.component_weights = torch.ones(
            [len(self.activated_sn), 1], requires_grad=False
        )
        self.preds = None

    def _forward(self, x, derived_tensors):
        preds = torch.concat(
            [sn(x, derived_tensors) for sn in self.activated_sn],
            dim=1,
        )
        self.preds = preds
        # print(preds.shape, self.component_weights.shape)
        output = torch.matmul(preds, self.component_weights)
        return output


def get_sequential(
    layers, n_inputs, n_outputs, act_func, dropout=0, use_norm=True, norm_type="batch"
):
    net = nn.Sequential()
    if norm_type == "batch":
        norm = nn.BatchNorm1d
    elif norm_type == "layer":
        norm = nn.LayerNorm
    else:
        raise Exception(f"Normalization {norm_type} not implemented.")
    if len(layers) > 0:
        net.add_module("input", nn.Linear(n_inputs, layers[0]))
        net.add_module("activate_0", act_func())
        if use_norm:
            net.add_module(f"norm_0", norm(layers[0]))
        if dropout != 0:
            net.add_module(f"dropout_0", nn.Dropout(dropout))
        for idx in range(1, len(layers)):
            net.add_module(str(idx), nn.Linear(layers[idx - 1], layers[idx]))
            net.add_module(f"activate_{idx}", act_func())
            if use_norm:
                net.add_module(f"norm_{idx}", norm(layers[idx]))
            if dropout != 0:
                net.add_module(f"dropout_{idx}", nn.Dropout(dropout))
        net.add_module("output", nn.Linear(layers[-1], n_outputs))
    else:
        net.add_module("single_layer", nn.Linear(n_inputs, n_outputs))
        net.add_module("activate", act_func())
        if use_norm:
            net.add_module("norm", norm(n_outputs))
        if dropout != 0:
            net.add_module("dropout", nn.Dropout(dropout))

    net.apply(init_weights)
    return net
