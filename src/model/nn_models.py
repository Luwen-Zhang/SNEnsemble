from src.utils import *
from src.model import AbstractNN
import torch.nn as nn
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
        self.nets = [
            get_sequential(layers, dims[-1], 1, nn.ReLU)
            for dims in self.derived_feature_dims
        ]
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
        cat_num_unique: List[int] = None,
        cat_embedding_dim=10,
        lstm_embedding_dim=10,
        n_hidden=3,
        lstm_layers=1,
    ):
        super(CatEmbedLSTMNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs
        self.run_any = False
        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.cont_norm = nn.LayerNorm(num_inputs)

        self.cat_embedding_dim = cat_embedding_dim
        if "categorical" in self.derived_feature_names:
            # See pytorch_widedeep.models.tabular.embeddings_layers.SameSizeCatEmbeddings
            self.cat_embeds = [
                nn.Embedding(
                    num_embeddings=num_unique + 1,
                    embedding_dim=cat_embedding_dim,
                    padding_idx=0,
                )
                for num_unique in cat_num_unique
            ]
            self.cat_dropout = nn.Dropout(0.1)
            self.cat_encoder = get_sequential(
                [32], cat_embedding_dim, len(cat_num_unique), nn.ReLU
            )
            self.run_cat = True
            self.run_any = True
        else:
            self.run_cat = False

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
            self.run_lstm = True
            self.run_any = True
        else:
            self.run_lstm = False

        if self.run_any:
            self.w = get_sequential(
                [32],
                self.net.output.out_features
                + int(self.run_cat) * len(cat_num_unique)
                + int(self.run_lstm),
                num_outputs,
                nn.ReLU,
            )

    def _forward(self, x, derived_tensors):
        all_res = [self.net(self.cont_norm(x))]

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

            seq_embed = self.embedding(seq + 90)
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
        if self.run_any:
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


def get_sequential(layers, n_inputs, n_outputs, act_func, dropout=0.1):
    net = nn.Sequential()
    if len(layers) > 0:
        net.add_module("input", nn.Linear(n_inputs, layers[0]))
        net.add_module("activate", act_func())
        for idx in range(len(layers) - 1):
            net.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
            net.add_module(f"activate_{idx}", act_func())
            net.add_module(f"norm_{idx}", nn.LayerNorm(layers[idx + 1]))
            net.add_module(f"dropout_{idx}", nn.Dropout(dropout))
        net.add_module("output", nn.Linear(layers[-1], n_outputs))
    else:
        net.add_module("single_layer", nn.Linear(n_inputs, n_outputs))
        net.add_module("activate", act_func())
        net.add_module("norm", nn.LayerNorm(n_outputs))
        net.add_module("dropout", nn.Dropout(dropout))

    net.apply(init_weights)
    return net
