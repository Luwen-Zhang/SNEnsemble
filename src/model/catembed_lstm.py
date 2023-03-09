from src.utils import *
from src.model import TorchModel, AbstractNN
from .base import get_sequential
from skopt.space import Integer
import torch.nn as nn
from typing import *


class CatEmbedLSTM(TorchModel):
    def __init__(
        self,
        trainer=None,
        manual_activate_sn=None,
        program=None,
        layers=None,
        model_subset=None,
    ):
        super(CatEmbedLSTM, self).__init__(
            trainer, program=program, model_subset=model_subset
        )
        self.manual_activate_sn = manual_activate_sn
        self.layers = layers

    def _get_program_name(self):
        return "CatEmbedLSTM"

    def _new_model(self, model_name, verbose, **kwargs):
        set_torch_random(0)
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, type in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][type] == "Material"
        ]
        return CatEmbedLSTMNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.layers if self.layers is None else self.layers,
            trainer=self.trainer,
            manual_activate_sn=self.manual_activate_sn,
            sn_coeff_vars_idx=sn_coeff_vars_idx,
            embed_continuous=True,
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            lstm_embedding_dim=kwargs["lstm_embedding_dim"],
            embedding_dim=kwargs["embedding_dim"],
            n_hidden=kwargs["n_hidden"],
            lstm_layers=kwargs["lstm_layers"],
        ).to(self.trainer.device)

    def _get_optimizer(self, model, warm_start, **kwargs):
        return torch.optim.Adam(
            model.parameters(),
            lr=kwargs["lr"] / 10 if warm_start else kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )

    def _get_model_names(self):
        return ["CatEmbedLSTM"]

    def _space(self, model_name):
        return [
            Integer(
                low=1,
                high=1000,
                prior="uniform",
                name="lstm_embedding_dim",
                dtype=int,
            ),
            Integer(
                low=1,
                high=1000,
                prior="uniform",
                name="embedding_dim",
                dtype=int,
            ),
            Integer(low=1, high=100, prior="uniform", name="n_hidden", dtype=int),
            Integer(low=1, high=10, prior="uniform", name="lstm_layers", dtype=int),
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        return {
            "lstm_embedding_dim": 10,  # bayes-opt: 1000
            "embedding_dim": 10,  # bayes-opt: 1
            "n_hidden": 10,  # bayes-opt: 1
            "lstm_layers": 1,  # bayes-opt: 1
            "lr": 0.003,  # bayes-opt: 0.0218894
            "weight_decay": 0.002,  # bayes-opt: 0.05
            "batch_size": 1024,  # bayes-opt: 32
        }


class BiasCatEmbedLSTM(CatEmbedLSTM):
    def _loss_fn(self, y_true, y_pred, model, *data, **kwargs):
        base_loss = self.trainer.loss_fn(y_pred, y_true)
        if not model.training:
            return base_loss
        else:
            where_weight = list(self.trainer.derived_data.keys()).index("Sample Weight")
            w = data[1 + where_weight]
            return (base_loss * w).mean()

    def _get_program_name(self):
        return "BiasCatEmbedLSTM"


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
        from src.model._thiswork_sn_formulas import sn_mapping

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
