import sys, inspect
from src.core.trainer import Trainer
import torch.nn as nn
import torch
import numpy as np


class AbstractSN(nn.Module):
    def __init__(self, trainer: Trainer):
        super(AbstractSN, self).__init__()
        self.sn_vars = self._get_sn_vars()
        self.trainer = trainer
        self.tabular_feature_names = self.trainer.feature_names
        self.derived_feature_names = list(self.trainer.derived_data.keys())
        self.feature_mapping = {}
        self.tabular_feature_indices = {}
        self.derived_feature_indices = {}
        self.to(trainer.device)
        self._register_variable()
        self._check_sn_vars()
        self._get_sn_vars_idx()

    @classmethod
    def test_sn_vars(cls, trainer):
        for var in cls._get_sn_vars():
            if (
                not var in trainer.feature_names
                and not var in trainer.derived_data.keys()
            ):
                return False
        else:
            return True

    def _check_sn_vars(self):
        if not self.test_sn_vars(self.trainer):
            raise Exception(
                f"Required columns of {self.__class__.__name__} do not exist in Trainer.feature_names, but "
                f"it is included in selected SN models. Do not force adding any SN model."
            )

    def _get_sn_vars_idx(self):
        for var in self.sn_vars:
            if var in self.tabular_feature_names:
                self.feature_mapping[var] = 0
                self.tabular_feature_indices[var] = self.tabular_feature_names.index(
                    var
                )
            elif var in self.derived_feature_names:
                self.feature_mapping[var] = 1
                self.derived_feature_indices[var] = self.derived_feature_names.index(
                    var
                )

    def _get_var_slices(self, x, additional_tensors):
        var_slices = []
        for var in self.sn_vars:
            if self.feature_mapping[var] == 0:
                var_slices.append(x[:, self.tabular_feature_indices[var]].view(-1, 1))
            else:
                var_slices.append(additional_tensors[self.derived_feature_indices[var]])
        return var_slices

    @staticmethod
    def _get_sn_vars():
        raise NotImplementedError

    def _register_variable(self):
        raise NotImplementedError

    def forward(self, x, additional_tensors):
        raise NotImplementedError


class linlogSN(AbstractSN):
    def __init__(self, trainer: Trainer):
        super(linlogSN, self).__init__(trainer)

    @staticmethod
    def _get_sn_vars():
        return ["Absolute Maximum Stress"]

    def _register_variable(self):
        material_features = np.array(
            list(self.trainer.args["feature_names_type"].keys())
        )[
            np.array(list(self.trainer.args["feature_names_type"].values()))
            == self.trainer.args["feature_types"].index("Material")
        ]
        self.material_features_idx = np.array(
            [self.trainer.feature_names.index(name) for name in material_features]
        )

        from src.core.nn_models import get_sequential

        self.a = get_sequential(
            n_inputs=len(material_features),
            n_outputs=1,
            layers=[16, 32, 16],
            act_func=nn.ReLU,
        )
        self.b = get_sequential(
            n_inputs=len(material_features),
            n_outputs=1,
            layers=[16, 32, 16],
            act_func=nn.ReLU,
        )

    def forward(self, x, additional_tensors):
        var_slices = self._get_var_slices(x, additional_tensors)
        return self.a(x[:, self.material_features_idx]) * var_slices[0] + self.b(
            x[:, self.material_features_idx]
        )


# class loglogSN(linlogSN):
#     def __init__(self, trainer: Trainer):
#         super(loglogSN, self).__init__(trainer)
#
#     def forward(self, x, additional_tensors):
#         var_slices = self._get_var_slices(x, additional_tensors)
#         sgn = torch.sign(var_slices[0])
#         return self.a(x[:, self.material_features_idx]) * torch.log10(
#             var_slices[0] * sgn + 1
#         ) * sgn + self.b(x[:, self.material_features_idx])


sn_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSN) and cls != AbstractSN:
        sn_mapping[name] = cls
