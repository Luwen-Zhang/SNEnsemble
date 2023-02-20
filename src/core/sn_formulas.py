import sys, inspect
from src.core.trainer import Trainer
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy as cp


class AbstractSN(nn.Module):
    def __init__(self, trainer: Trainer):
        super(AbstractSN, self).__init__()
        self.sn_vars = self._get_sn_vars()
        self.trainer = trainer
        self.tabular_feature_names = self.trainer.feature_names
        self.derived_feature_names = list(self.trainer.derived_data.keys())
        self.feature_mapping = {}
        self.tabular_feature_indices = {}
        self.to(trainer.device)
        self.material_features = np.array(
            list(self.trainer.args["feature_names_type"].keys())
        )[
            np.array(list(self.trainer.args["feature_names_type"].values()))
            == self.trainer.args["feature_types"].index("Material")
        ]
        self.material_features_idx = np.array(
            [
                self.tabular_feature_names.index(name)
                for name in self.material_features
                if name in self.tabular_feature_names
            ]
        )
        self.stress_unrelated_features_idx = np.array(
            [
                idx
                for idx, name in enumerate(self.tabular_feature_names)
                if "Stress" not in name
            ]
        )
        from src.core.nn_models import get_sequential

        self.template_sequential = get_sequential(
            n_inputs=len(self.stress_unrelated_features_idx),
            n_outputs=1,
            layers=[16, 32, 64, 32, 16],
            act_func=nn.ReLU,
        )

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

    @classmethod
    def activated(cls):
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

    def _get_var_slices(self, x, derived_tensors):
        var_slices = []
        for var in self.sn_vars:
            if self.feature_mapping[var] == 0:
                var_slices.append(x[:, self.tabular_feature_indices[var]].view(-1, 1))
            else:
                var_slices.append(derived_tensors[var])
        return var_slices

    @staticmethod
    def _get_sn_vars():
        raise NotImplementedError

    def _register_variable(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        raise NotImplementedError

    def get_tex(self):
        raise NotImplementedError


class linlogSN(AbstractSN):
    def __init__(self, trainer: Trainer):
        super(linlogSN, self).__init__(trainer)

    @staticmethod
    def _get_sn_vars():
        return ["Absolute Maximum Stress"]

    def _register_variable(self):
        self.a = cp(self.template_sequential)
        self.b = cp(self.template_sequential)

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        mat = x[:, self.stress_unrelated_features_idx]
        a, b = -torch.abs(self.a(mat)), torch.abs(self.b(mat))
        return a * var_slices[0] + b

    def get_tex(self):
        return r"a\sigma_{max}+b"

    @classmethod
    def activated(cls):
        return True


class loglogSN(linlogSN):
    def __init__(self, trainer: Trainer):
        super(loglogSN, self).__init__(trainer)
        self.s_zero_slip = trainer.get_zero_slip(self._get_sn_vars()[0])

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = var_slices[0] - self.s_zero_slip
        mat = x[:, self.stress_unrelated_features_idx]
        a, b = -torch.abs(self.a(mat)), torch.abs(self.b(mat))
        return a * torch.log10(torch.abs(s) + 1e-5) + b

    def get_tex(self):
        return r"\mathrm{sgn}(\sigma_{max})a\mathrm{log}\left(\left|\sigma_{max}/\mathrm{std}(\sigma_{max})\right|+1\right)+b"

    @classmethod
    def activated(cls):
        return True


class TrivialSN(linlogSN):
    def __init__(self, trainer: Trainer):
        super(TrivialSN, self).__init__(trainer)
        self.s_zero_slip = trainer.get_zero_slip(self._get_sn_vars()[0])
        self.s_min = 1e8
        self.s_max = -1e8

    def _register_variable(self):
        self.a = cp(self.template_sequential)
        self.b = cp(self.template_sequential)
        self.c = cp(self.template_sequential)
        self.d = cp(self.template_sequential)

    def get_tex(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = torch.abs(var_slices[0] - self.s_zero_slip)

        if self.training:
            self.s_min = np.min([self.s_min, torch.min(s).cpu().numpy()])
            self.s_max = np.max([self.s_max, torch.max(s).cpu().numpy()])

        mat = x[:, self.stress_unrelated_features_idx]
        a, b, c, d = (
            torch.clamp(torch.abs(self.a(mat)), self.s_min, self.s_max),
            -torch.abs(self.b(mat)),
            -torch.clamp(
                torch.abs(self.c(mat)),
                (self.s_max - self.s_min) / 100,
                (self.s_max - self.s_min) / 1,
            ),
            torch.clamp(torch.abs(self.d(mat)), 0, 10),
        )
        return torch.pow(s - a, 3) * b + c * (s - a) + d

    @classmethod
    def activated(cls):
        return True


class SigmoidSN(linlogSN):
    def __init__(self, trainer: Trainer):
        super(SigmoidSN, self).__init__(trainer)
        self.s_zero_slip = trainer.get_zero_slip(self._get_sn_vars()[0])
        self.s_min = 1e8
        self.s_max = -1e8

    def _register_variable(self):
        self.a = cp(self.template_sequential)
        self.b = cp(self.template_sequential)
        self.c = cp(self.template_sequential)
        self.d = cp(self.template_sequential)

    def get_tex(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = torch.abs(var_slices[0] - self.s_zero_slip)

        if self.training:
            self.s_min = np.min([self.s_min, torch.min(s).cpu().numpy()])
            self.s_max = np.max([self.s_max, torch.max(s).cpu().numpy()])

        mat = x[:, self.stress_unrelated_features_idx]
        a, b, c, d = (
            -torch.abs(self.a(mat)),
            torch.abs(self.b(mat)).clamp(self.s_min, self.s_max),
            torch.abs(self.c(mat)),
            torch.abs(self.d(mat)),
        )
        return torch.log(1 / (a * (s - b)).clamp(1e-5, 1 - 1e-5) - 1) * c + d

    @classmethod
    def activated(cls):
        return False


class KohoutSN(linlogSN):
    def __init__(self, trainer: Trainer):
        super(KohoutSN, self).__init__(trainer)
        self.s_zero_slip = trainer.get_zero_slip(self._get_sn_vars()[0])

    def get_tex(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = torch.abs(var_slices[0] - self.s_zero_slip) + 1
        mat = x[:, self.stress_unrelated_features_idx]
        a, b, B = (
            torch.abs(self.a(mat)) + 1e-4,
            -torch.abs(self.b(mat)) - 1e-4,
            torch.abs(self.B(mat)),
        )
        C = B + torch.abs(self.C(mat))
        tmp = torch.pow(torch.abs(s / a), 1 / b)
        return torch.log10(torch.abs((B - tmp) / (tmp / C - 1)) + 1)

    def _register_variable(self):
        self.a = cp(self.template_sequential)
        self.b = cp(self.template_sequential)
        self.B = cp(self.template_sequential)
        self.C = cp(self.template_sequential)

    @classmethod
    def activated(cls):
        return False


sn_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSN) and cls != AbstractSN and cls.activated():
        sn_mapping[name] = cls
