from src.utils import *
import torch.nn as nn
from copy import deepcopy as cp
import inspect
from src.model.base import get_sequential


class AbstractSN(nn.Module):
    def __init__(
        self,
        cont_feature_names,
        derived_feature_names,
        s_zero_slip,
        sn_coeff_vars_idx,
    ):
        super(AbstractSN, self).__init__()
        self.cont_feature_names = cp(cont_feature_names)
        self.derived_feature_names = cp(derived_feature_names)
        self.s_zero_slip = s_zero_slip
        self.feature_mapping = {}
        self.tabular_feature_indices = {}
        self.sn_coeff_vars_idx = np.array(sn_coeff_vars_idx)

        self.template_sequential = get_sequential(
            n_inputs=len(self.sn_coeff_vars_idx),
            n_outputs=1,
            layers=[16, 32, 64, 32, 16],
            act_func=nn.ReLU,
        )

        self._register_variable()
        self._check_sn_vars()
        self._get_sn_vars_idx()

    @classmethod
    def test_sn_vars(cls, cont_feature_names, derived_feature_names):
        for var in cls.get_sn_vars():
            if not var in cont_feature_names and not var in derived_feature_names:
                return False
        else:
            return True

    @classmethod
    def activated(cls):
        return True

    def _check_sn_vars(self):
        if not self.test_sn_vars(self.cont_feature_names, self.derived_feature_names):
            raise Exception(
                f"Required columns of {self.__class__.__name__} do not exist in Trainer.cont_feature_names, but "
                f"it is included in selected SN models. Do not force adding any SN model."
            )

    def _get_sn_vars_idx(self):
        for var in self.get_sn_vars():
            if var in self.cont_feature_names:
                self.feature_mapping[var] = 0
                self.tabular_feature_indices[var] = self.cont_feature_names.index(var)
            elif var in self.derived_feature_names:
                self.feature_mapping[var] = 1
            else:
                raise Exception(
                    f"Required sn variable {var} not found. Run test_sn_vars to check."
                )

    def _get_var_slices(self, x, derived_tensors):
        var_slices = []
        for var in self.get_sn_vars():
            if self.feature_mapping[var] == 0:
                var_slices.append(x[:, self.tabular_feature_indices[var]].view(-1, 1))
            else:
                var_slices.append(derived_tensors[var])
        return var_slices

    @staticmethod
    def get_sn_vars():
        raise NotImplementedError

    def _register_variable(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        raise NotImplementedError

    def get_tex(self):
        raise NotImplementedError


class linlogSN(AbstractSN):
    @staticmethod
    def get_sn_vars():
        return ["Absolute Maximum Stress"]

    def _register_variable(self):
        self.a = cp(self.template_sequential)
        self.b = cp(self.template_sequential)

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        mat = x[:, self.sn_coeff_vars_idx]
        a, b = -torch.abs(self.a(mat)), torch.abs(self.b(mat))
        return a * var_slices[0] + b

    def get_tex(self):
        return r"a\sigma_{max}+b"

    @classmethod
    def activated(cls):
        return True


class loglogSN(linlogSN):
    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = var_slices[0] - self.s_zero_slip
        mat = x[:, self.sn_coeff_vars_idx]
        a, b = -torch.abs(self.a(mat)), torch.abs(self.b(mat))
        return a * torch.log10(torch.abs(s) + 1e-5) + b

    def get_tex(self):
        return r"\mathrm{sgn}(\sigma_{max})a\mathrm{log}\left(\left|\sigma_{max}/\mathrm{std}(\sigma_{max})\right|+1\right)+b"

    @classmethod
    def activated(cls):
        return True


class TrivialSN(linlogSN):
    def _register_variable(self):
        self.s_min = 1e8
        self.s_max = -1e8
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

        mat = x[:, self.sn_coeff_vars_idx]
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
        return False


class SigmoidSN(linlogSN):
    def _register_variable(self):
        self.s_min = 1e8
        self.s_max = -1e8
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

        mat = x[:, self.sn_coeff_vars_idx]
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
    def get_tex(self):
        raise NotImplementedError

    def forward(self, x, derived_tensors):
        var_slices = self._get_var_slices(x, derived_tensors)
        s = torch.abs(var_slices[0] - self.s_zero_slip) + 1
        mat = x[:, self.sn_coeff_vars_idx]
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
