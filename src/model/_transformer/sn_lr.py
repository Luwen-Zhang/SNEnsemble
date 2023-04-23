import sys
import torch
from torch import nn
from torch.nn import Parameter
import inspect
import torch.nn.functional as F
import numpy as np


class SNMarker(nn.Module):
    def __init__(self):
        super(SNMarker, self).__init__()
        self.activ = F.softplus
        self.register_buffer("running_weight_a", torch.tensor([0.0]))
        self.register_buffer("running_weight_b", torch.tensor([0.0]))
        self.register_buffer("running_approx_a", torch.tensor([1.0]))
        self.register_buffer("running_approx_b", torch.tensor([5.0]))
        self.momentum = 0.1
        self.weight = 1e-1

    def _update(self, value, name):
        if self.training:
            with torch.no_grad():
                setattr(
                    self,
                    name,
                    self.momentum * value + (1 - self.momentum) * getattr(self, name),
                )
            return value
        else:
            return getattr(self, name)

    def _linear(self, s, n, a, b):
        X = torch.concat([s, torch.ones_like(s)], dim=1)
        approx_a, approx_b = torch.linalg.lstsq(X.T @ X, X.T @ n).solution
        running_approx_a = self._update(-approx_a, "running_approx_a")
        running_approx_b = self._update(approx_b, "running_approx_b")
        weight_a = self._update(
            running_approx_a / (torch.mean(a) + 1e-8) * self.weight, "running_weight_a"
        )
        weight_b = self._update(
            running_approx_b / (torch.mean(b) + 1e-8) * self.weight, "running_weight_b"
        )
        a = -a * weight_a - running_approx_a
        b = b * weight_b + running_approx_b
        return a * s + b


class LinLog(SNMarker):
    def __init__(self):
        super(LinLog, self).__init__()
        self.n_coeff = 2

    def forward(
        self,
        s: torch.Tensor,
        coeff: torch.Tensor,
        naive_pred: torch.Tensor,
    ):
        s = torch.abs(s)
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = self.activ(a), self.activ(b)
        return self._linear(s, naive_pred, a, b)


class LogLog(SNMarker):
    def __init__(self):
        super(LogLog, self).__init__()
        self.n_coeff = 2

    def forward(
        self,
        s: torch.Tensor,
        coeff: torch.Tensor,
        naive_pred: torch.Tensor,
    ):
        s = torch.clamp(torch.abs(s), min=1e-8)
        log_s = torch.log10(s)
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = self.activ(a), self.activ(b)
        return self._linear(log_s, naive_pred, a, b)


class SN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SN, self).__init__()
        self.sns = nn.ModuleList()
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(cls, SNMarker) and cls != SNMarker:
                self.sns.append(cls())
        self.n_coeff_ls = [sn.n_coeff for sn in self.sns]

    def forward(self, s, coeff, naive_pred):
        coeffs, sn_component_weights = coeff.split(
            [sum(self.n_coeff_ls), len(self.n_coeff_ls)], dim=1
        )
        coeffs = coeffs.split(self.n_coeff_ls, dim=1)

        x_sn = torch.concat(
            [
                sn(s.view(-1, 1), coeff, naive_pred)
                for sn, coeff in zip(self.sns, coeffs)
            ],
            dim=1,
        )
        x_sn = torch.mul(
            x_sn,
            nn.functional.normalize(
                torch.abs(sn_component_weights),
                p=1,
                dim=1,
            ),
        )
        x_sn = torch.sum(x_sn, dim=1).view(-1, 1)
        return x_sn
