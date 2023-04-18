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
        self.activ = F.relu
        self.register_buffer("running_a", torch.tensor([0.0]))
        self.register_buffer("running_b", torch.tensor([0.0]))
        self.momentum = 0.1

    def _get_weight(self, ori_a, obj_a, ori_b, obj_b):
        if self.training:
            weight_a = torch.max(obj_a) / (torch.max(ori_a) + 1e-8) * 1e-1
            weight_b = torch.max(obj_b) / (torch.max(ori_b) + 1e-8) * 1e-1
            with torch.no_grad():
                self.running_a = (
                    self.momentum * weight_a + (1 - self.momentum) * self.running_a
                )
                self.running_b = (
                    self.momentum * weight_b + (1 - self.momentum) * self.running_b
                )
        else:
            weight_a = self.running_a
            weight_b = self.running_b
        return weight_a, weight_b


class LinLog(SNMarker):
    def __init__(self):
        super(LinLog, self).__init__()
        self.n_coeff = 2

    def forward(
        self,
        s: torch.Tensor,
        coeff: torch.Tensor,
        grad_s: torch.Tensor,
        naive_pred: torch.Tensor,
    ):
        s = torch.abs(s)
        grad_s = F.relu(-grad_s)
        approx_b = F.relu(torch.mul(grad_s, s) + naive_pred)
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = self.activ(a), self.activ(b)
        weight_a, weight_b = self._get_weight(a, grad_s, b, approx_b)
        a = -a * weight_a - grad_s
        b = b * weight_b + approx_b
        return a * torch.abs(s) + b


class LogLog(SNMarker):
    def __init__(self):
        super(LogLog, self).__init__()
        self.n_coeff = 2

    def forward(
        self,
        s: torch.Tensor,
        coeff: torch.Tensor,
        grad_s: torch.Tensor,
        naive_pred: torch.Tensor,
    ):
        s = torch.clamp(torch.abs(s), min=1e-8)
        grad_s = F.relu(-grad_s)
        log_s = torch.log10(s)
        grad_log_s = F.relu(torch.mul(torch.mul(grad_s, float(np.log(10))), s))
        approx_b = F.relu(torch.mul(grad_log_s, log_s) + naive_pred)
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = self.activ(a), self.activ(b)
        weight_a, weight_b = self._get_weight(a, grad_s, b, approx_b)
        a = -a * weight_a - grad_log_s
        b = b * weight_b + approx_b
        return a * log_s + b


class SN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SN, self).__init__()
        self.sns = nn.ModuleList()
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(cls, SNMarker) and cls != SNMarker:
                self.sns.append(cls())
        self.n_coeff_ls = [sn.n_coeff for sn in self.sns]

    def forward(self, s, coeff, grad_s, naive_pred):
        coeffs, sn_component_weights = coeff.split(
            [sum(self.n_coeff_ls), len(self.n_coeff_ls)], dim=1
        )
        coeffs = coeffs.split(self.n_coeff_ls, dim=1)

        x_sn = torch.concat(
            [
                sn(s.view(-1, 1), coeff, grad_s, naive_pred)
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
