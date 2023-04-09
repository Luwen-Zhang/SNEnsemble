import sys
import torch
from torch import nn
import inspect


class SNMarker(nn.Module):
    ...


class LinLog(SNMarker):
    def __init__(self):
        super(LinLog, self).__init__()
        self.n_coeff = 2

    def forward(self, s: torch.Tensor, coeff: torch.Tensor):
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = -torch.abs(a), torch.abs(b)
        return a * torch.abs(s) + b


class LogLog(SNMarker):
    def __init__(self):
        super(LogLog, self).__init__()
        self.n_coeff = 2

    def forward(self, s: torch.Tensor, coeff: torch.Tensor):
        a, b = coeff.chunk(self.n_coeff, 1)
        a, b = -torch.abs(a), torch.abs(b)
        return a * torch.log10(torch.clamp(torch.abs(s), min=1e-8)) + b


class SN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SN, self).__init__()
        self.sns = nn.ModuleList()
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(cls, SNMarker) and cls != SNMarker:
                self.sns.append(cls())
        self.n_coeff_ls = [sn.n_coeff for sn in self.sns]

    def forward(self, s, coeff):
        coeffs, sn_component_weights = coeff.split(
            [sum(self.n_coeff_ls), len(self.n_coeff_ls)], dim=1
        )
        coeffs = coeffs.split(self.n_coeff_ls, dim=1)

        x_sn = torch.concat(
            [sn(s.view(-1, 1), coeff) for sn, coeff in zip(self.sns, coeffs)],
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
