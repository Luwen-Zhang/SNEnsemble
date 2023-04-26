import sys
import torch
from torch import nn
from torch.nn import Parameter
import inspect
import torch.nn.functional as F
import numpy as np
from ..base import get_sequential
from .kmeans import Cluster, KMeans


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
        a = -a * weight_a - running_approx_a * (1 - weight_a)
        b = b * weight_b + running_approx_b * (1 - weight_b)
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
    def __init__(self, layers, *args, **kwargs):
        super(SN, self).__init__()
        self.sns = nn.ModuleList()
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(cls, SNMarker) and cls != SNMarker:
                self.sns.append(cls())
        self.n_coeff_ls = [sn.n_coeff for sn in self.sns]
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1,
            n_outputs=sum(self.n_coeff_ls) + len(self.n_coeff_ls),
            act_func=nn.ReLU,
        )

    def forward(self, s, naive_pred):
        coeffs_proj = self.coeff_head(naive_pred) + naive_pred
        sn_coeffs, sn_weights = coeffs_proj.split(
            [sum(self.n_coeff_ls), len(self.n_coeff_ls)], dim=1
        )
        sn_coeffs = sn_coeffs.split(self.n_coeff_ls, dim=1)
        x_sn = torch.concat(
            [
                sn(s.view(-1, 1), coeff, naive_pred)
                for sn, coeff in zip(self.sns, sn_coeffs)
            ],
            dim=1,
        )
        x_sn = torch.mul(
            x_sn,
            nn.functional.normalize(
                torch.abs(sn_weights),
                p=1,
                dim=1,
            ),
        )
        x_sn = torch.sum(x_sn, dim=1).view(-1, 1)
        return x_sn


class _SNCluster(Cluster):
    def __init__(self, n_input, layers, momentum):
        super(_SNCluster, self).__init__(n_input, momentum=momentum)
        self.sn = SN(layers)


class KMeansSN(nn.Module):
    def __init__(self, n_clusters: int, n_input: int, layers):
        super(KMeansSN, self).__init__()

        self.sns = [
            _SNCluster(n_input=n_input, layers=layers, momentum=0.1)
            for i in range(n_clusters)
        ]
        self.kmeans = KMeans(n_clusters=n_clusters, n_input=n_input, clusters=self.sns)

    def forward(self, x, s, naive_pred):
        x_cluster = self.kmeans(x)
        out = naive_pred.clone()
        for i_cluster, cluster in enumerate(self.kmeans.clusters):
            idx_in_cluster = torch.where(x_cluster == i_cluster)[0]
            if len(idx_in_cluster) >= 2:
                pred = cluster.sn(s[idx_in_cluster], naive_pred[idx_in_cluster, :])
                out[idx_in_cluster] = pred
        return out
