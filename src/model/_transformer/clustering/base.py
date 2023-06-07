import sys
import torch
from torch import nn
import inspect
import torch.nn.functional as F
from src.model.base import get_sequential
from .common.base import (
    AbstractClustering,
    AbstractMultilayerClustering,
)
from typing import Type, List


class SNMarker(nn.Module):
    def __init__(self):
        super(SNMarker, self).__init__()
        self.activ = F.softplus
        self.register_buffer("running_weight_a", torch.tensor([0.0]))
        self.register_buffer("running_weight_b", torch.tensor([0.0]))
        self.register_buffer("running_approx_a", torch.tensor([1.0]))
        self.register_buffer("running_approx_b", torch.tensor([5.0]))
        self.exp_avg_factor = 0.1
        self.weight = 1e-1

    def _update(self, value, name):
        if self.training:
            with torch.no_grad():
                setattr(
                    self,
                    name,
                    self.exp_avg_factor * value
                    + (1 - self.exp_avg_factor) * getattr(self, name),
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


available_sn = []
for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if issubclass(cls, SNMarker) and cls != SNMarker:
        available_sn.append(cls)


def get_sns():
    sns = nn.ModuleList([i() for i in available_sn])
    return sns


n_coeff_ls = [sn.n_coeff for sn in get_sns()]
proj_dims = [sum(n_coeff_ls), len(n_coeff_ls)]
proj_dim = sum(n_coeff_ls) + len(n_coeff_ls)


class SN(nn.Module):
    def __init__(self, n_cluster_features, layers, *args, **kwargs):
        super(SN, self).__init__()
        self.sns = get_sns()
        self.n_coeff_ls = n_coeff_ls
        self.proj_dim = proj_dim
        self.proj_dims = proj_dims
        self.coeff_head = get_sequential(
            layers=[32],
            n_inputs=self.proj_dim,
            n_outputs=self.proj_dim,
            act_func=nn.ReLU,
        )

    def forward(self, x, s, naive_pred, coeffs_proj):
        coeffs_proj = self.coeff_head(coeffs_proj) + coeffs_proj
        sn_coeffs, sn_weights = coeffs_proj.split(self.proj_dims, dim=1)
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


class AbstractSNClustering(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        layers,
        algorithm_class: Type[AbstractClustering],
        n_pca_dim: int = None,
    ):
        super(AbstractSNClustering, self).__init__()
        self.n_clusters = n_clusters
        self.clustering = algorithm_class(
            n_clusters=n_clusters, n_input=n_input, n_pca_dim=n_pca_dim
        )
        self.sns = nn.ModuleList(
            [SN(n_cluster_features=n_input, layers=layers) for i in range(n_clusters)]
        )
        self.n_coeff_ls = n_coeff_ls
        self.proj_dim = proj_dim
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1 + n_input,
            n_outputs=self.proj_dim,
            act_func=nn.ReLU,
        )

    def forward(self, x, s, naive_pred):
        coeffs_proj = self.coeff_head(torch.cat([x, naive_pred], dim=1)) + naive_pred
        x_cluster = self.clustering(x)
        out = naive_pred.clone()
        for i_cluster in range(self.n_clusters):
            idx_in_cluster = torch.where(x_cluster == i_cluster)[0]
            if len(idx_in_cluster) >= 2:
                pred = self.sns[i_cluster](
                    x[idx_in_cluster, :],
                    s[idx_in_cluster],
                    naive_pred[idx_in_cluster, :],
                    coeffs_proj[idx_in_cluster, :],
                )
                out[idx_in_cluster] = pred

        return out


class AbstractMultilayerSNClustering(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input_1: int,
        n_input_2: int,
        input_1_idx: List[int],
        input_2_idx: List[int],
        layers,
        algorithm_class: Type[AbstractMultilayerClustering],
        n_clusters_per_cluster: int = 5,
        n_pca_dim: int = None,
    ):
        super(AbstractMultilayerSNClustering, self).__init__()
        self.clustering = algorithm_class(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=input_1_idx,
            input_2_idx=input_2_idx,
            n_clusters_per_cluster=n_clusters_per_cluster,
            n_pca_dim=n_pca_dim,
            shared_second_layer_clusters=False,
        )
        self.n_clusters = self.clustering.n_total_clusters
        self.sns = nn.ModuleList(
            [
                SN(n_cluster_features=self.clustering.n_input, layers=layers)
                for i in range(self.n_clusters)
            ]
        )
        self.n_coeff_ls = n_coeff_ls
        self.proj_dim = proj_dim
        self.coeff_head = get_sequential(
            layers=layers,
            n_inputs=1 + self.clustering.n_input,
            n_outputs=proj_dim,
            act_func=nn.ReLU,
        )

    def forward(self, x, s, naive_pred):
        coeffs_proj = self.coeff_head(torch.cat([x, naive_pred], dim=1)) + naive_pred
        x_cluster = self.clustering(x)
        out = naive_pred.clone()
        for i_cluster in range(self.n_clusters):
            idx_in_cluster = torch.where(x_cluster == i_cluster)[0]
            if len(idx_in_cluster) >= 2:
                pred = self.sns[i_cluster](
                    x[idx_in_cluster, :],
                    s[idx_in_cluster],
                    naive_pred[idx_in_cluster, :],
                    coeffs_proj[idx_in_cluster, :],
                )
                out[idx_in_cluster] = pred

        return out
