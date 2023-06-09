import sys
import torch
from torch import nn
import inspect
from src.model.base import get_sequential, get_linear
from .common.base import AbstractClustering


class AbstractSN(nn.Module):
    def __init__(self, **kwargs):
        super(AbstractSN, self).__init__()
        self.activ = torch.abs
        self._register_params(**kwargs)
        self.lstsq_input = None
        self.lstsq_output = None

    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.ones(n_clusters))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 5))

    def _linear(self, s, x_cluster):
        self.lstsq_input = s
        self.lstsq_output = -torch.mul(self.activ(self.a[x_cluster]), s) + self.activ(
            self.b[x_cluster]
        )
        return self.lstsq_output


class LinLog(AbstractSN):
    def forward(self, s: torch.Tensor, x_cluster: torch.Tensor):
        s = torch.abs(s)
        return self._linear(s, x_cluster)


class LogLog(AbstractSN):
    def forward(self, s: torch.Tensor, x_cluster: torch.Tensor):
        s = torch.clamp(torch.abs(s), min=1e-8)
        log_s = torch.log10(torch.add(s, 1))
        return self._linear(log_s, x_cluster)


available_sn = []
for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if issubclass(cls, AbstractSN) and cls != AbstractSN:
        available_sn.append(cls)


def get_sns(**kwargs):
    sns = nn.ModuleList([i(**kwargs) for i in available_sn])
    return sns


class AbstractSNClustering(nn.Module):
    def __init__(
        self, layers, hidden_rep_dim: int, clustering: AbstractClustering, **kwargs
    ):
        super(AbstractSNClustering, self).__init__()
        self.clustering = clustering
        self.n_clusters = self.clustering.n_total_clusters
        self.tune_head = get_linear(
            n_inputs=hidden_rep_dim, n_outputs=1, nonlinearity="relu"
        )
        self.tune_head_normalize = nn.Sigmoid()
        self.sns = get_sns(n_clusters=self.n_clusters)

        # self.weight = 0.8
        # self.exp_avg_factor = 0.8
        # # Solved by exponential averaging
        # self.register_buffer(
        #     "running_tune_weight", torch.mul(torch.ones(self.n_clusters), self.weight)
        # )
        # Solved by logistic regression
        self.running_sn_weight = nn.Parameter(
            torch.mul(torch.ones((self.n_clusters, len(self.sns))), 1 / len(self.sns))
        )
        self.ridge_input = None
        self.ridge_output = None
        self.x_cluster = None

    # def _update(self, value, name):
    #     if self.training:
    #         with torch.no_grad():
    #             setattr(
    #                 self,
    #                 name,
    #                 self.exp_avg_factor * value
    #                 + (1 - self.exp_avg_factor) * getattr(self, name),
    #             )
    #         return value
    #     else:
    #         return getattr(self, name)

    def forward(self, x, s, hidden, naive_pred):
        # Projection from hidden output to SN weights and tuning output
        x_tune = self.tune_head_normalize(self.tune_head(hidden))

        # Clustering
        x_cluster = self.clustering(x)
        resp = torch.zeros((x.shape[0], self.n_clusters), device=x.device)
        resp[torch.arange(x.shape[0]), x_cluster] = 1
        nk = torch.add(torch.sum(resp, dim=0), 1e-12)
        self.x_cluster = x_cluster
        self.resp = resp
        self.nk = nk

        # Calculate SN results in each cluster
        x_sn = torch.concat([sn(s, x_cluster).unsqueeze(-1) for sn in self.sns], dim=1)
        # Weighted sum of SN predictions
        self.ridge_input = x_sn
        x_sn = torch.mul(x_sn, self.running_sn_weight[x_cluster, :])
        x_sn = torch.sum(x_sn, dim=1).view(-1, 1)
        self.ridge_output = x_sn.flatten()

        # Calculate mean prediction and tuning in each cluster
        # if self.training:
        #     with torch.no_grad():
        #         mean_pred_clusters = torch.flatten(
        #             torch.matmul(resp.T, x_sn) / nk.unsqueeze(-1)
        #         )
        #         estimate_weight = torch.mul(mean_pred_clusters, self.weight)
        #         # Not updating if no data point in this cluster.
        #         invalid_weight = nk < 1
        #         estimate_weight[invalid_weight] = self.running_tune_weight[
        #             invalid_weight
        #         ]
        #         # Exponential averaging update
        #         tune_weight = self._update(estimate_weight, "running_tune_weight")[
        #             x_cluster
        #         ]
        # else:
        #     tune_weight = self.running_tune_weight[x_cluster]
        # Weighted sum of prediction and tuning
        out = x_sn + torch.mul(x_tune, naive_pred - x_sn)
        return out
