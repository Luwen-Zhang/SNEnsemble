import sys
import torch
from torch import nn
import inspect
from .common.base import AbstractClustering
from typing import Dict


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

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress"]


class LinLog(AbstractSN):
    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.abs(required_cols["Relative Maximum Stress"])
        return self._linear(s, x_cluster)


class LogLog(AbstractSN):
    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(torch.abs(required_cols["Relative Maximum Stress"]), min=1e-8)
        """
        # To add fatigue limit:
        s_sw = torch.clamp(
            s
            - torch.clamp(torch.sigmoid(self.sw[x_cluster]), max=self.min_s[x_cluster]),
            min=1e-8,
        )
        if self.training:
            s_mat = torch.mul(
                torch.ones(s.shape[0], self.min_s.shape[0], device=s.device), 100
            )
            s_mat[torch.arange(s.shape[0]), x_cluster] = s
            self.min_s = torch.min(
                torch.concat(
                    [self.min_s.view(-1, 1), torch.min(s_mat, dim=0)[0].view(-1, 1)],
                    dim=1,
                ),
                dim=1,
            )[0]
        log_s = torch.log10(s_sw)
        """
        log_s = torch.log10(s)
        return self._linear(log_s, x_cluster)

    """
    # To add fatigue limit:
    def _register_params(self, n_clusters=1, **kwargs):
        super(LogLog, self)._register_params(n_clusters=n_clusters, **kwargs)
        self.sw = nn.Parameter(torch.zeros(n_clusters))
        self.register_buffer("min_s", torch.ones(n_clusters))
    """


available_sn = []
for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if issubclass(cls, AbstractSN) and cls != AbstractSN:
        available_sn.append(cls)


def get_sns(**kwargs):
    sns = nn.ModuleList([i(**kwargs) for i in available_sn])
    return sns


class AbstractSNClustering(nn.Module):
    def __init__(self, clustering: AbstractClustering, datamodule, **kwargs):
        super(AbstractSNClustering, self).__init__()
        self.clustering = clustering
        self.n_clusters = self.clustering.n_total_clusters
        self.sns = get_sns(n_clusters=self.n_clusters)

        required_cols = []
        for sn in self.sns:
            required_cols += sn.required_cols()
        self.required_cols = list(sorted(set(required_cols)))
        self.required_indices = [
            datamodule.cont_feature_names.index(col) for col in required_cols
        ]
        self.zero_slip = [datamodule.get_zero_slip(col) for col in required_cols]
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

    def extract_cols(self, x):
        return {
            col: x[:, idx] - zero_slip
            for col, idx, zero_slip in zip(
                self.required_cols, self.required_indices, self.zero_slip
            )
        }

    def forward(self, x, clustering_features):
        required_cols = self.extract_cols(x)
        # Clustering
        x = x[:, clustering_features]
        x_cluster = self.clustering(x)
        resp = torch.zeros((x.shape[0], self.n_clusters), device=x.device)
        resp[torch.arange(x.shape[0]), x_cluster] = 1
        nk = torch.add(torch.sum(resp, dim=0), 1e-12)
        self.x_cluster = x_cluster
        self.resp = resp
        self.nk = nk

        # Calculate SN results in each cluster in parallel through vectorization.
        x_sn = torch.concat(
            [sn(required_cols, x_cluster, self.sns).unsqueeze(-1) for sn in self.sns],
            dim=1,
        )
        # Weighted sum of SN predictions
        self.ridge_input = x_sn
        x_sn = torch.mul(
            x_sn,
            nn.functional.normalize(
                torch.abs(self.running_sn_weight[x_cluster, :]), p=1
            ),
        )
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
        return x_sn
