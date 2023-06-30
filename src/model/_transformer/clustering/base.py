import sys
import torch
from torch import nn
import inspect
from src.model._transformer.clustering.common.base import AbstractClustering
from typing import Dict, List


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
        if self.use_fatigue_limit:
            self.sw = nn.Parameter(torch.zeros(n_clusters))
            self.register_buffer("min_s", torch.ones(n_clusters))

    def _linear(self, s, x_cluster):
        self.lstsq_input = s
        self.lstsq_output = -torch.mul(self.activ(self.a[x_cluster]), s) + self.activ(
            self.b[x_cluster]
        )
        return self.lstsq_output

    @staticmethod
    def required_cols() -> List[str]:
        return ["Relative Maximum Stress"]

    @property
    def use_fatigue_limit(self):
        return False

    @property
    def fatigue_limit(self):
        if not self.use_fatigue_limit or not hasattr(self, "sw"):
            raise Exception(
                f"Set the property `use_fatigue_limit` to True or register attributes `sw` and `min_s` "
                f"(see AbstractSN._register_params)."
            )
        return torch.clamp(torch.sigmoid(self.sw), max=self.min_s)

    def update_fatigue_limit(self, s, x_cluster):
        if not self.use_fatigue_limit:
            raise Exception(
                f"Set the property `use_fatigue_limit` to True if `update_fatigue_limit` is called."
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

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.8,
            weight_decay=0,
        )


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
        log_s = torch.log10(s)
        return self._linear(log_s, x_cluster)


# class LogLogFatigueLimit(AbstractSN):
#     def forward(
#         self,
#         required_cols: Dict[str, torch.Tensor],
#         x_cluster: torch.Tensor,
#         sns: nn.ModuleList,
#     ):
#         s = torch.clamp(torch.abs(required_cols["Relative Maximum Stress"]), min=1e-8)
#         s_sw = torch.clamp(s - self.fatigue_limit[x_cluster], min=1e-8)
#         self.update_fatigue_limit(s, x_cluster)
#         log_s = torch.log10(s_sw)
#         return self._linear(log_s, x_cluster)
#
#     @property
#     def use_fatigue_limit(self):
#         return True


class Sendeckyj(AbstractSN):
    # Sendeckyj, G.P. Fitting models to composite materials fatigue data. In Test Methods and Design Allowables for
    # Fibrous Composites; ASTM STP 734; Chamis, C.C., Ed.; ASTM International: West Conshohocken, PA, USA, 1981; pp.
    # 245–260.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.001))
        self.b = nn.Parameter(
            torch.mul(torch.ones(n_clusters), 0.083)
        )  # beta is b+0.01

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        self.lstsq_input = s
        alpha = self.activ(self.a[x_cluster]) + 1e-4
        beta = self.activ(self.b[x_cluster]) + 0.01
        self.lstsq_output = self.formula(s, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, alpha, beta):
        return torch.log10(
            torch.clamp(
                torch.nan_to_num(
                    torch.pow(1 / s, 1 / beta) - 1 + alpha,
                    nan=1,
                    posinf=1e10,
                    neginf=1e10,
                ),
                min=1e-8,
            )
        ) - torch.log10(alpha)

    @staticmethod
    def required_cols() -> List[str]:
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.05,
            weight_decay=0,
        )


class Hwang(AbstractSN):
    # Hwang, W.; Han, K.S. Fatigue of Composites—Fatigue Modulus Concept and Life Prediction. J. Compos. Mater. 1986,
    # 20, 154–165.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 35))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.21))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, alpha, beta):
        return 1 / beta * (torch.log10(alpha) + torch.log10(1 - s))

    @staticmethod
    def required_cols() -> List[str]:
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.1,
            weight_decay=0,
        )


class Kohout(AbstractSN):
    # Kohout, J.; Vechet, S. A new function for fatigue curves characterization and its multiple merits. Int. J. Fatigue
    # 2001, 23, 175–183.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 776.25))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.0895))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        alpha = self.activ(self.a[x_cluster]) + 1e-8 + 1
        beta = -self.activ(self.b[x_cluster]) - 1e-8
        self.lstsq_input = s
        # this is simplified since alpha << Nf
        self.lstsq_output = self.formula(s, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, alpha, beta):
        return 1 / beta * torch.log10(s) + torch.log10(alpha)

    @staticmethod
    def required_cols() -> List[str]:
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.1,
            weight_decay=0,
        )


class KimZhang(AbstractSN):
    # Kim, H.S.; Zhang, J. Fatigue Damage and Life Prediction of Glass/Vinyl Ester Composites. J. Reinf. Plast. Compos.
    # 2001, 20, 834–848.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 38.44))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 10.809))  # beta is b+1

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        r = required_cols["R-value_UNSCALED"]
        s_max = required_cols["Maximum Stress_UNSCALED"]
        s_min = required_cols["Minimum Stress_UNSCALED"]
        s_ut = torch.clone(s)
        where_use_min = torch.logical_or(r < -1, r > 1)
        where_use_max = torch.logical_not(where_use_min)
        s_ut[where_use_min] = s_min[where_use_min] / s[where_use_min]
        s_ut[where_use_max] = s_max[where_use_max] / s[where_use_max]
        s_ut = torch.clamp(torch.abs(s_ut), min=1e-8)
        log_alpha = -self.activ(self.a[x_cluster]) - 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8 + 1
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, s_ut, log_alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, s_ut, log_alpha, beta):
        return (
            -beta * torch.log10(s_ut)
            - log_alpha
            - torch.log10(beta - 1)
            + torch.log10(
                torch.clamp(
                    torch.nan_to_num(
                        torch.pow(s, 1 - beta), nan=1, posinf=1e10, neginf=1e10
                    )
                    - 1,
                    min=1e-8,
                )
            )
        )

    @staticmethod
    def required_cols():
        return [
            "Relative Maximum Stress_UNSCALED",
            "R-value_UNSCALED",
            "Maximum Stress_UNSCALED",
            "Minimum Stress_UNSCALED",
        ]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.2,
            weight_decay=0,
        )


class KawaiKoizumi(AbstractSN):
    # Kawai, M.; Koizumi, M. Nonlinear constant fatigue life diagrams for carbon/epoxy laminates at room temperature.
    # Compos. Part A Appl. Sci. Manuf. 2007, 38, 2342–2353.
    # The paper uses data at the critical stress ratio (UCS/UTS) to fit the proposed model, and predict data at other
    # stress ratios.
    # In the following implementation, params are fitted using data at all stress ratios. The basic idea of the paper
    # is that data at the critical stress ratio is representative enough to fit the material parameters under the small
    # data assumption, which is not quite the thing for us.
    def _register_params(self, n_clusters=1, **kwargs):
        # the initial values are from Section 4.2 of the paper. Note that in the paper, for different materials, these
        # params vary a lot.
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.003))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 1))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), 8.5))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        # Using relative maximum stress means that the reference strength (\sigma_B) is selected to be |UTS| if
        # |s_max|>|s_min| and |UCS| if |s_max|<|s_min|. This is definitely a simplification of the original formula for
        # implementation, but it is ok since the paper does not give strict restrictions on the reference strength.
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        gamma = self.activ(self.c[x_cluster]) + 1e-8
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, alpha, beta, gamma)
        return self.lstsq_output

    @staticmethod
    def formula(s, alpha, beta, gamma):
        return -torch.log10(alpha) + beta * torch.log10(1 - s) - gamma * torch.log10(s)

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.1,
            weight_decay=0,
        )


class Poursatip(AbstractSN):
    # Poursartip, A.; Ashby, M.F.; Beaumont, P.W.R. The fatigue damage mechanics of carbon fibre composite laminate:
    # I—Development of the model. Compos. Sci. Technol. 1986, 25, 193–218.
    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s_max = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        s_a = torch.clamp(
            torch.abs(required_cols["Relative Peak-to-peak Stress_UNSCALED"]),
            min=1e-5,
            max=2 - 1e-5,
        )
        r = required_cols["R-value_UNSCALED"]
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        gamma = self.activ(self.c[x_cluster]) + 1e-8
        alpha_alter = self.activ(self.d[x_cluster]) + 1e-8
        beta_alter = self.activ(self.e[x_cluster]) + 1e-8
        self.lstsq_input = s_max
        # The third component suggests that the formula can not be applied for r<-1 or r>1. Indeed, the original paper
        # only discuss 0<r<1 where no compression is applied.
        where_valid = torch.logical_and(r > 0, r < 1)
        where_not_valid = torch.logical_not(where_valid)
        self.lstsq_output = torch.ones_like(s_max)
        self.lstsq_output[where_valid] = self.formula(
            s_a[where_valid],
            s_max[where_valid],
            r[where_valid],
            alpha[where_valid],
            beta[where_valid],
            gamma[where_valid],
        )
        # This is equation 30 from the review:
        # Burhan, Ibrahim, and Ho Kim. “S-N Curve Models for Composite Materials Characterisation: An Evaluative
        # Review.” Journal of Composites Science 2, no. 3 (July 2, 2018): 38.
        self.lstsq_output[where_not_valid] = PoursatipSimplified.formula(
            s_max[where_not_valid],
            alpha_alter[where_not_valid],
            beta_alter[where_not_valid],
        )
        return self.lstsq_output

    @staticmethod
    def formula(s_a, s_max, r, alpha, beta, gamma):
        return (
            torch.log10(alpha)
            - beta * torch.log10(s_a)
            + gamma * torch.log10((1 - r) / (1 + r))
            + torch.log10(1 - s_max)
        )

    @staticmethod
    def required_cols():
        return [
            "Relative Peak-to-peak Stress_UNSCALED",
            "Relative Maximum Stress_UNSCALED",
            "R-value_UNSCALED",
        ]

    def _register_params(self, n_clusters=1, **kwargs):
        # Compared to values in the paper, alpha is 3.108x10^4x1.222^p, beta is 6.393, gamma is p.
        # p depends on the stress range: p=1.6 for high stress range and 2.7 for small stress range.
        self.a = nn.Parameter(
            torch.mul(torch.ones(n_clusters), 3.108 * 1e4 * 1.222 ** ((1.6 + 2.7) / 2))
        )
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 6.393))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), (1.6 + 2.7) / 2))
        self.d = nn.Parameter(torch.mul(torch.ones(n_clusters), 17000.0))
        self.e = nn.Parameter(torch.mul(torch.ones(n_clusters), 6.393))


class PoursatipSimplified(AbstractSN):
    # Burhan, Ibrahim, and Ho Kim. “S-N Curve Models for Composite Materials Characterisation: An Evaluative Review.”
    # Journal of Composites Science 2, no. 3 (July 2, 2018): 38.
    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s_max = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s_max
        self.lstsq_output = self.formula(s_max, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s_max, alpha, beta):
        return torch.log10(alpha) - beta * torch.log10(s_max) + torch.log10(1 - s_max)

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED"]

    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 17000.0))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 6.393))


class DAmore(AbstractSN):
    # A. D’Amore, G. Caprino, P. Stupak, J. Zhou, and L. Nicolais. “Effect of Stress Ratio on the Flexural Fatigue
    # Behaviour of Continuous Strand Mat Reinforced Plastics.” Science and Engineering of Composite Materials 5, no. 1
    # (March 1, 1996): 1–8.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        r = required_cols["R-value_UNSCALED"]
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s
        # The applicability when r<0 is not known.
        self.lstsq_output = self.formula(s, r, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, r, alpha, beta):
        return (
            torch.log10(torch.clamp(1 + (1 / s - 1) / alpha / (1 - r), min=1e-8)) / beta
        )

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED", "R-value_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.05,
            weight_decay=0,
        )


class DAmoreSimplified(AbstractSN):
    # Burhan, Ibrahim, and Ho Kim. “S-N Curve Models for Composite Materials Characterisation: An Evaluative Review.”
    # Journal of Composites Science 2, no. 3 (July 2, 2018): 38.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.053))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s
        # The applicability when r<0 is not known.
        self.lstsq_output = self.formula(s, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, alpha, beta):
        return 1 / beta * torch.log10(1 + (1 / s - 1) / alpha)

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.05,
            weight_decay=0,
        )


class Epaarachchi(AbstractSN):
    # Epaarachchi, J.A.; Clausen, P.D. An empirical model for fatigue behavior prediction of glass fibre-reinforced
    # plastic composites for various stress ratios and test frequencies. Compos. Part A Appl. Sci. Manuf. 2003, 34,
    # 313–326.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.053))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.0007))
        self.d = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.245))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        r = required_cols["R-value_UNSCALED"]
        f = required_cols["Frequency_UNSCALED"]
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8
        alpha_alter = self.activ(self.c[x_cluster]) + 1e-8
        beta_alter = self.activ(self.d[x_cluster]) + 1e-8

        where_valid = r < 1
        where_not_valid = torch.logical_not(where_valid)
        s_max = required_cols["Maximum Stress_UNSCALED"]
        s_min = required_cols["Minimum Stress_UNSCALED"]

        s_used = torch.clone(s)
        s_ut = torch.clone(s)
        where_use_min = torch.logical_or(r < -1, r > 1)
        where_use_max = torch.logical_not(where_use_min)
        s_used[where_use_min] = s_min[where_use_min]
        s_used[where_use_max] = s_max[where_use_max]
        s_used = torch.abs(s_used)
        s_ut[where_use_min] = s_min[where_use_min] / s[where_use_min]
        s_ut[where_use_max] = s_max[where_use_max] / s[where_use_max]
        s_ut = torch.abs(s_ut)
        # Verification: torch.allclose(s_used/s_ut, s)

        self.lstsq_input = s
        self.lstsq_output = torch.ones_like(s)
        self.lstsq_output[where_valid] = self.formula(
            s[where_valid],
            r[where_valid],
            f[where_valid],
            alpha[where_valid],
            beta[where_valid],
        )
        self.lstsq_output[where_not_valid] = EpaarachchiSimplified.formula(
            s_used[where_not_valid],
            s_ut[where_not_valid],
            alpha_alter[where_not_valid],
            beta[where_not_valid],
        )
        return self.lstsq_output

    @staticmethod
    def formula(s, r, f, alpha, beta):
        return (
            torch.log10(1 / s - 1)
            - torch.log10(alpha)
            - 0.6 * torch.log10(s)
            - 1.6 * torch.log10(1 - r)
            + beta * torch.log10(f)
        ) / beta

    @staticmethod
    def required_cols():
        return [
            "Relative Maximum Stress_UNSCALED",
            "Frequency_UNSCALED",
            "R-value_UNSCALED",
            "Maximum Stress_UNSCALED",
            "Minimum Stress_UNSCALED",
        ]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.01,
            weight_decay=0,
        )


class EpaarachchiSimplified(AbstractSN):
    # Burhan, Ibrahim, and Ho Kim. “S-N Curve Models for Composite Materials Characterisation: An Evaluative Review.”
    # Journal of Composites Science 2, no. 3 (July 2, 2018): 38.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.0007))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.245))

    def forward(
        self,
        required_cols: Dict[str, torch.Tensor],
        x_cluster: torch.Tensor,
        sns: nn.ModuleList,
    ):
        s = torch.clamp(
            torch.abs(required_cols["Relative Maximum Stress_UNSCALED"]),
            min=1e-5,
            max=1 - 1e-5,
        )
        r = required_cols["R-value_UNSCALED"]
        alpha = self.activ(self.a[x_cluster]) + 1e-8
        beta = self.activ(self.b[x_cluster]) + 1e-8

        s_max = required_cols["Maximum Stress_UNSCALED"]
        s_min = required_cols["Minimum Stress_UNSCALED"]

        s_used = torch.clone(s)
        s_ut = torch.clone(s)
        where_use_min = torch.logical_or(r < -1, r > 1)
        where_use_max = torch.logical_not(where_use_min)
        s_used[where_use_min] = s_min[where_use_min]
        s_used[where_use_max] = s_max[where_use_max]
        s_used = torch.abs(s_used)
        s_ut[where_use_min] = s_min[where_use_min] / s[where_use_min]
        s_ut[where_use_max] = s_max[where_use_max] / s[where_use_max]
        s_ut = torch.abs(s_ut)
        # Verification: torch.allclose(s_used/s_ut, s)

        self.lstsq_input = s
        self.lstsq_output = self.formula(s_used, s_ut, alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, s_ut, alpha, beta):
        return (
            1
            / beta
            * torch.log10(
                torch.clamp(s_ut - s, min=1e-8) / alpha / torch.pow(s, 1.6) + 1
            )
        )

    @staticmethod
    def required_cols():
        return [
            "Relative Maximum Stress_UNSCALED",
            "R-value_UNSCALED",
            "Maximum Stress_UNSCALED",
            "Minimum Stress_UNSCALED",
        ]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.01,
            weight_decay=0,
        )


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
        self.required_cols: List[str] = list(sorted(set(required_cols)))
        self.required_indices = [
            datamodule.cont_feature_names.index(col.split("_UNSCALED")[0])
            for col in self.required_cols
        ]
        self.zero_slip = [
            datamodule.get_zero_slip(col.split("_UNSCALED")[0])
            for col in self.required_cols
        ]
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

    def extract_cols(self, x, derived_tensors):
        unscaled = derived_tensors["Unscaled"]
        return {
            col: x[:, idx] - zero_slip
            if not col.endswith("_UNSCALED")
            else unscaled[:, idx]
            for col, idx, zero_slip in zip(
                self.required_cols, self.required_indices, self.zero_slip
            )
        }

    def forward(self, x, clustering_features, derived_tensors):
        required_cols = self.extract_cols(x, derived_tensors)
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
