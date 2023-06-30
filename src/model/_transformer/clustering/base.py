import sys
import torch
from torch import nn
import numpy as np
import inspect
from src.model._transformer.clustering.common.base import AbstractClustering
from typing import Dict, List


def clamp_for_log(x):
    return torch.clamp(x, min=1e-8)


def clamp_overflow(x):
    return torch.nan_to_num(x, nan=1, posinf=1e10, neginf=1e10)


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
        s = clamp_for_log(torch.abs(required_cols["Relative Maximum Stress"]))
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
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(0.001)))
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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 0.01
        self.lstsq_output = self.formula(s, log_alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, log_alpha, beta):
        return (
            torch.log10(
                clamp_for_log(
                    clamp_overflow(
                        torch.pow(1 / s, 1 / beta) - 1 + clamp_overflow(10**log_alpha)
                    )
                )
            )
            - log_alpha
        )

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
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(35)))
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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, log_alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, log_alpha, beta):
        return 1 / beta * (log_alpha + torch.log10(1 - s))

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
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(776.25)))
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
        log_alpha = self.a[x_cluster]
        beta = -self.activ(self.b[x_cluster]) - 1e-8
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, log_alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s, log_alpha, beta):
        return torch.log10(
            clamp_for_log(
                clamp_overflow(
                    10 ** (1 / beta * torch.log10(s) + log_alpha)
                    - clamp_overflow(10**log_alpha)
                )
            )
        )

    @staticmethod
    def required_cols() -> List[str]:
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.05,
            weight_decay=0,
        )


class KimZhang(AbstractSN):
    # Kim, H.S.; Zhang, J. Fatigue Damage and Life Prediction of Glass/Vinyl Ester Composites. J. Reinf. Plast. Compos.
    # 2001, 20, 834–848.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), -38.44))
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
        s_ut = clamp_for_log(torch.abs(s_ut))
        log_alpha = self.a[x_cluster]
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
            + torch.log10(clamp_for_log(clamp_overflow(torch.pow(s, 1 - beta)) - 1))
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
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(0.003)))
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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 1e-8
        gamma = self.activ(self.c[x_cluster]) + 1e-8
        self.lstsq_input = s
        self.lstsq_output = self.formula(s, log_alpha, beta, gamma)
        return self.lstsq_output

    @staticmethod
    def formula(s, log_alpha, beta, gamma):
        return -log_alpha + beta * torch.log10(1 - s) - gamma * torch.log10(s)

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.8,
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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 1e-8
        gamma = self.activ(self.c[x_cluster]) + 1e-8
        log_alpha_alter = self.d[x_cluster]
        beta_alter = self.activ(self.e[x_cluster]) + 1e-8
        self.lstsq_input = s_max
        # The third component suggests that the formula can not be applied for r<-1 or r>1.
        where_valid = torch.logical_and(r > -1, r < 1)
        where_not_valid = torch.logical_not(where_valid)
        self.lstsq_output = torch.ones_like(s_max)
        self.lstsq_output[where_valid] = self.formula(
            s_a[where_valid],
            s_max[where_valid],
            r[where_valid],
            log_alpha[where_valid],
            beta[where_valid],
            gamma[where_valid],
        )
        # This is equation 30 from the review:
        # Burhan, Ibrahim, and Ho Kim. “S-N Curve Models for Composite Materials Characterisation: An Evaluative
        # Review.” Journal of Composites Science 2, no. 3 (July 2, 2018): 38.
        self.lstsq_output[where_not_valid] = PoursatipSimplified.formula(
            s_max[where_not_valid],
            log_alpha_alter[where_not_valid],
            beta_alter[where_not_valid],
        )
        return self.lstsq_output

    @staticmethod
    def formula(s_a, s_max, r, log_alpha, beta, gamma):
        return (
            log_alpha
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
            torch.mul(
                torch.ones(n_clusters),
                np.log10(3.108 * 1e4 * 1.222 ** ((1.6 + 2.7) / 2)),
            )
        )
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 6.393))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), (1.6 + 2.7) / 2))
        self.d = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(17000.0)))
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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 1e-8
        self.lstsq_input = s_max
        self.lstsq_output = self.formula(s_max, log_alpha, beta)
        return self.lstsq_output

    @staticmethod
    def formula(s_max, log_alpha, beta):
        return log_alpha - beta * torch.log10(s_max) + torch.log10(1 - s_max)

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED"]

    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(17000.0)))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 6.393))


class DAmore(AbstractSN):
    # A. D’Amore, G. Caprino, P. Stupak, J. Zhou, and L. Nicolais. “Effect of Stress Ratio on the Flexural Fatigue
    # Behaviour of Continuous Strand Mat Reinforced Plastics.” Science and Engineering of Composite Materials 5, no. 1
    # (March 1, 1996): 1–8.
    def _register_params(self, n_clusters=1, **kwargs):
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.d = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))

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
        alpha_alter = self.activ(self.c[x_cluster]) + 1e-8
        beta_alter = self.activ(self.d[x_cluster]) + 1e-8
        self.lstsq_input = s
        where_r = torch.logical_and(r > -1, r < 1)
        where_1_r = torch.logical_not(where_r)
        self.lstsq_output = torch.ones_like(s)
        self.lstsq_output[where_r] = self.formula(
            s[where_r], r[where_r], alpha[where_r], beta[where_r]
        )
        # Caprino, G., and G. Giorleo. “Fatigue Lifetime of Glass Fabric/Epoxy Composites.” Composites Part A:
        # Applied Science and Manufacturing 30, no. 3 (March 1, 1999): 299–304.
        self.lstsq_output[where_1_r] = self.formula(
            s[where_1_r],
            1 / r[where_1_r],
            alpha_alter[where_1_r],
            beta_alter[where_1_r],
        )
        return self.lstsq_output

    @staticmethod
    def formula(s, r, alpha, beta):
        return torch.log10(clamp_for_log(1 + (1 / s - 1) / alpha / (1 - r))) / beta

    @staticmethod
    def required_cols():
        return ["Relative Maximum Stress_UNSCALED", "R-value_UNSCALED"]

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.01,
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
        self.a = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(0.053)))
        self.b = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))
        self.c = nn.Parameter(torch.mul(torch.ones(n_clusters), np.log10(0.053)))
        self.d = nn.Parameter(torch.mul(torch.ones(n_clusters), 0.2))

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
        log_alpha = self.a[x_cluster]
        beta = self.activ(self.b[x_cluster]) + 1e-8
        log_alpha_alter = self.c[x_cluster]
        beta_alter = self.activ(self.d[x_cluster]) + 1e-8

        where_r = torch.logical_and(r > -1, r < 1)
        where_1_r = torch.logical_not(where_r)

        self.lstsq_input = s
        self.lstsq_output = torch.ones_like(s)
        self.lstsq_output[where_r] = self.formula(
            s[where_r],
            r[where_r],
            f[where_r],
            log_alpha[where_r],
            beta[where_r],
        )
        # Follow a similar approach of Caprino et al for D'Amore's model.
        self.lstsq_output[where_1_r] = self.formula(
            s[where_1_r],
            1 / r[where_1_r],
            f[where_1_r],
            log_alpha_alter[where_1_r],
            beta_alter[where_1_r],
        )
        return self.lstsq_output

    @staticmethod
    def formula(s, r, f, log_alpha, beta):
        return (
            torch.log10(1 / s - 1)
            - log_alpha
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
            torch.log10(clamp_for_log(s_ut - s) / alpha / torch.pow(s, 1.6) + 1) / beta
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
            lr=0.005,
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


if __name__ == "__main__":
    s_max = 10 ** torch.tensor(
        [
            2.866007194244604,
            2.8363309352517985,
            2.803956834532374,
            2.76978417266187,
            2.7320143884892087,
            2.7068345323741005,
            2.918165467625899,
            2.6906474820143886,
            2.672661870503597,
            2.653776978417266,
            2.634892086330935,
            2.6142086330935252,
            2.593525179856115,
        ]
    )
    s_ut = 829
    s_max_grid = torch.linspace(torch.min(s_max), torch.max(s_max), 100)
    s_grid = s_max_grid / s_ut
    s = s_max / s_ut

    truth = [
        3.3962264150943398,
        3.883647798742138,
        4.276729559748428,
        4.622641509433962,
        4.8742138364779874,
        5.220125786163522,
        0.015723270440251458,
        5.298742138364779,
        5.4559748427672945,
        5.880503144654086,
        5.9905660377358485,
        6.069182389937107,
        6.682389937106918,
    ]
    sendeckyj = Sendeckyj.formula(s_grid, torch.tensor([-3]), torch.tensor([0.093]))
    hwang = Hwang.formula(s_grid, torch.tensor([np.log10(35)]), torch.tensor([0.21]))
    kohout = Kohout.formula(
        s_grid, torch.tensor([np.log10(776.25)]), torch.tensor([-0.0895])
    )
    kimzhang = KimZhang.formula(
        s_grid, torch.tensor([s_ut]), torch.tensor([-38.44]), torch.tensor([11.809])
    )
    import matplotlib.pyplot as plt

    plt.figure()
    y = torch.log10(s_max).numpy()
    y_grid = torch.log10(s_max_grid).numpy()
    plt.scatter(truth, y, label="truth")
    plt.plot(sendeckyj.numpy(), y_grid, label="sendeckyj")
    plt.plot(hwang.numpy(), y_grid, label="hwang")
    plt.plot(kohout.numpy(), y_grid, label="kohout")
    plt.plot(kimzhang.numpy(), y_grid, label="kimzhang")
    plt.xlim([0, 10])
    plt.ylim([2.5, 3])
    plt.legend()
    plt.show()

    s_max = torch.tensor(
        [
            828.8793103448274,
            734.4827586206895,
            685.3448275862067,
            636.206896551724,
            587.0689655172413,
            537.9310344827585,
            508.1896551724136,
            488.7931034482757,
            468.10344827586187,
            448.70689655172396,
            429.31034482758605,
            409.91379310344814,
            390.51724137931024,
        ]
    )
    s_ut = 829
    s_max_grid = torch.linspace(torch.min(s_max), torch.max(s_max), 100)
    s_grid = s_max_grid / s_ut
    s = s_max / s_ut

    truth = [
        -0.015804597701150058,
        3.3716897678611657,
        3.864562204192022,
        4.2593954248366,
        4.605209037638043,
        4.869323304034258,
        5.214080459770114,
        5.280496393959882,
        5.445021974306962,
        5.870915032679739,
        5.986350574712643,
        6.052766508902411,
        6.674737998647735,
    ]
    kawaikoizumi = KawaiKoizumi.formula(
        s_grid, torch.tensor([-4]), torch.tensor([1.0]), torch.tensor([8.5])
    )
    poursatip = PoursatipSimplified.formula(
        s_grid, torch.tensor([np.log10(17000.0)]), torch.tensor([6.393])
    )
    damore = DAmoreSimplified.formula(
        s_grid, torch.tensor([0.053]), torch.tensor([0.2])
    )
    epaarachchi = EpaarachchiSimplified.formula(
        s_max_grid, s_ut, torch.tensor([0.0007]), torch.tensor([0.245])
    )

    plt.figure()
    y = s_max.numpy()
    y_grid = s_max_grid.numpy()
    plt.scatter(truth, y, label="truth")
    plt.plot(kawaikoizumi.numpy(), y_grid, label="kawaikoizumi")
    plt.plot(kimzhang.numpy(), y_grid, label="kimzhang")
    plt.plot(poursatip.numpy(), y_grid, label="poursatip")
    plt.plot(damore.numpy(), y_grid, label="damore")
    plt.plot(epaarachchi.numpy(), y_grid, label="epaarachchi")
    plt.xlim([-1, 9])
    plt.ylim([300, 900])
    plt.legend()
    plt.show()

    s_max = torch.tensor(
        [
            2075,
            2043.75,
            2018.75,
            1975,
            1337.5,
            1350,
            1293.75,
            1300,
            968.7500000000002,
            962.5000000000002,
            762.5,
            756.25,
            756.25,
            587.5,
            587.5,
            587.5,
            487.5000000000002,
            487.5000000000002,
            487.5000000000002,
            487.5000000000002,
            387.5,
            387.5,
            381.25,
        ]
    )
    s_ut = 2013
    s_max_grid = torch.linspace(torch.min(s_max), torch.max(s_max), 100)
    s_grid = s_max_grid / s_ut
    r = 0.1
    f = 1  # not known
    s_a_grid = s_max_grid * (1 - r)
    s = s_max / s_ut

    truth = [
        0.015479876160990447,
        0.015479876160990447,
        0.015479876160990447,
        0.015479876160990447,
        2.195046439628483,
        2.6408668730650153,
        2.4427244582043337,
        2.517027863777089,
        3.111455108359133,
        3.260061919504644,
        3.8297213622291024,
        3.9287925696594432,
        4.027863777089783,
        4.547987616099071,
        4.696594427244582,
        4.845201238390095,
        4.7213622291021675,
        4.969040247678018,
        5.142414860681114,
        5.3653250773993815,
        5.340557275541796,
        5.662538699690403,
        6.058823529411766,
    ]
    kimzhang = KimZhang.formula(
        s_grid, torch.tensor([s_ut]), torch.tensor([-26.505]), torch.tensor([7.3813])
    )
    kawaikoizumi = KawaiKoizumi.formula(
        s_grid, torch.tensor([-2]), torch.tensor([1.0]), torch.tensor([5.5])
    )
    poursatip = Poursatip.formula(
        s_a_grid,
        s_grid,
        torch.tensor([r]),
        torch.tensor([16]),
        torch.tensor([3.9]),
        torch.tensor([6.393]),
    )
    poursatip_simp = PoursatipSimplified.formula(
        s_grid, torch.tensor([np.log10(60.0)]), torch.tensor([6.393])
    )
    damore = DAmore.formula(
        s_grid, torch.tensor([r]), torch.tensor([0.033]), torch.tensor([0.44])
    )
    damore_simp = DAmoreSimplified.formula(
        s_grid, torch.tensor([0.033]), torch.tensor([0.44])
    )
    epaarachchi = Epaarachchi.formula(
        s_grid,
        torch.tensor([r]),
        torch.tensor([f]),
        torch.tensor([-1.5]),
        torch.tensor([0.51]),
    )
    epaarachchi_simp = EpaarachchiSimplified.formula(
        s_max_grid, s_ut, torch.tensor([0.00035]), torch.tensor([0.51])
    )

    plt.figure()
    y = s_max.numpy()
    y_grid = s_max_grid.numpy()
    plt.scatter(truth, y, label="truth")
    plt.plot(kawaikoizumi.numpy(), y_grid, label="kawaikoizumi")
    plt.plot(kimzhang.numpy(), y_grid, label="kimzhang")
    plt.plot(poursatip.numpy(), y_grid, label="poursatip")
    plt.plot(poursatip_simp.numpy(), y_grid, label="poursatip simp")
    plt.plot(damore.numpy(), y_grid, label="damore")
    plt.plot(damore_simp.numpy(), y_grid, label="damore simp")
    plt.plot(epaarachchi.numpy(), y_grid, label="epaarachchi")
    plt.plot(epaarachchi_simp.numpy(), y_grid, label="epaarachchi simp")
    plt.xlim([-1, 7])
    plt.ylim([350, 2100])
    plt.legend()
    plt.show()
