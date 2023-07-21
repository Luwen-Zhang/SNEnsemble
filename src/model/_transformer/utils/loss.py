import torch
from torch import nn
from tabensemb.utils import torch_with_grad
import torch.nn.functional as F


class BalancedLoss(nn.Module):
    def __init__(self):
        super(BalancedLoss, self).__init__()
        self.register_buffer("weight", torch.tensor([1.0]))
        self.exp_avg_factor = 0.1

    def forward(self, **kwargs):
        base_loss = kwargs["base_loss"]
        my_loss = self._forward(**kwargs)
        if self.training:
            with torch.no_grad():
                self.weight = (
                    self.exp_avg_factor * base_loss / (my_loss + 1e-8) * 0.1
                    + (1 - self.exp_avg_factor) * self.weight
                )
        return self.weight * my_loss + base_loss

    def _forward(self, **kwargs):
        raise NotImplementedError


class BiasLoss(nn.Module):
    def forward(self, base_loss: torch.Tensor, w: torch.Tensor):
        return (base_loss * w).mean()


class StressGradLoss(BalancedLoss):
    def __init__(self):
        super(StressGradLoss, self).__init__()
        self.register_buffer("grad_2_weight", torch.tensor([1000.0]))

    def _forward(self, y_pred, s, **kwargs):
        with torch_with_grad():
            grad_s = torch.autograd.grad(
                outputs=y_pred,
                inputs=s,
                grad_outputs=torch.ones_like(y_pred),
                retain_graph=True,
                create_graph=True,  # True to compute higher order derivatives, and is more expensive.
            )[0].view(-1, 1)
            grad_s_2 = torch.autograd.grad(
                outputs=grad_s,
                inputs=s,
                grad_outputs=torch.ones_like(grad_s),
                retain_graph=True,
                create_graph=False,  # True to compute higher order derivatives, and is more expensive.
            )[0].view(-1, 1)
        grad_loss = torch.mean(F.relu(grad_s) ** 2)
        grad_2_loss = torch.mean(F.relu(-grad_s_2) ** 2)
        if self.training:
            with torch.no_grad():
                self.grad_2_weight = (
                    self.exp_avg_factor * grad_loss / (grad_2_loss + 1e-8)
                    + (1 - self.exp_avg_factor) * self.grad_2_weight
                )
        return grad_loss + self.grad_2_weight * grad_2_loss


class ConsGradLoss(BalancedLoss):
    def _forward(self, data, y_pred, n_cont, cont_feature_names, balance=1e3, **kwargs):
        with torch_with_grad():
            implemented_features = ["Relative Mean Stress"]
            feature_idx_mapping = {
                x: cont_feature_names.index(x)
                for x in implemented_features
                if x in cont_feature_names
            }
            grad = torch.autograd.grad(
                outputs=y_pred,
                inputs=data[0],
                grad_outputs=torch.ones_like(y_pred),
                retain_graph=True,
                create_graph=False,  # True to compute higher order derivatives, and is more expensive.
            )[0]
            feature_loss = torch.zeros((n_cont,))
            for feature, idx in feature_idx_mapping.items():
                grad_feature = grad[:, idx]
                if feature == "Relative Mean Stress":
                    feature_loss[idx] = torch.mean(F.relu(grad_feature) ** 2)
                else:
                    raise Exception(
                        f"Operation on the gradient of feature {feature} is not implemented."
                    )

        return torch.mul(torch.sum(feature_loss), balance)
