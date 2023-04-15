import torch
from typing import List
from torch import nn
from src.utils import torch_with_grad
import torch.nn.functional as F


def BiasLoss(base_loss: torch.Tensor, w: torch.Tensor):
    return (base_loss * w).mean()


def StressGradLoss(y_pred, s, base_loss, *data):
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
    base_loss = base_loss + grad_loss + 1e3 * grad_2_loss
    return base_loss


def ConsGradLoss(balance=1e3, *data, **kwargs):
    with torch_with_grad():
        base_loss: torch.Tensor = kwargs["base_loss"]
        y_pred: torch.Tensor = kwargs["y_pred"]
        n_cont: int = kwargs["n_cont"]
        cont_feature_names: List[str] = kwargs["cont_feature_names"]
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

    base_loss = base_loss + torch.mul(torch.sum(feature_loss), balance)
    return base_loss
