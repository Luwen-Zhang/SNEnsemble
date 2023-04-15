import torch
from typing import List
from torch import nn
from src.utils import torch_with_grad
import torch.nn.functional as F


def BiasLoss(base_loss: torch.Tensor, w: torch.Tensor):
    return (base_loss * w).mean()


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
