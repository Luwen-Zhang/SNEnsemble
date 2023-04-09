from src.utils import *
from src.model import TorchModel, AbstractNN
from .base import get_sequential
from copy import deepcopy as cp
import torch.nn as nn


class ThisWork(TorchModel):
    def __init__(
        self,
        trainer,
        manual_activate=None,
        *args,
        **kwargs,
    ):
        super(ThisWork, self).__init__(trainer, *args, **kwargs)
        self.activated_sn = None
        self.manual_activate = manual_activate
        from src.model._transformer.sn_formulas import sn_mapping

        if self.manual_activate is not None:
            for sn in self.manual_activate:
                if sn not in sn_mapping.keys():
                    raise Exception(f"SN model {sn} is not implemented or activated.")

    def _get_program_name(self):
        return "ThisWork"

    def _new_model(self, model_name, verbose, **kwargs):
        if self.activated_sn is None:
            self.activated_sn = self._get_activated_sn()
        return ThisWorkNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.args["layers"],
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        )

    def _get_activated_sn(self):
        from src.model._transformer.sn_formulas import sn_mapping

        activated_sn = []
        sn_coeff_vars_idx = [
            self.trainer.cont_feature_names.index(name)
            for name, type in self.trainer.args["feature_names_type"].items()
            if self.trainer.args["feature_types"][type] == "Material"
        ]
        for key, sn in sn_mapping.items():
            if sn.test_sn_vars(
                self.trainer.cont_feature_names,
                list(self.trainer.derived_data.keys()),
            ) and (self.manual_activate is None or key in self.manual_activate):
                activated_sn.append(
                    sn(
                        cont_feature_names=self.trainer.cont_feature_names,
                        derived_feature_names=list(self.trainer.derived_data.keys()),
                        s_zero_slip=self.trainer.get_zero_slip(sn.get_sn_vars()[0]),
                        sn_coeff_vars_idx=sn_coeff_vars_idx,
                    )
                )
        print(f"Activated SN models: {[sn.__class__.__name__ for sn in activated_sn]}")
        return activated_sn

    def _get_model_names(self):
        return ["ThisWork"]


class ThisWorkRidge(ThisWork):
    def _get_program_name(self):
        return "ThisWorkRidge"

    def _new_model(self, model_name, verbose, **kwargs):
        if self.activated_sn is None:
            self.activated_sn = self._get_activated_sn()
        return ThisWorkRidgeNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.args["layers"],
            activated_sn=self.activated_sn,
            trainer=self.trainer,
        )

    def _train_step(self, model, train_loader, optimizer, **kwargs):
        model.train()
        avg_loss = 0
        for idx, tensors in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(*([data] + additional_tensors))
            self.ridge(model, yhat)
            loss = model.loss_fn(
                yhat, y, model, *([data] + additional_tensors), **kwargs
            )
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * len(y)

        avg_loss /= len(train_loader.dataset)
        return avg_loss

    def ridge(self, model, yhat, alpha=0.2):
        # https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
        X = model.preds
        y = yhat.view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        lhs = X.T @ X
        rhs = X.T @ y
        ridge = alpha * torch.eye(lhs.shape[0])
        w = torch.linalg.lstsq(rhs, lhs + ridge).solution
        from copy import copy

        model.component_weights = copy(w.T)
        # print(model.component_weights)


class ThisWorkNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn, trainer):
        super(ThisWorkNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.sn_coeff_vars_idx = activated_sn[0].sn_coeff_vars_idx
        self.component_weights = get_sequential(
            [16, 64, 128, 64, 16],
            len(self.sn_coeff_vars_idx),
            len(self.activated_sn),
            nn.ReLU,
        )

    def _forward(self, x, derived_tensors):
        preds = torch.concat(
            [sn(x, derived_tensors) for sn in self.activated_sn],
            dim=1,
        )

        output = torch.mul(
            preds,
            nn.functional.normalize(
                torch.abs(self.component_weights(x[:, self.sn_coeff_vars_idx])),
                p=1,
                dim=1,
            ),
        )  # element wise multiplication
        output = torch.sum(output, dim=1).view(-1, 1)
        return output


class ThisWorkRidgeNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn, trainer):
        super(ThisWorkRidgeNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.sn_coeff_vars_idx = activated_sn[0].sn_coeff_vars_idx
        self.component_weights = torch.ones(
            [len(self.activated_sn), 1], requires_grad=False
        )
        self.preds = None

    def _forward(self, x, derived_tensors):
        preds = torch.concat(
            [sn(x, derived_tensors) for sn in self.activated_sn],
            dim=1,
        )
        self.preds = preds
        # print(preds.shape, self.component_weights.shape)
        output = torch.matmul(preds, self.component_weights)
        return output
