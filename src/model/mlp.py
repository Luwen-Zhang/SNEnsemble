from src.utils import *
from src.model import TorchModel, AbstractNN
from .base import get_sequential
import torch.nn as nn


class MLP(TorchModel):
    def __init__(self, trainer, layers=None, *args, **kwargs):
        super(MLP, self).__init__(trainer, *args, **kwargs)
        self.layers = layers

    def _get_program_name(self):
        return "MLP"

    def _new_model(self, model_name, verbose, **kwargs):
        return _MLPNN(
            len(self.trainer.cont_feature_names),
            len(self.trainer.label_name),
            self.trainer.args["layers"] if self.layers is None else self.layers,
            trainer=self.trainer,
        )

    def _get_model_names(self):
        return ["MLP"]


class _MLPNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer):
        super(_MLPNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs
        self.net = get_sequential(
            layers, num_inputs, num_outputs, nn.ReLU, norm_type="layer"
        )
        self.nets = nn.ModuleList(
            [
                get_sequential(layers, dims[-1], 1, nn.ReLU, norm_type="layer")
                for dims in self.derived_feature_dims
            ]
        )
        self.weight = get_sequential(
            [32], len(self.nets) + 1, num_outputs, nn.ReLU, norm_type="layer"
        )

    def _forward(self, x, derived_tensors):
        if len(derived_tensors) > 0:
            x = [self.net(x)] + [
                net(y.to(torch.float32))
                for net, y in zip(self.nets, derived_tensors.values())
            ]
            x = torch.concat(x, dim=1)
            output = self.weight(x)
        else:
            output = self.net(x)

        return output
