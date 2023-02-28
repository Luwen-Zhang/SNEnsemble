import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


class AbstractNN(nn.Module):
    def __init__(self, trainer):
        super(AbstractNN, self).__init__()
        self.derived_feature_names = list(trainer.derived_data.keys())
        self.derived_feature_dims = trainer.get_derived_data_sizes()

    def forward(self, *tensors):
        x = tensors[0]
        additional_tensors = tensors[1:]
        derived_tensors = {}
        for tensor, name in zip(additional_tensors, self.derived_feature_names):
            derived_tensors[name] = tensor
        return self._forward(x, derived_tensors)

    def _forward(self, x, derived_tensors):
        raise NotImplementedError


class NN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, trainer):
        super(NN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.nets = [
            get_sequential(layers, dims[-1], 1, nn.ReLU)
            for dims in self.derived_feature_dims
        ]
        self.weight = get_sequential([32], len(self.nets) + 1, num_outputs, nn.ReLU)

    def _forward(self, x, derived_tensors):
        if len(derived_tensors) > 0:
            x = [self.net(x)] + [
                net(y) for net, y in zip(self.nets, derived_tensors.values())
            ]
            x = torch.concat(x, dim=1)
            output = self.weight(x)
        else:
            output = self.net(x)

        return output


class ThisWorkNN(AbstractNN):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn, trainer):
        super(ThisWorkNN, self).__init__(trainer)
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.stress_unrelated_features_idx = activated_sn[
            0
        ].stress_unrelated_features_idx
        self.component_weights = get_sequential(
            [16, 64, 128, 64, 16],
            len(self.stress_unrelated_features_idx),
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
                torch.abs(
                    self.component_weights(x[:, self.stress_unrelated_features_idx])
                ),
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
        self.stress_unrelated_features_idx = activated_sn[
            0
        ].stress_unrelated_features_idx
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


def get_sequential(layers, n_inputs, n_outputs, act_func):
    net = nn.Sequential()
    net.add_module("input", nn.Linear(n_inputs, layers[0]))
    net.add_module("activate", act_func())
    for idx in range(len(layers) - 1):
        net.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
        net.add_module("activate" + str(idx), act_func())
    net.add_module("output", nn.Linear(layers[-1], n_outputs))

    net.apply(init_weights)
    return net
