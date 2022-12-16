import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


class NN(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers):
        super(NN, self).__init__()
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)

    def forward(self, x, additional_tensors):
        x = self.net(x)
        output = x

        return output


class ThisWorkNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers, activated_sn):
        super(ThisWorkNN, self).__init__()
        num_inputs = n_inputs
        num_outputs = n_outputs

        self.net = get_sequential(layers, num_inputs, num_outputs, nn.ReLU)
        self.activated_sn = nn.ModuleList(activated_sn)
        self.component_weights = nn.Parameter(
            torch.Tensor([0 for x in activated_sn]).view(-1, 1),
            requires_grad=True,
        )

    def forward(self, x, additional_tensors):
        preds = torch.concat(
            [sn(x, additional_tensors) for sn in self.activated_sn],
            dim=1,
        )

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
