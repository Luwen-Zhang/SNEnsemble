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

        self.net = self.get_sequential(layers, num_inputs, num_outputs, nn.ReLU)

    def forward(self, x, additional_tensors):
        x = self.net(x)
        output = x

        return output

    def get_sequential(self, layers, n_inputs, n_outputs, act_func):
        net = nn.Sequential()
        net.add_module("input", nn.Linear(n_inputs, layers[0]))
        net.add_module("activate", act_func())
        for idx in range(len(layers) - 1):
            net.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
            net.add_module("activate" + str(idx), act_func())
        net.add_module("output", nn.Linear(layers[-1], n_outputs))

        net.apply(init_weights)

        return net