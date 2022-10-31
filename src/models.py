import torch
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


class NN(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers, use_sequence):
        super(NN, self).__init__()
        num_inputs = n_inputs
        num_outputs = n_outputs
        self.use_sequence = use_sequence

        if self.use_sequence:
            self.net = self.get_sequential(layers, num_inputs, num_outputs*4, nn.ReLU)

            self.net_layers = self.get_sequential(layers, 4, num_outputs*4, nn.ReLU)

            self.net_post = self.get_sequential(layers, num_outputs*8, num_outputs, nn.ReLU)
        else:
            self.net = self.get_sequential(layers, num_inputs, num_outputs, nn.ReLU)

    def forward(self, x, additional_tensors):
        x = self.net(x)
        if self.use_sequence:
            y = self.net_layers(additional_tensors[0])
            output = self.net_post(torch.cat([x, y], dim=1))
        else:
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