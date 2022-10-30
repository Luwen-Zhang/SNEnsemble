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

        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(num_inputs, layers[0]))
        self.net.add_module("ReLU", nn.ReLU())
        for idx in range(len(layers) - 1):
            self.net.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
            self.net.add_module("ReLU" + str(idx), nn.ReLU())
        self.net.add_module("output", nn.Linear(layers[-1], num_outputs))
        self.net.apply(init_weights)

        if self.use_sequence:
            self.net_layers = nn.Sequential()
            self.net_layers.add_module("input", nn.Linear(4, layers[0]))
            self.net_layers.add_module("ReLU", nn.ReLU())
            for idx in range(len(layers) - 1):
                self.net_layers.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
                self.net_layers.add_module("ReLU" + str(idx), nn.ReLU())
            self.net_layers.add_module("output", nn.Linear(layers[-1], num_outputs))

            self.net_layers.apply(init_weights)

            self.net_post = nn.Sequential()
            self.net_post.add_module('input', nn.Linear(num_outputs * 2, num_outputs))

    def forward(self, x, additional_tensors):
        x = self.net(x)
        if self.use_sequence and len(additional_tensors) == 1:
            y = self.net_layers(additional_tensors[0])
            output = self.net_post(torch.cat([x, y], dim=1))
        else:
            output = x

        return output
