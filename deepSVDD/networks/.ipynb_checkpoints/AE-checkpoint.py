import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    # layer_dims should include dimension of input
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        inp = layer_dims[0]
        _pairs = []
        for i in range(1,len(layer_dims)):
            op = layer_dims[i]
            layers.append(nn.Linear(inp,op))
            layers.append(nn.Tanh())
            _pairs.append([op,inp])
            inp = op
        for i in range(len(_pairs)-1,-1):
            inp = _pairs[i][0]
            op = _pairs[i][1]
            layers.append(nn.Linear(inp, op))
            if i == 0:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Tanh())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x


class FC_enc(nn.Module):
    # layer_dims should include dimension of input
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        inp = layer_dims[0]
        self.rep_dim = layer_dims[-1]
        _pairs = []
        for i in range(1,len(layer_dims)):
            op = layer_dims[i]
            layers.append(nn.Linear(inp, op, bias=False))
            layers.append(nn.ReLU())
            inp = op
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x

class FC_dec(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        inp = layer_dims[0]
        self.rep_dim = layer_dims[-1]
        _pairs = []
        for i in range(1,len(layer_dims)):
            op = layer_dims[i]
            layers.append(nn.Linear(inp, op, bias=False))
            if i!= len(layer_dims)-1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            inp = op
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x