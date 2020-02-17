from torch import nn
from torch.autograd import Variable
import torch


class _DeepNeuralNetworkArchitecture(nn.Module):
    def __init__(self, n_features, n_out):
        super().__init__()
        self.n_features = n_features
        self.n_out = n_out

        self.layer_stack = nn.Sequential(
            nn.Linear(self.n_features, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, self.n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer_stack(x)

        return x

