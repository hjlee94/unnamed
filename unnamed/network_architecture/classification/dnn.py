from torch import nn
import torch
import numpy as np

class _DeepNeuralNetworkArchitecture(nn.Module):

    def __init__(self, n_features, n_out):
        super().__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.layer_stack = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            #
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(64, self.n_out),
            # nn.Dropout(p=0.2),
            nn.Softmax(dim=1)
        )

        # self._init_weights()

    def forward(self, x):
        x = self.layer_stack(x)

        return x

    def _init_weights(self):
        for layer in self.layer_stack:
            if layer.__class__.__name__ in ['Linear']:
                base_value = np.sqrt(2 / layer.weight.shape[0])

                layer.weight.data.normal_(mean=0, std=base_value)

                layer.bias.data = torch.zeros(layer.bias.shape)
