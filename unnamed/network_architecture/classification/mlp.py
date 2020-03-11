from torch import nn

class _DeepNeuralNetworkArchitecture(nn.Module):

    def __init__(self, n_features, n_out):
        super().__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.layer_stack = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.Dropout(p=0.1),
            nn.PReLU(),

            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.PReLU(),

            nn.Linear(512, self.n_out),
            nn.Dropout(p=0.1),
            nn.Sigmoid()
        )

        self._init_weights()

    def forward(self, x):
        x = self.layer_stack(x)

        return x

    def _init_weights(self):
        for layer in self.layer_stack:
            if layer.__class__.__name__ != 'Linear':
                continue

            layer.weight.data.normal_()
