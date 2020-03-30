from torch import nn

class _ConvolutionalNeuralNetworkArchitecture(nn.Module):
    '''
    1, 256, 256
    6, 125, 125
    6, 41, 41
    16, 18, 18
    16, 5, 5

    1, 32, 32
    6, 30, 30
    6, 15, 15
    16, 13, 13
    16, 6, 6

    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
            nn.Conv2d(self.n_channel, 6, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(16 * 6 * 6, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Linear(64, self.n_out),
            nn.Softmax()
        )

        # self._init_weights()

    def forward(self, x):
        x = self.cnn_layer_stack(x)

        x = x.view((-1, 16 * 6 * 6))

        x = self.linear_layer_stack(x)

        return x

    def _init_weights(self):
        for layer in self.layer_stack:
            if layer.__class__.__name__ != 'Linear':
                continue

            layer.weight.data.normal_()
