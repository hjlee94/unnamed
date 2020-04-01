from torch import nn

class _ConvolutionalNeuralNetworkArchitecture(nn.Module):
    '''
    1, 256, 256
    6, 125, 125
    6, 41, 41
    16, 18, 18
    16, 5, 5

    1, 32, 32
    30, 30, 30
    60, 28, 28
    120, 26, 26
    120, 13, 13


    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
            nn.Conv2d(self.n_channel, 30, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(30),

            nn.Conv2d(30, 60, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(60),

            nn.Conv2d(60, 120, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(120),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(120),

            # nn.Conv2d(6, 16, 3),
            # nn.MaxPool2d(2, 2),
            # nn.ReLU(True),
            # nn.BatchNorm2d(16),
        )

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(120 * 13 * 13, 120),
            nn.ReLU(True),
            nn.BatchNorm1d(120),
            nn.Dropout(0.1),

            nn.Linear(120, 60),
            nn.ReLU(True),
            nn.BatchNorm1d(60),
            nn.Dropout(0.1),

            nn.Linear(60, self.n_out),
            nn.Softmax()
        )

        # self._init_weights()

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 120 * 13 * 13))

        x = self.linear_layer_stack(x)

        return x

    def _init_weights(self):
        for layer in self.layer_stack:
            if layer.__class__.__name__ != 'Linear':
                continue

            layer.weight.data.normal_()
