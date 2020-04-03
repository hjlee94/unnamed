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
            nn.Conv2d(self.n_channel, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),

            nn.Conv2d(10, 30, 3),
            nn.BatchNorm2d(30),
            nn.ReLU(True),

            nn.Conv2d(30, 60, 3),
            nn.BatchNorm2d(60),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2),
        )

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(60 * 13 * 13, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(30, self.n_out),
            nn.Softmax(dim=1)
        )

        # self._init_weights()

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 60 * 13 * 13))

        x = self.linear_layer_stack(x)

        return x

    def _init_weights(self):
        for layer in self.layer_stack:
            if layer.__class__.__name__ != 'Linear':
                continue

            layer.weight.data.normal_()
