from torch import nn
import torch
torch.manual_seed(25)


class _2C1D(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
            )

        self.linear_layer_stack = nn.Sequential(
                nn.Linear(128 * 6 * 6, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, self.n_out),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 128 * 6 * 6))

        x = self.linear_layer_stack(x)

        return x

class _3C2D(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
            )

        self.linear_layer_stack = nn.Sequential(
                nn.Linear(256 * 2 * 2, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, self.n_out),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 256 * 2 * 2))

        x = self.linear_layer_stack(x)

        return x


class _ConvolutionalNeuralNetworkArchitecture3(nn.Module):
    '''
    1, 32, 32

    30, 30, 30
    60, 28, 28
    90, 26, 26
    90, 13, 13


    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
            )

        self.linear_layer_stack = nn.Sequential(
                nn.Linear(128 * 13 * 13, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Linear(256, self.n_out),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 128 * 13 * 13))

        x = self.linear_layer_stack(x)

        return x


class _ConvolutionalNeuralNetworkArchitecture2(nn.Module):
    '''
    1, 32, 32

    30, 30, 30
    60, 28, 28
    90, 26, 26
    90, 13, 13


    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
            )

        self.linear_layer_stack = nn.Sequential(
                nn.Linear(128 * 13 * 13, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),


                nn.Linear(256, self.n_out),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 128 * 13 * 13))

        x = self.linear_layer_stack(x)

        return x

class _ConvolutionalNeuralNetworkArchitecture(nn.Module):
    '''
    1, 32, 32

    30, 30, 30
    60, 28, 28
    90, 26, 26
    90, 13, 13


    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 30, 3),
                nn.BatchNorm2d(30),
                nn.ReLU(),

                nn.Conv2d(30, 60, 3),
                nn.BatchNorm2d(60),
                nn.ReLU(),

                nn.Conv2d(60, 90, 3),
                nn.BatchNorm2d(90),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
            )

        self.linear_layer_stack = nn.Sequential(
                nn.Linear(90 * 13 * 13, 120),
                nn.BatchNorm1d(120),
                nn.ReLU(),

                nn.Linear(120, 60),
                nn.BatchNorm1d(60),
                nn.ReLU(),

                nn.Linear(60, 30),
                nn.BatchNorm1d(30),
                nn.ReLU(),

                nn.Linear(30, self.n_out),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.cnn_layer_stack(x)
        x = x.view((-1, 90 * 13 * 13))

        x = self.linear_layer_stack(x)

        return x

class _BiConvolutionalNeuralNetworkArchitecture(nn.Module):
    '''
    1, 32, 32

    top                 bottom
    1, 16, 32           1, 16, 32

    10, 14, 28          10, 14, 28
    30, 12, 24          30, 12, 24
    30, 6, 12           30, 6, 12

    60, 6, 10           60, 6, 10
    120, 6, 8
    120, 3, 4


    '''
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.n_channel = n_channel
        self.n_out = n_out

        self.cnn_top_layer_stack = nn.Sequential(
                nn.Conv2d(self.n_channel, 30, (3,5)),
                nn.BatchNorm2d(30),
                nn.ReLU(),

                nn.Conv2d(30, 60, (3,5)),
                nn.BatchNorm2d(60),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(60, 60, (1, 3)),
                nn.BatchNorm2d(60),
                nn.ReLU(),

                nn.Conv2d(60, 60, (1, 3)),
                nn.BatchNorm2d(60),
                nn.ReLU(),

                nn.MaxPool2d((1,2), 2),
            )

        self.cnn_bottom_layer_stack = nn.Sequential(
            nn.Conv2d(self.n_channel, 30, (3, 5)),
            nn.BatchNorm2d(30),
            nn.ReLU(),

            nn.Conv2d(30, 60, (3, 5)),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(60, 60, (1, 3)),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.Conv2d(60, 60, (1, 3)),
            nn.BatchNorm2d(60),
            nn.ReLU(),

            nn.MaxPool2d((1, 2), 2),
        )

        self.bilinear_layer = nn.Bilinear(4, 4, 4)

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(60 * 3 * 4, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, self.n_out),
            nn.Softmax(dim=1)
        )

        # self._init_weights()

    def forward(self, x):
        n_dim = x.dim()

        if n_dim > 3:
            axis = 2
        else:
            axis = 1

        x1 = x.narrow(axis, 0, 16)
        x2 = x.narrow(axis, 16, 16)

        x1 = self.cnn_top_layer_stack(x1)
        x2 = self.cnn_bottom_layer_stack(x2)

        x = self.bilinear_layer(x1,x2)

        x = x.view((-1, 60 * 3 * 4))

        x = self.linear_layer_stack(x)

        return x
