from torch import nn
from torch.autograd import Variable
import torch


class _AutoEncoderArchitecture(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, self.n_hidden)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, self.n_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class _ConvolutionalAutoEncodeArchitecture(nn.Module):
    '''
    input image size is 3 x 128 x 128
    hidden image size is 16 x 30 x 30
    output image size is

    '''
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=0),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5

            nn.Conv2d(16, 3, 3, stride=1, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 5, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(16, 6, 3, stride=2, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(6, 3, 2, stride=1, padding=0),  # b, 3, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


