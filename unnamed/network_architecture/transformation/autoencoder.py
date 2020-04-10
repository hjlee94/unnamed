from torch import nn
from torch.autograd import Variable
import torch


class _AutoEncoderArchitecture(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.n_hidden)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, self.n_features),
            nn.Sigmoid()
        )

    def _encode(self, x):
        x = self.encode(x)

        return x

    def _decode(self, x):
        x = self.decoder(x)

        return x

    def forward(self, x):
        x = self._encode(x)
        x = self._decode(x)

        return x

class _ConvolutionalAutoEncodeArchitecture(nn.Module):
    '''
    input image size is 3 x 128 x 128
    hidden image size is 16 x 30 x 30
    output image size is

    1 x 32 x 32

    10 x 30 x 30
    10 x 15 x 15

    30 x 13 x 13

    -----------

    30 x 13 x 13

    10 x 15 x 15
    10 x 30 x 30

    1 x 32 x 32

    '''
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 30, 3)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(30, 60, 3)

        self.deconv1 = nn.ConvTranspose2d(60, 30, 3, stride=1, padding=0)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(30, 1, 3, stride=1, padding=0)

    def _encode(self, x):
        index_list = list()

        x = self.conv1(x)
        x = self.sigmoid(x)

        x, pool1_index = self.pool1(x)

        x = self.conv2(x)
        x = self.sigmoid(x)

        index_list.append(pool1_index)

        return x, index_list

    def _decode(self, x, index_list=[]):
        x = self.deconv1(x)
        x = self.sigmoid(x)

        x = self.unpool1(x, index_list.pop())

        x = self.deconv2(x)
        x = self.sigmoid(x)

        return x


    def forward(self, x):
        x, index_list = self._encode(x)
        x = self._decode(x, index_list)

        return x


