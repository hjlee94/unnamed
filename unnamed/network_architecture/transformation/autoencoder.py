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
        x = self.encoder(x)

        return x

    def _decode(self, x):
        x = self.decoder(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

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
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(1, 30, 3)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(30, 60, 3)

        self.linear1 = nn.Linear(60 * 13 * 13, 1024)

        self.unlinear1 = nn.Linear(1024, 60 * 13 * 13)

        self.deconv1 = nn.ConvTranspose2d(60, 30, 3, stride=1, padding=0)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(30, 1, 3, stride=1, padding=0)

        self.index_list = list()

    def _encode(self, x):
        x = self.conv1(x)
        x = self.tanh(x)

        x, pool1_index = self.pool1(x)

        x = self.conv2(x)
        x = self.tanh(x)

        self.index_list.append(pool1_index)

        x = x.view((-1, 60 * 13 * 13))

        x = self.linear1(x)


        return x

    def _decode(self, x):
        x = self.unlinear1(x)
        x = self.tanh(x)

        x = x.view((-1, 60, 13, 13))

        x = self.deconv1(x)
        x = self.tanh(x)

        x = self.unpool1(x, self.index_list.pop())

        x = self.deconv2(x)
        x = self.tanh(x)

        return x


    def forward(self, x):
        x = self._encode(x)
        x = self._decode(x)

        return x

class _ConvolutionalAutoEncodeArchitecture2(nn.Module):
    '''
    input image size is 3 x 128 x 128
    hidden image size is 16 x 30 x 30
    output image size is

    1 x 32 x 32

    30 x 30 x 30
    30 x 15 x 15
    60 x 12 x 12

    90 x 10 x 10
    90 x 5 x 5
    120 x 3 x 3

    '''
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 30, 3)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(30, 60, 4)

        self.conv3 = nn.Conv2d(60, 90, 3)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.conv4 = nn.Conv2d(90, 120, 3)

        self.linear1 = nn.Linear(120 * 3 * 3, 512)

        self.unlinear1 = nn.Linear(512, 120 * 3 * 3)

        self.deconv1 = nn.ConvTranspose2d(120, 90, 3, stride=1, padding=0)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(90, 60, 3, stride=1, padding=0)

        self.deconv3 = nn.ConvTranspose2d(60, 30, 4, stride=1, padding=0)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv4 = nn.ConvTranspose2d(30, 1, 3, stride=1, padding=0)


    def _encode(self, x):
        index_list = list()

        x = self.conv1(x)
        x = self.tanh(x)

        x, pool1_index = self.pool1(x)
        index_list.append(pool1_index)

        x = self.conv2(x)
        x = self.tanh(x)

        x = self.conv3(x)
        x = self.tanh(x)

        x, pool2_index = self.pool2(x)
        index_list.append(pool2_index)

        x = self.conv4(x)
        x = self.tanh(x)

        x = x.view((-1, 120 * 3 * 3))

        x = self.linear1(x)

        return x, index_list

    def _decode(self, x, index_list=[]):
        x = self.unlinear1(x)
        x = self.tanh(x)

        x = x.view((-1, 120, 3, 3))

        x = self.deconv1(x)
        x = self.tanh(x)

        x = self.unpool1(x, index_list.pop())

        x = self.deconv2(x)
        x = self.tanh(x)

        x = self.deconv3(x)
        x = self.tanh(x)

        x = self.unpool2(x, index_list.pop())

        x = self.deconv4(x)
        x = self.tanh(x)

        return x


    def forward(self, x):
        x, index_list = self._encode(x)

        x = self._decode(x, index_list)

        return x

