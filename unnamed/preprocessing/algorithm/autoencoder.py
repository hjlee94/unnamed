from unnamed.network_architecture.transformation.autoencoder import _AutoEncoderArchitecture
from unnamed.network_architecture.transformation.autoencoder import _ConvolutionalAutoEncodeArchitecture
from torch import nn
from torch.autograd import Variable
import torch

class BaseAutoEncoder:
    def __init__(self):
        self._model = None
        self.gpu_available = torch.cuda.is_available()

    def fit(self, X):
        pass

    def transform(self, X):
        X = torch.from_numpy(X)

        if self.gpu_available:
            X = X.cuda()

        inputs = Variable(X).float()

        with torch.no_grad():
            outputs = self._model.encoder(inputs)

        if self.gpu_available:
            outputs = outputs.cpu()

        outputs = outputs.data.numpy()

        return outputs

    def inverse_transform(self, X):
        X = torch.from_numpy(X)

        if self.gpu_available:
            X = X.cuda()

        inputs = Variable(X).float()

        with torch.no_grad():
            outputs = self._model.decoder(inputs)

        if self.gpu_available:
            outputs = outputs.cpu()

        outputs = outputs.data.numpy()

        return outputs

class BasicAutoEncoder(BaseAutoEncoder):
    def __init__(self, output_size=None, num_epoch=200, learning_rate=1e-2):

        super().__init__()

        self._learning_rate = learning_rate
        self._num_epoch = num_epoch

        self._output_size = output_size

        self.architecture = _AutoEncoderArchitecture

    def fit(self, X):
        n_features = X.shape[1]

        if self._output_size is None:
            self._output_size = n_features // 2

        self._model = self.architecture(n_features, self._output_size)
        X = torch.from_numpy(X)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()

        inputs = Variable(X).float()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=1e-15)

        for epoch in range(self._num_epoch):
            outputs = self._model.forward(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epoch, loss.data.item()))

class ConvolutionalAutoEncoder(BaseAutoEncoder):
    def __init__(self, num_epoch=200, learning_rate=1e-2):

        super().__init__()

        self._learning_rate = learning_rate
        self._num_epoch = num_epoch

        self.architecture = _ConvolutionalAutoEncodeArchitecture

    def fit(self, X):
        n_features = X.shape[1]

        self._model = self.architecture()
        X = torch.from_numpy(X)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()

        inputs = Variable(X).float()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=1e-15)

        for epoch in range(self._num_epoch):
            outputs = self._model.forward(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epoch, loss.data.item()))
