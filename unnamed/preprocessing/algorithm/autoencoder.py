from unnamed.network_architecture.transformation.autoencoder import _AutoEncoderArchitecture
from unnamed.network_architecture.transformation.autoencoder import _ConvolutionalAutoEncodeArchitecture
from unnamed.network_architecture.transformation.autoencoder import _ConvolutionalAutoEncodeArchitecture2
from unnamed.classification.interface.dataset import NumpyDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch
import time

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
            outputs = self._model._encode(inputs)

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
            outputs = self._model(inputs)

        if self.gpu_available:
            outputs = outputs.cpu()

        outputs = outputs.data.numpy()

        return outputs

    # def inverse_transform(self, X):
    #     X = torch.from_numpy(X)
    #
    #     if self.gpu_available:
    #         X = X.cuda()
    #
    #     inputs = Variable(X).float()
    #
    #     with torch.no_grad():
    #         outputs = self._model._decode(inputs)
    #
    #     if self.gpu_available:
    #         outputs = outputs.cpu()
    #
    #     outputs = outputs.data.numpy()
    #
    #     return outputs

class BasicAutoEncoder(BaseAutoEncoder):
    def __init__(self, output_size=None, num_epoch=200, batch_size=512, learning_rate=0.01):

        super().__init__()

        self._learning_rate = learning_rate
        self._num_epoch = num_epoch
        self._batch_size = batch_size

        self._output_size = output_size

        self.architecture = _AutoEncoderArchitecture

    def fit(self, X):
        n_features = X.shape[1]

        if self._output_size is None:
            self._output_size = n_features // 4

        self._model = self.architecture(n_features, self._output_size)
        X = torch.from_numpy(X)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()

        inputs = Variable(X).float()
        dataset = NumpyDataset(inputs, inputs)
        train_loader = DataLoader(dataset, batch_size=self._batch_size)

        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=0.5, nesterov=True)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        for epoch in range(self._num_epoch):
            s0 = time.time()

            for batch_index, (x, y) in enumerate(train_loader):
                outputs = self._model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            e0 = time.time()

            scheduler.step()
            elapsed_time = e0 - s0

            print('epoch [{}/{}], loss:{:.4f}, elapsed_time:{:.2f}, learning_rate:{:.8f}'.format(
                epoch + 1, self._num_epoch, loss.data.item(), elapsed_time, scheduler.get_lr()[0]))

class ConvolutionalAutoEncoder(BaseAutoEncoder):
    def __init__(self, num_epoch=2000, batch_size=512, learning_rate=0.01):

        super().__init__()

        self._learning_rate = learning_rate
        self._num_epoch = num_epoch
        self._batch_size = batch_size

        self.architecture = _ConvolutionalAutoEncodeArchitecture
        # self.architecture = _ConvolutionalAutoEncodeArchitecture2

    def fit(self, X):
        n_features = X.shape[1]

        self._model = self.architecture()
        X = torch.from_numpy(X)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()

        inputs = Variable(X).float()

        dataset = NumpyDataset(inputs, inputs)
        train_loader = DataLoader(dataset, batch_size=self._batch_size)

        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=0.5, nesterov=True)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
        # scheduler = MultiStepLR(optimizer, milestones=[2000, 210, 290, 295], gamma=0.5)

        for epoch in range(self._num_epoch):
            s0 = time.time()

            for batch_index, (x, y) in enumerate(train_loader):
                outputs = self._model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            e0 = time.time()
            # scheduler.step()
            elapsed_time = e0 - s0

            # print('epoch [{}/{}], loss:{:.4f}, elapsed_time:{:.2f}, learning_rate:{:.8f}'.format(
            #     epoch + 1, self._num_epoch, loss.data.item(), elapsed_time, scheduler.get_lr()[0]))
            print('epoch [{}/{}], loss:{:.4f}, elapsed_time:{:.2f}'.format(
                epoch + 1, self._num_epoch, loss.data.item(), elapsed_time))