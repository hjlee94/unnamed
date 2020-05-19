from unnamed.network_architecture.classification.cnn import _ConvolutionalNeuralNetworkArchitecture4
from unnamed.network_architecture.classification.cnn import _ConvolutionalNeuralNetworkArchitecture5
from unnamed.network_architecture.classification.cnn import _3C2D, _2C1D
from unnamed.classification.interface.dataset import NumpyDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch import nn
from unnamed.log import Logger
import numpy as np
import torch
import time

torch.manual_seed(25)


class ConvolutionalNeuralNetwork:
    def __init__(self, num_epoch=200, batch_size=1024, learning_rate=0.08):
        self._learning_rate = learning_rate
        self._num_epoch = num_epoch
        self._batch_size = batch_size

        self.architecture = _ConvolutionalNeuralNetworkArchitecture4

        self.gpu_available = torch.cuda.is_available()

        self._logger = Logger.get_instance()

    def fit(self, X, y, validation_set=None, parameter_path=None):
        n_cls = len(np.unique(y))

        self._model = self.architecture(1, n_cls)

        if parameter_path is not None:
            self.load_parameter(self._model, parameter_path)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()
            y = y.cuda()

        inputs = Variable(X).float()
        targets = Variable(y).long()

        dataset = NumpyDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=self._batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=0.3, nesterov=True)
        # optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        # scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
        # scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)
        # scheduler = MultiStepLR(optimizer, milestones=[20, 80, 90, 95], gamma=0.8)
        scheduler = MultiStepLR(optimizer, milestones=[200, 210, 290, 295], gamma=0.5)
        # scheduler = MultiStepLR(optimizer, milestones=[100, 150, 290, 295], gamma=0.5)

        for epoch in range(self._num_epoch):
            s0 = time.time()

            for batch_index, (x, y) in enumerate(train_loader):
                outputs = self._model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            e0 = time.time()
            elapsed_time = e0 - s0

            scheduler.step()

            if validation_set:
                X_tes = validation_set[0]
                y_tes = validation_set[1]

                y_pred = self.predict(X_tes)
                test_acc = np.mean(y_pred == y_tes)

                self._logger.log_i('epoch [{}/{}], loss:{:.4f}, test_acc:{:.3f}, elapsed_time:{:.2f}, learning_rate:{:f}'.format(
                    epoch + 1, self._num_epoch, loss.data.item(), test_acc, elapsed_time, scheduler.get_lr()[0]))
            else:
                self._logger.log_i('epoch [{}/{}], loss:{:.4f}, elapsed_time:{:.2f}, learning_rate:{:.8f}'.format(
                    epoch + 1, self._num_epoch, loss.data.item(), elapsed_time, scheduler.get_lr()[0]))

    def _predict(self, X):
        X = torch.from_numpy(X)

        if self.gpu_available:
            X = X.cuda()

        inputs = Variable(X).float()
        y_pred = None

        dataset = NumpyDataset(inputs)
        test_loader = DataLoader(dataset, batch_size=self._batch_size)

        with torch.no_grad():
            for batch_index, (x, _) in enumerate(test_loader):
                outputs = self._model(x)

                if self.gpu_available:
                    outputs = outputs.cpu()

                if y_pred is None:
                    y_pred = outputs
                else:
                    y_pred = torch.cat((y_pred, outputs), 0)

            y_pred = y_pred.data.numpy()

        return y_pred

    def predict(self, X):
        outputs = self._predict(X)
        outputs = np.argmax(outputs, axis=1)

        return outputs

    def predict_proba(self, X):
        outputs = self._predict(X)

        return outputs

    def save_parameter(self, model_path):
        torch.save(self._model.state_dict(), model_path)

    def load_parameter(self, model, parameter_path):
        model.load_state_dict(torch.load(parameter_path))
        return model

    def __str__(self):
        return str(self._model)