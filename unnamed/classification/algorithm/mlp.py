from unnamed.network_architecture.classification.mlp import _DeepNeuralNetworkArchitecture
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch


class DeepNeuralNetwork:
    def __init__(self, num_epoch=200, learning_rate=1e-2):
        self._learning_rate = learning_rate
        self._num_epoch = num_epoch

        self.architecture = _DeepNeuralNetworkArchitecture

        self.gpu_available = torch.cuda.is_available()

    def fit(self, X, y):
        n_features = X.shape[1]
        n_cls = len(np.unique(y))

        self._model = self.architecture(n_features, n_cls)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        if self.gpu_available:
            self._model = self._model.cuda()
            X = X.cuda()
            y = y.cuda()

        inputs = Variable(X).float()
        targets = Variable(y).long()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=1e-15)

        for epoch in range(self._num_epoch):
            outputs = self._model.forward(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epoch, loss.data.item()))

    def _predict(self, X):
        X = torch.from_numpy(X)

        if self.gpu_available:
            X = X.cuda()

        inputs = Variable(X).float()

        with torch.no_grad():
            outputs = self._model.forward(inputs)

        if self.gpu_available:
            outputs = outputs.cpu()

        outputs = outputs.data.numpy()

        return outputs

    def predict(self, X):
        outputs = self._predict(X)
        outputs = np.argmax(outputs, axis=1)

        return outputs


    def predict_proba(self, X):
        outputs = self._predict(X)

        return outputs
