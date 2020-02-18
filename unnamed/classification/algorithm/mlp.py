from unnamed.network_architecture.classification.mlp import _DeepNeuralNetworkArchitecture
from unnamed.classification.interface.dataset import NumpyDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch


class DeepNeuralNetwork:
    def __init__(self, num_epoch=10, batch_size=256, learning_rate=1e-2):
        self._learning_rate = learning_rate
        self._num_epoch = num_epoch
        self._batch_size = batch_size

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

        dataset = NumpyDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=self._batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=1e-5)

        for epoch in range(self._num_epoch):
            for batch_index, (x, y) in enumerate(train_loader):
                outputs = self._model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epoch, loss.data.item()))

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

    # def predict_proba(self, X):
    #     outputs = self._predict(X)
    #
    #     return outputs

    def save_model(self, model_path):
        torch.save(self._model.state_dict(), model_path)