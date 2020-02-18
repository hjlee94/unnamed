from unnamed.classification.interface import ModelInterface, DatasetInterface
from unnamed.log import Logger
from unnamed.classification.algorithm.mlp import DeepNeuralNetwork
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import time


class Classifier:
    algorithm_table = dict()
    algorithm_table['random_forest'] = RandomForestClassifier
    algorithm_table['mlp'] = DeepNeuralNetwork
    algorithm_table['svc'] = SVC

    def __init__(self, alg_name=None, parameters={}, label_pos=0, preprocess_method=None, remove_zero_vector=False):
        self.alg_name = alg_name

        # model
        self._model = None
        self._parameters = parameters

        self._dataset = None
        self._label_pos = label_pos
        self._preprocess_method = preprocess_method
        self._remove_zero_vector = remove_zero_vector

        # train set
        self._X_tra = None
        self._y_tra = None

        # test set
        self._X_tes = None
        self._y_tes = None
        
        self.logger = Logger.get_instance()

    def _load_dataset(self, data_path):
        self._dataset = DatasetInterface(data_path, label_pos=self._label_pos,
                                         preprocess_method=self._preprocess_method,
                                         remove_zero_vector=self._remove_zero_vector)
        self._dataset.report()

    def _load_pickle_model(self, model_path):
        self._model = ModelInterface(model=None)
        self._model.load_model(model_path)

    def _save_pickle_model(self, model_path):
        self._model.save_model(model_path)

    def _init_model(self):
        classifier = Classifier.algorithm_table[self.alg_name]
        self._model = ModelInterface(model=classifier(**self._parameters))

    def train(self, data_path, model_path):
        self._init_model()
        self._load_dataset(data_path)
        (self._X_tra, self._y_tra) = self._dataset.get_XY()
        
        self._model.fit(self._X_tra, self._y_tra)

        self._model.get_score(self._X_tra, self._y_tra, metric='acc', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='err', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='tpr', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='fpr', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='precision', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='recall', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='roc_auc', mark='train')
        self._model.get_score(self._X_tra, self._y_tra, metric='prc', mark='train')

        self._model.report()

        if model_path is not None:
            self._save_pickle_model(model_path)

    def evaluate(self, data_path, n_folds):
        self._init_model()
        self._load_dataset(data_path)
        X,y = self._dataset.get_XY()

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=25)
        for idx_tra, idx_tes in kf.split(X, y):
            X_tra = X[idx_tra]
            y_tra = y[idx_tra]
            X_tes = X[idx_tes]
            y_tes = y[idx_tes]

            self._model.fit(X_tra, y_tra)

            self._model.get_score(X_tra, y_tra, metric='acc', mark='train')
            self._model.get_score(X_tra, y_tra, metric='err', mark='train')
            self._model.get_score(X_tra, y_tra, metric='tpr', mark='train')
            self._model.get_score(X_tra, y_tra, metric='fpr', mark='train')
            self._model.get_score(X_tra, y_tra, metric='precision', mark='train')
            self._model.get_score(X_tra, y_tra, metric='recall', mark='train')
            self._model.get_score(X_tra, y_tra, metric='roc_auc', mark='train')
            self._model.get_score(X_tra, y_tra, metric='prc', mark='train')

            self._model.get_score(X_tes, y_tes, metric='acc', mark='test')
            self._model.get_score(X_tes, y_tes, metric='err', mark='test')
            self._model.get_score(X_tes, y_tes, metric='tpr', mark='test')
            self._model.get_score(X_tes, y_tes, metric='fpr', mark='test')
            self._model.get_score(X_tes, y_tes, metric='precision', mark='test')
            self._model.get_score(X_tes, y_tes, metric='recall', mark='test')
            self._model.get_score(X_tes, y_tes, metric='roc_auc', mark='test')
            self._model.get_score(X_tes, y_tes, metric='prc', mark='test')

        self._model.report()
        self.logger.log_i('\n'+str(confusion_matrix(y_tes, self._model.predict(X_tes))))

    def test(self, model_path, data_path):
        self._load_pickle_model(model_path)
        self._load_dataset(data_path)
        (self._X_tes, self._y_tes) = self._dataset.get_XY()

        self._model.classes = set(self._y_tes)

        self._model.get_score(self._X_tes, self._y_tes, metric='acc', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='err', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='tpr', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='fpr', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='precision', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='recall', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='roc_auc', mark='test')
        self._model.get_score(self._X_tes, self._y_tes, metric='prc', mark='test')

        self._model.report()

    def predict(self, model_path, data_path, output_path):
        pass

