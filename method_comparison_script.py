from unnamed.classification.algorithm.cnn import ConvolutionalNeuralNetwork
from unnamed.classification.algorithm.dnn import DeepNeuralNetwork
from unnamed.classification.interface.dataset import DatasetInterface, DataInstance
from unnamed.classification.interface.model import ModelInterface
from unnamed.preprocessing import FeatureReducer, DataSampler, DataScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np
import sys


def single_model_method(dataset):
    X_tra, X_tes, y_tra, y_tes = dataset

    dd = DataInstance(X_tra, y_tra)
    print(dd)

    model = ModelInterface(LGBMClassifier(max_depth=10, n_estimators=100, num_leaves=512))
    model.fit(X_tra, y_tra)

    model.get_score(X_tra, y_tra, metric='acc', mark='train')
    model.get_score(X_tra, y_tra, metric='err', mark='train')
    model.get_score(X_tra, y_tra, metric='tpr', mark='train')
    model.get_score(X_tra, y_tra, metric='fpr', mark='train')
    model.get_score(X_tra, y_tra, metric='precision', mark='train')
    model.get_score(X_tra, y_tra, metric='recall', mark='train')
    model.get_score(X_tra, y_tra, metric='roc_auc', mark='train')
    model.get_score(X_tra, y_tra, metric='prc', mark='train')

    model.get_score(X_tes, y_tes, metric='acc', mark='test')
    model.get_score(X_tes, y_tes, metric='err', mark='test')
    model.get_score(X_tes, y_tes, metric='tpr', mark='test')
    model.get_score(X_tes, y_tes, metric='fpr', mark='test')
    model.get_score(X_tes, y_tes, metric='precision', mark='test')
    model.get_score(X_tes, y_tes, metric='recall', mark='test')
    model.get_score(X_tes, y_tes, metric='roc_auc', mark='test')
    model.get_score(X_tes, y_tes, metric='prc', mark='test')

    model.report()

    print('\n' + str(confusion_matrix(y_tes, model.predict(X_tes))))


def multiple_model_method(dataset):
    X_tra, X_tes, y_tra, y_tes = dataset

    n_class = len(np.unique(y_tra))

    y_sheet = np.zeros((y_tes.shape[0], n_class), dtype=float)

    for class_num in range(n_class):
        X_tra_tmp = X_tra
        y_tra_tmp = np.where(y_tra == class_num, 1, 0)
        X_tes_tmp = X_tes
        y_tes_tmp = np.where(y_tes == class_num, 1, 0)

        dd = DataInstance(X_tra_tmp, y_tra_tmp)
        print(dd)

        model = ModelInterface(LGBMClassifier(max_depth=10, n_estimators=100, num_leaves=512))
        model.fit(X_tra_tmp, y_tra_tmp)

        model.get_score(X_tra_tmp, y_tra_tmp, metric='acc', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='err', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='tpr', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='fpr', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='precision', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='recall', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='roc_auc', mark='train')
        model.get_score(X_tra_tmp, y_tra_tmp, metric='prc', mark='train')

        model.get_score(X_tes_tmp, y_tes_tmp, metric='acc', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='err', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='tpr', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='fpr', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='precision', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='recall', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='roc_auc', mark='test')
        model.get_score(X_tes_tmp, y_tes_tmp, metric='prc', mark='test')

        model.report()

        print('\n' + str(confusion_matrix(y_tes_tmp, model.predict(X_tes_tmp))))

        y_prob = model.predict_proba(X_tes_tmp)
        print(y_prob)
        print(y_prob.shape)

        y_sheet[:,class_num] = y_prob[:,1]

    y_pred = np.argmax(y_sheet, axis=1)
    print('\n' + str(confusion_matrix(y_tes, y_pred)))
    print(np.mean(y_tes == y_pred))

np.random.seed(25)

dd = DatasetInterface(sys.argv[1], label_pos=0, remove_zero_vector=False)
print(dd)
X, y = dd.get_XY()

X = DataScaler('scale').fit_transform(X)
# X = X.reshape(X.shape[0], 1, 32, 32)

X_tra, X_tes, y_tra, y_tes = train_test_split(X, y, test_size=0.33, random_state=25)

dataset = (X_tra, X_tes, y_tra, y_tes)

single_model_method(dataset)
multiple_model_method(dataset)