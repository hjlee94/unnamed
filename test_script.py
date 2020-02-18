from unnamed.classification.algorithm.mlp import DeepNeuralNetwork
from unnamed.classification.interface import DatasetInterface, ModelInterface, DataInstance
from unnamed.preprocessing import FeatureReducer, DataSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

dd = DatasetInterface('./resource/iris.csv', label_pos=-1, preprocess_method='scale')
print(dd)
X, y = dd.get_XY()

X_sampled, y_sampled = DataSampler(method='random').fit_sample(X,y, n=20000)

obj = DataInstance(X_sampled, y_sampled)
print(obj)
X, y = obj.get_XY()

# autoencoder = FeatureReducer('basic')
# autoencoder.fit(X)
# print(X)
# X_transformed = autoencoder.transform(X)
# print(X_transformed)
# X_inverse_transformed = autoencoder.inverse_transform(X_transformed)
# print(X_inverse_transformed)

model = ModelInterface(DeepNeuralNetwork())
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=25)
for idx_tra, idx_tes in kf.split(X, y):

    X_tra = X[idx_tra]
    y_tra = y[idx_tra]
    X_tes = X[idx_tes]
    y_tes = y[idx_tes]

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
print('\n'+str(confusion_matrix(y_tes, model.predict(X_tes))))