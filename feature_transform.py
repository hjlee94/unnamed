from unnamed.classification.interface.dataset import DatasetInterface, DataInstance
from unnamed.preprocessing.preprocessor import DataScaler, FeatureReducer
from PIL import Image
import numpy as np
import sys, os


input_path = sys.argv[1]

file_name = os.path.basename(input_path).split('.')[0]
file_path = os.path.abspath(input_path).split(os.path.sep)[:-1]
file_path = os.path.sep.join(file_path)

dd = DatasetInterface(sys.argv[1], label_pos=0, remove_zero_vector=False)
print(dd)
X, y = dd.get_XY()

scaler = DataScaler('scale')
X = scaler.fit_transform(X) # -1 ~ +1

print(X.shape)

# X = X.reshape((X.shape[0], 1, 32,32))
# transformer = FeatureReducer('conv',num_epoch=200, learning_rate = 0.01)
transformer = FeatureReducer('basic',num_epoch=200, output_size=128, learning_rate = 0.01)

transformer.fit(X)

print(X.shape)

X_transformed = transformer.transform(X)
print(X_transformed.shape)
new_dd = DataInstance(X_transformed.reshape((X.shape[0], -1)), y)
new_dd.save_instance(os.path.join(file_path, file_name+'_encoded.csv'))


X_transformed = transformer.inverse_transform(X)
print(X_transformed.shape)
X_transformed = scaler.inverse_transform(X_transformed.reshape((X.shape[0], -1)))
new_dd = DataInstance(X_transformed.reshape((X.shape[0], -1)), y)
new_dd.save_instance(os.path.join(file_path, file_name+'_decoded.csv'))