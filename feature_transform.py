from unnamed.classification.interface.dataset import DatasetInterface, DataInstance
from unnamed.preprocessing.preprocessor import DataScaler, FeatureReducer
from PIL import Image
import numpy as np


def complement(x):
    x[x <= -1] = -1
    x[x >= +1] = +1
    x += 1
    x = x / 2.0
    x = x * 255
    x[x < 0] = 0
    x[x > 255] = 255

    x = np.array(x, dtype=np.uint8)

    return x

def to_image(x, name):
    x = complement(x)

    if len(x.shape) > 2:
        x = x[0]

    x = Image.fromarray(x)
    # x.show()
    x.save('%s.bmp'%name, "BMP")

dd = DatasetInterface('D:\\Saint_Security\\AML\\wem.csv')
print(dd)

X, y = dd.get_XY()

scaler = DataScaler('scale')
X = scaler.fit_transform(X) # -1 ~ +1

print(X.shape)

X = X.reshape((X.shape[0], 1, 32,32))


transformer = FeatureReducer('conv')

transformer.fit(X)

print(X.shape)

X_transformed = transformer.transform(X)

print(X_transformed.shape)


new_dd = DataInstance(X_transformed.reshape((X.shape[0], -1)), y)
new_dd.save_instance('./resource/wem_code_layer.spa')

X_inverse_transformed = transformer.inverse_transform(X_transformed)

print(X_inverse_transformed.shape)
print(X_inverse_transformed[0])


new_dd = DataInstance(X_inverse_transformed.reshape((X.shape[0], -1)), y)
new_dd.save_instance('./resource/wem_inverse_transformed.csv')

for class_id in range(len(np.unique(y))):
    data_index = np.where(y == class_id)[0]
    for i in data_index[:100]:
        to_image(X[i], 'result/wem/%d/%d-original'%(class_id,i))
        to_image(X_transformed[i], 'result/wem/%d/%d-transformed'%(class_id,i))
        to_image(X_inverse_transformed[i], 'result/wem/%d/%d-inverse-transformed'%(class_id,i))