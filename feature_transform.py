from unnamed.classification.interface.dataset import DatasetInterface, DataInstance
from unnamed.preprocessing.algorithm.autoencoder import BasicAutoEncoder

dd = DatasetInterface('./resource/wem.spa', preprocess_method='scale')
print(dd)
X, y = dd.get_XY()

autoencoder = BasicAutoEncoder(256, num_epoch=300, batch_size=1000, learning_rate=1e-3)
# X = X.reshape((X.shape[0], 1,32,32))
autoencoder.fit(X)

print(X.shape)
print(X[0])

X_transformed = autoencoder.transform(X)

print(X_transformed.shape)

# new_dd = DataInstance(X_transformed.reshape((X_transformed.shape[0], 108)), y)
new_dd = DataInstance(X_transformed, y)
new_dd.save_instance('./resource/wem_transformed.spa')

X_inverse_transformed = autoencoder.inverse_transform(X_transformed)

print(X_inverse_transformed.shape)
print(X_inverse_transformed[0])
