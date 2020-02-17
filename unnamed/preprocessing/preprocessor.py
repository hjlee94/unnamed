from sklearn.preprocessing import StandardScaler, MinMaxScaler
from unnamed.preprocessing.algorithm.autoencoder import BasicAutoEncoder, ConvolutionalAutoEncoder
import numpy as np
import pickle

class DataScaler:
    preprocessor_table = dict()
    preprocessor_table['scale'] = StandardScaler
    preprocessor_table['minmax'] = MinMaxScaler

    def __init__(self, preprocess_method):
        self.encoder = DataScaler.preprocessor_table[preprocess_method.lower()]()
        self.X = None

    def get_encoder_name(self):
        if self.encoder is None:
            return 'None'

        return self.encoder.__class__.__name__

    def fit(self, X):
        self.encoder.fit(X)

    def transform(self, X):
        X_transformed = self.encoder.transform(X)
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.transform(X)

        return X_transformed

    def inverse_transform(self, X):
        X_inverse_transformed = self.encoder.inverse_transform(X)
        return X_inverse_transformed

    def save_encoder(self, output_path):
        fd = open(output_path, 'wb')
        pickle.dump(self.encoder, fd)
        fd.close()

    def load_encoder(self, input_path):
        fd = open(input_path, 'rb')
        self.encoder = pickle.load(fd)
        fd.close()


class FeatureReducer:
    preprocessor_table = dict()
    preprocessor_table['basic'] = BasicAutoEncoder
    preprocessor_table['conv'] = ConvolutionalAutoEncoder

    def __init__(self, preprocess_method):
        self.encoder = FeatureReducer.preprocessor_table[preprocess_method.lower()]()
        self.X = None

    def get_encoder_name(self):
        if self.encoder is None:
            return 'None'

        return self.encoder.__class__.__name__

    def fit(self, X):
        self.encoder.fit(X)

    def transform(self, X):
        X_transformed = self.encoder.transform(X)
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.transform(X)

        return X_transformed

    def inverse_transform(self, X):
        X_inverse_transformed = self.encoder.inverse_transform(X)
        return X_inverse_transformed

    def save_encoder(self, output_path):
        fd = open(output_path, 'wb')
        pickle.dump(self.encoder, fd)
        fd.close()

    def load_encoder(self, input_path):
        fd = open(input_path, 'rb')
        self.encoder = pickle.load(fd)
        fd.close()