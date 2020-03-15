from unnamed.preprocessing.algorithm.autoencoder import BasicAutoEncoder, ConvolutionalAutoEncoder
from unnamed.log import Logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pickle

class DataSampler:
    def __init__(self, method, replace_allow=False, random_state=20):
        self.X = None
        self.y = None
        self.unique_cls = None
        self.method = method
        self.replace_allow = replace_allow
        self.logger = Logger.get_instance()

        self.random_state = 20

        np.random.seed(self.random_state)

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.unique_cls = np.unique(self.y)

    def fit_sample(self, X, y, n):
        self.fit(X, y)

        resampled_set = None

        if self.method == 'random':
            resampled_set = self.random_sampling(n)
        else:
            self.logger.log_e('wrong method')

        return resampled_set

    def random_sampling(self, n_target_samples):
        sample_candidate = list()

        for cls in self.unique_cls:
            candidate = np.where(self.y == cls)[0]
            n_samples = n_target_samples

            replace = False

            if self.replace_allow:
                if len(candidate) < n_target_samples:
                    replace = True
            else:
                if len(candidate) < n_target_samples:
                    n_samples = len(candidate)

            self.logger.log_i('%d-class sampled %d from %d with replace:%s'%(cls, n_samples, len(candidate), str(replace)))

            np.random.seed(1)
            idx = np.random.choice(candidate, n_samples, replace=replace)
            sample_candidate += list(idx)

        X_sampled = self.X[sample_candidate]
        y_sampled = self.y[sample_candidate]

        return (X_sampled, y_sampled)

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

    def __init__(self, preprocess_method, **parameters):
        self.encoder = FeatureReducer.preprocessor_table[preprocess_method.lower()](**parameters)
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