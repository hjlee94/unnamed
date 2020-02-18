from unnamed.preprocessing import DataScaler
from unnamed.log import Logger
from torch.utils.data import Dataset
import numpy as np
import sys
import tqdm

class DataInstance:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        self.n_data = self.X.shape[0]
        self.n_dim = self.X.shape[1]

        self.class_info = dict()

        self.logger = Logger.get_instance()

        self._redefine_y()
        self._get_class_info()

    def _redefine_y(self):
        y = self.y
        classes = np.unique(y)

        n_class = len(classes)
        new_class = range(n_class)

        for new_cls, old_cls in zip(new_class,classes):
            idx = np.where(y == old_cls)[0]
            y[idx] = new_cls

        y = np.array(y, dtype=np.int)
        self.y = y

    def _get_class_info(self):
        for cls in set(self.y):
            idx = np.where(self.y == cls)[0]
            self.class_info[cls] = len(idx)

    def set_X(self, X):
        self.X = X

        self.n_data = self.X.shape[0]
        self.n_dim = self.X.shape[1]

    def set_Y(self, y):
        self.y = y

        self.class_info = dict()
        self._get_class_info()

    def get_XY(self):
        return (self.X, self.y)

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.y

    def get_class_info(self):
        return self.class_info

    def save_instance(self, filename):
        print('[INFO] Save Instances to %s'%filename)
        fd = open(filename, 'w')

        for index in range(self.X.shape[0]):
            fd.write('%d '% self.y[index])

            dimension_indicies = np.where(self.X[index] != 0)[0]

            feature_list = []

            for dimension_index in dimension_indicies:
                feature_list.append('%d:%f'%(dimension_index+1, self.X[index][dimension_index]))

            fd.write(' '.join(feature_list))
            fd.write('\n')
        fd.close()

    def report(self):
        self.logger.log_i('No. data : %s' % (self.n_data))
        self.logger.log_i('No. classes : %s' % (len(set(self.y))))
        self.logger.log_i('No. dim : %s' % (self.n_dim))
        self.logger.log_i('Matrix shape : %s' % (str(self.X.shape)))
        self.logger.log_i('No. data in each class : %s' % (str(self.class_info)))

    def __str__(self):
        s = str()
        s += '[INFO] No. data : %s\n' % (self.n_data)
        s += '[INFO] No. classes : %s\n' % (len(set(self.y)))
        s += '[INFO] No. dim : %s\n' % (self.n_dim)
        s += '[INFO] Matrix shape : %s\n' % (str(self.X.shape))
        s += '[INFO] No. data in each class : %s\n' % (str(self.class_info))

        return s

class DatasetInterface:
    EXT_SPARSE = 'spa'
    EXT_CSV = 'csv'

    def __init__(self, filename, label_pos=-1, preprocess_method=None, remove_zero_vector=False):
        self.filename = filename

        self.X = None
        self.y = list()
        self.class_info = dict()

        self.data_object = None

        self.label_pos = label_pos

        self.n_data = 0
        self.n_dim = 0

        self.preprocess_method = preprocess_method
        self.remove_zero_vector = remove_zero_vector

        self.logger = Logger.get_instance()

        self._load_data()

    def _preprocess(self):
        self.preprocessor = DataScaler(self.preprocess_method)
        self.X = self.preprocessor.fit_transform(self.X)

    def _load_data(self):
        self.logger.log_i('Loading %s'%(self.filename))
        filename = self.filename.lower()

        if filename.endswith(DatasetInterface.EXT_SPARSE):
            self._load_spa()

        elif filename.endswith(DatasetInterface.EXT_CSV):
            self._load_csv()

        else:
            self.logger.log_e('Unknown extension from %s'%filename)
            sys.exit(-1)

        if self.remove_zero_vector:
            self._remove_vector()

        if self.preprocess_method is not None:
            self._preprocess()

        self.data_object = DataInstance(self.X, self.y)

    def _get_class_info(self):
        for cls in set(self.y):
            idx = np.where(self.y == cls)[0]
            self.class_info[cls] = len(idx)

    def _load_spa(self):
        dense_vector = list()

        fd = open(self.filename)

        for line in tqdm.tqdm(fd, unit='B'):
            data = line.strip().split()

            cls = int(data.pop(0))
            self.y.append(cls)

            dense_vector.append(data)

            for element in data:
                element = element.split(':')
                idx = int(element[0])

                if idx > self.n_dim:
                    self.n_dim = idx

            self.n_data += 1

        fd.close()

        self.y = np.array(self.y)

        # parse dense matrix X
        self.X = np.zeros((self.n_data, self.n_dim), dtype=float)
        for row_idx, data in enumerate(tqdm.tqdm(dense_vector, unit='B')):
            for element in data:
                element = element.split(':')
                idx = int(element[0])
                value = float(element[1])

                self.X[row_idx, idx-1] = value

    def _load_csv(self):
        self.X = list()

        fd = open(self.filename)

        for line in tqdm.tqdm(fd, unit='B'):
            data = line.strip().split(',')
            cls = int(data.pop(self.label_pos))
            self.y.append(cls)

            data = list(map(float,data))
            self.X.append(data)

            self.n_data += 1

        self.n_dim = len(data)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def _remove_vector(self):
        remain_index = list()

        for idx in range(self.X.shape[0]):
            vector = self.X[idx]
            zero_count = len(np.where(vector==0)[0])

            if zero_count == self.X.shape[1]:
                continue

            remain_index.append(idx)

        self.logger.log_i('Removing zero vector...(%s vectors)'%(self.X.shape[0] - len(remain_index)))

        self.X = self.X[remain_index]
        self.y = self.y[remain_index]

    def get_XY(self):
        return self.data_object.get_XY()

    def report(self):
        self.logger.log_i('======== Data Description ========')
        self.logger.log_i('File name : %s' % (self.filename))
        self.logger.log_i('Data preprocessed by %s' % (self.preprocess_method))
        self.data_object.report()

    def __str__(self):
        s = str()
        s += '======== Data Description ========\n'
        s += 'File name : %s\n'%(self.filename)
        s += 'Data preprocessed by %s\n'%(self.preprocess_method)
        s += str(self.data_object)

        return s

class NumpyDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

        if y is None:
            self.has_target = False
        else:
            self.has_target = True

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        vector = self.X[idx]
        target = 0

        if self.has_target:
            target = self.y[idx]

        if self.transform:
            vector = self.transform(vector)

        return (vector, target)

