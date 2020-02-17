from unnamed.preprocessing import DataScaler
from unnamed.log import Logger
import numpy as np
import sys
import tqdm

class DatasetInterface:
    EXT_SPARSE = 'spa'
    EXT_CSV = 'csv'

    def __init__(self, filename, label_pos=-1, preprocess_method=None):
        self.filename = filename

        self.X = None
        self.y = list()
        self.class_info = dict()

        self.label_pos = label_pos

        self.n_data = 0
        self.n_dim = 0

        self.preprocess_method = preprocess_method

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

        self._get_class_info()

        if self.preprocess_method is not None:
            self._preprocess()

    def _get_class_info(self):
        for cls in set(self.y):
            idx = np.where(self.y == cls)[0]
            self.class_info[cls] = len(idx)

    def _load_spa(self):
        dense_data = list()

        fd = open(self.filename)

        for line in tqdm.tqdm(fd, unit='B'):
            data = line.strip().split()

            cls = int(data.pop(0))
            self.y.append(cls)

            dense_data.append(data)

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

        row_idx = 0

        for data in tqdm.tqdm(dense_data, unit='B'):
            for element in data:
                element = element.split(':')
                idx = int(element[0])
                value = float(element[1])

                self.X[row_idx, idx-1] = value

            row_idx += 1

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

    def getXY(self):
        return (self.X, self.y)

    def report(self):
        self.logger.log_i('======== Data Description ========')
        self.logger.log_i('File name : %s' % (self.filename))
        self.logger.log_i('No. data : %s' % (self.n_data))
        self.logger.log_i('No. classes : %s' % (len(set(self.y))))
        self.logger.log_i('No. dim : %s' % (self.n_dim))
        self.logger.log_i('Matrix shape : %s' % (str(self.X.shape)))
        self.logger.log_i('No. data in each class : %s' % (str(self.class_info)))
        self.logger.log_i('Data preprocessed by %s' % (self.preprocess_method))

    def __str__(self):
        s = str()
        s += '======== Data Description ========\n'
        s += 'File name : %s\n'%(self.filename)
        s += 'No. data : %s\n'%(self.n_data)
        s += 'No. classes : %s\n'%(len(set(self.y)))
        s += 'No. dim : %s\n'%(self.n_dim)
        s += 'Matrix shape : %s\n'%(str(self.X.shape))
        s += 'No. data in each class : %s\n'%(str(self.class_info))
        s += 'Data preprocessed by %s\n'%(self.preprocess_method)

        return s
