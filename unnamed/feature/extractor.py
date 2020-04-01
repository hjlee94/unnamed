from unnamed.feature.binary import *
from unnamed.error.exceptions import *
from unnamed.log import Logger
from multiprocessing import Pool, cpu_count
import tqdm
import glob
import os
import traceback


class FeatureExtractor:
    feature_table = {
        '1-gram': UniGramMatrix,
        '2-gram': TwoGramMatrix,
        'wem': WindowEntropyMap,
        'entropy_histogram': EntropyHistogram
    }

    @staticmethod
    def extract_parallel(data):
        input_path, feature_unit = data

        file_name = ""
        try:
            file_name = os.path.basename(input_path)
            vector = feature_unit.fit_transform(input_path)

        except Exception as e:
            e = str(e) + '\n'
            e += traceback.format_exc()
            return [False, file_name, e]

        return [True, file_name, vector]

    @staticmethod
    def feature_to_sparse(label, vector):
        non_zero_index = np.where(vector != 0)[0]

        spa_formatted_string = ['%d'%label]

        for index in sorted(non_zero_index):
            value = vector[index]
            spa_formatted_string.append('%d:%d'%(index + 1, value))

        spa_formatted_string = '%s\n'%(' '.join(spa_formatted_string))
        return spa_formatted_string

    @staticmethod
    def feature_to_csv(label, vector):
        vector = list(map(str, vector))
        vector = ','.join(vector)
        csv_formatted_string = '%d,%s\n' % (label, vector)

        return csv_formatted_string

    def __init__(self, input_path, output_path, feature, batch_size=256, n_jobs=1):
        self._input_path = input_path
        self._output_path = output_path
        self._n_jobs = n_jobs

        self._batch_size = batch_size

        if self._n_jobs > cpu_count():
            self._n_jobs = cpu_count()

        self.pool = Pool(self._n_jobs)

        self.logger = Logger.get_instance()

        self.feature_unit = FeatureExtractor.feature_table[feature]

    def _retrieve(self, dir_path):
        dir_path = os.path.join(dir_path, '*')
        file_path_list = glob.glob(dir_path)

        input_list = list()

        for file_path in file_path_list:
            if os.path.isdir(file_path):
                input_list += self._retrieve(file_path)
            else:
                input_list.append(file_path)

        return input_list

    def _gather_files(self, input_path):
        dir_names = os.listdir(input_path)
        self.logger.log_i('No. sub directory : %d' % len(dir_names))

        data_list = list()
        label_list = list()

        self._n_cls = 0

        for dir_name in sorted(dir_names):
            dir_path = os.path.join(input_path, dir_name)
            input_list = self._retrieve(dir_path)
            n_data = len(input_list)

            data_list += input_list
            label_list += [self._n_cls for _ in range(n_data)]

            self.logger.log_i('No. of data %s(%d) : %d' % (dir_name, self._n_cls, n_data))

            self._n_cls += 1

        return (data_list, label_list)

    def _extract_feature(self, data_list):
        feature_list = list()
        n_total = len(data_list)
        data_list = map(lambda x: (x, self.feature_unit()), data_list)

        # It should be imap not imap_unordered. Sequence must be preserved.
        for res in tqdm.tqdm(self.pool.imap(FeatureExtractor.extract_parallel, data_list), total=n_total):
            is_success = res[0]
            file_name = res[1]
            result = res[2]

            if is_success:
                feature_list.append(result)
            else:
                self.logger.log_e("%s : %s(%s)" % (file_name, is_success, result))

        return feature_list

    def _collect(self, feature_list, label_list):
        fd = open(self._output_path, 'a+')

        file_extension = os.path.basename(self._output_path).split('.')[-1].lower()

        if file_extension == 'spa':
            format_type = FeatureExtractor.feature_to_sparse
        elif file_extension == 'csv':
            format_type = FeatureExtractor.feature_to_csv
        else:
            raise UndefinedExtension(file_extension)

        n_data = len(feature_list)

        for i in range(n_data):
            label = label_list[i]
            vector = feature_list[i]

            formatted_string = format_type(label, vector)

            fd.write(formatted_string)

        fd.close()

    def process(self):
        self.logger.log_i('Feature Extraction')
        self.logger.log_i('batch size : %d'%(self._batch_size))

        self.logger.log_i('[Step 1] gathering files')
        data_list, label_list = self._gather_files(self._input_path)

        n_data = len(data_list)

        batch_step = n_data // self._batch_size

        if n_data % self._batch_size != 0:
            batch_step += 1

        self.logger.log_i('[Step 2] Batch process start')

        for step in range(batch_step):
            s_index = step * self._batch_size
            e_index = s_index + self._batch_size

            batch_set = data_list[s_index : e_index]
            batch_label = label_list[s_index : e_index]

            self.logger.log_i('batch_job : ( %d / %d )' % (step+1, batch_step))

            self.logger.log_i('[Step 3] Extract features')
            feature_list = self._extract_feature(batch_set)

            self.logger.log_i('[Step 3] Reduce features')
            self._collect(feature_list, batch_label)

if __name__ == '__main__':
    fe = FeatureExtractor('../../resource/testset', '../../resource/output', batch_size=3, n_jobs=3)
    fe.process()



