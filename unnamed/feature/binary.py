from unnamed.common.decorator import *
import numpy as np

class _BinaryFeature:
    def __init__(self):
        pass

    @abstract
    def extract(self, input_data):
        pass

    def read_binary(self, input_path):
        fd = open(input_path, 'rb')

        input_data = list(fd.read())
        fd.close()

        return input_data

    def fit_transform(self, input_data):
        if isinstance(input_data, str):
            input_data = list(self.read_binary(input_data))

        output_data = self.extract(input_data)

        return output_data

class UniGramMatrix(_BinaryFeature):
    def __init__(self):
        super().__init__()

    def extract(self, input_data):
        output_data = np.zeros((256))

        input_length = len(input_data)

        for i in range(input_length):
            row_byte = input_data[i]

            row_byte = int(row_byte)

            output_data[row_byte] += 1

        return output_data

class TwoGramMatrix(_BinaryFeature):
    def __init__(self):
        super().__init__()

    def extract(self, input_data):
        output_data = np.zeros((256,256))

        input_length = len(input_data)

        for i in range(input_length - 1):
            row_byte = input_data[i]
            col_byte = input_data[i+1]

            row_byte = int(row_byte)
            col_byte = int(col_byte)

            output_data[row_byte, col_byte] += 1

        output_data = output_data.reshape(1, -1)
        output_data = output_data.ravel()

        return output_data


class WindowEntropyMap(_BinaryFeature):
    def __init__(self):
        super().__init__()

        self.step_size = 256
        self.window_size = 1024
        self.maxtrix_size = 32
        self.row_size = np.round(float(8.1) / self.maxtrix_size, 4)

    def _map_row_index(self, val):
        row_index = int(val / self.row_size)

        return row_index

    def _map_value(self, entropy_list, histogram_list):
        entropy_matrix = np.zeros((self.maxtrix_size, self.maxtrix_size), dtype=int)

        for entropyVal, byte_histogram in zip(entropy_list, histogram_list):
            row_index= self._map_row_index(entropyVal)

            entropy_matrix[row_index, :] += byte_histogram

        return entropy_matrix

    def _get_entropy(self, byte_frequency):
        entropy_value = -np.sum(np.log(byte_frequency) * byte_frequency)

        return entropy_value

    def _slide_window(self, byte_sequence):
        entropy_list = list()
        histogram_list = list()

        for i in range(0, len(byte_sequence) - self.window_size + 1, self.step_size):
            byte_window = byte_sequence[i:i + self.window_size]
            byte_window = np.array(byte_window)

            byte_histogram = np.histogram(byte_window, bins=256)[0]

            byte_frequency = (byte_histogram / float(np.sum(byte_histogram))) + 1e-10
            entropy_value = self._get_entropy(byte_frequency)

            byte_histogram = np.sum(byte_histogram.reshape(self.maxtrix_size, -1), axis=1)

            histogram_list.append(byte_histogram)
            entropy_list.append(entropy_value)

        return entropy_list, histogram_list

    def extract(self, input_data):
        (entropy_list, histogram_list) = self._slide_window(input_data)

        output_data = self._map_value(entropy_list, histogram_list)
        output_data = output_data.reshape(1,-1)
        output_data = output_data.ravel()

        return output_data

class EntropyHistogram(_BinaryFeature):
    def __init__(self):
        super().__init__()

        self.step_size = 256
        self.window_size = 1024
        self.entropy_level = 1
        self.row_size = np.round(float(0.7) / self.entropy_level, 4)

    def _map_row_index(self, entropy_sequence):
        row_indicies = entropy_sequence // self.row_size
        row_indicies = np.array(row_indicies, dtype=np.uint8)

        return row_indicies

    def _map_value(self, entropy_list):
        entropy_matrix = np.zeros((self.entropy_level, 256), dtype=float)

        for entropy_sequence in entropy_list:
            row_indicies = self._map_row_index(entropy_sequence)

            for index in range(256):
                row_index = row_indicies[index]
                entropy_matrix[row_index, index] += entropy_sequence[index]

        return entropy_matrix

    def _get_entropy(self, byte_frequency):
        indicies = np.where(byte_frequency != 0.0)[0]
        entorpy_sequence = np.zeros_like(byte_frequency)

        for index in indicies:
            entorpy_sequence[index] = -np.log(byte_frequency[index]) * byte_frequency[index]

        return entorpy_sequence

    def _slide_window(self, byte_sequence):
        entropy_list = list()

        for i in range(0, len(byte_sequence) - self.window_size + 1, self.step_size):
            byte_window = byte_sequence[i:i + self.window_size]
            byte_window = np.array(byte_window)

            byte_histogram = np.histogram(byte_window, bins=256)[0]

            byte_frequency = (byte_histogram / float(np.sum(byte_histogram)))
            entropy_sequence = self._get_entropy(byte_frequency)

            entropy_list.append(entropy_sequence)

        return entropy_list

    def extract(self, input_data):
        entropy_list = self._slide_window(input_data)
        output_data = self._map_value(entropy_list)
        output_data = output_data.reshape(1,-1)
        output_data = output_data.ravel()

        return output_data


if __name__ == '__main__':
    vector = WindowEntropyMap().fit_transform('C:\\Users\\lhj\\Documents\\GitHub\\unnamed\\resource\\data\\TAB\\0\\result_0_0.bmp')
    print(list(vector.reshape((32,-1))))
    print()
