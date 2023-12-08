import numpy as np
from tensorflow.python.framework import dtypes


class DataSet(object):
    def __init__(self, data, labels, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype

        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]



def load_carhacking_datasets(dataset_name='fuzzy', data_dir='../data/', dtype=dtypes.float32):
    
    data = np.load(data_dir + '29x29_' + dataset_name + '_data.npy')
    labels = np.load(data_dir + '29x29_' + dataset_name + '_label.npy')

    dataset = DataSet(data, labels, dtype=dtype)

    return dataset

def load_test_datasets(dataset_name='fuzzy', data_dir='../data/', dtype=dtypes.float32):
    # '../dataset/new_data/29x29_carhackingattack_data.npy' || 'carhackingattack_label.npy'
    
    test_data = np.load(data_dir + '29x29_' + dataset_name + '_data.npy')
    test_labels = np.load(data_dir + '29x29_' + dataset_name + '_label.npy')

    test = DataSet(test_data, test_labels, dtype=dtype)

    return test
