import numpy as np
import itertools
import pickle
from NeuralUtils import slice_index

def flatten(iter, depth=1):
    for i in range(depth):
        iter = itertools.chain.from_iterable(iter)
    return list(iter)

def mem_shift(data, mem_size):
    data_len = len(data) - (mem_size - 1)
    result = np.zeros((data_len, mem_size,) + data.shape[1:])
    for i in range(data_len):
        result[i] = data[i:i+mem_size]
    return result

def probs_temperature(data, temperature):
    data = np.exp(np.log(data) / temperature)
    data /= np.sum(data)
    return data

def shuffle_seq(x, y):
    order = np.random.permutation(len(x))
    return x[order], y[order]

def iter_batches(data, batch_size):
    data_len = len(data)
    batch_count = int(np.ceil(data_len / batch_size))
    for i in range(batch_count):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, data_len)
        yield batch_start, batch_end

def random_slice(data_len, seg_len):
    start = np.random.randint(0, data_len - (seg_len - 1))
    return slice(start, start + seg_len)

def split_data(data, ratio=0.2):
    pos = int(len(data) * (1 - ratio))
    return data[:pos], data[pos:]

class Dataset:
    def __init__(self):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, index):
        pass
    def iter_batches(self, batch_size):
        for batch_start, batch_end in iter_batches(self, batch_size):
            yield self[batch_start:batch_end]

class DatasetMemsize(Dataset):
    def __init__(self, dataset, mem_size):
        self.dataset = dataset
        self.mem_size = mem_size
    def __len__(self):
        return len(self.dataset) - self.mem_size
    def __getitem__(self, index):
        index = slice_index(index, len(self))
        index = slice(index.start, index.stop + self.mem_size)
        data = self.dataset[index]
        x, y = mem_shift(data, self.mem_size)[:-1], data[self.mem_size:]
        return x, y

class DatasetBatches:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle = False
    def __len__(self):
        return int(np.ceil(len(self.dataset) / NN.batch_size))
    def prepare(self):
        data_len = len(self.dataset)
        if self.shuffle:
            self.order = np.random.permutation(data_len)
    def __getitem__(self, batch):
        data_len = len(self.dataset)
        batch_start = batch * NN.batch_size
        batch_end = min(batch_start + NN.batch_size, data_len)
        batch_len = batch_end - batch_start
        if not self.shuffle:
            return self.dataset[batch_start:batch_end]
        else:
            batch_order = self.order[batch_start:batch_end]
            x = np.zeros((batch_len, NN.mem_size, NN.classes))
            y = np.zeros((batch_len, NN.classes))
            for i, index in enumerate(batch_order):
                x_vec, y_vec = self.dataset[index:index+1]
                x[i] = x_vec[0]
                y[i] = y_vec[0]
            return x, y
class DatasetCache(DatasetBatches):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.prefix = "nn"
    def prepare(self):
        dataset_file = "cache/" + self.prefix + "_data_"
        super().prepare()
        batch_count = int(np.ceil(len(self.dataset) / NN.batch_size))
        for batch in range(batch_count):
            x, y = super()[batch]
            pickle.dump((x, y), open(dataset_file + str(batch) + ".pickle", "wb"))
    def __getitem__(self, batch):
        dataset_file = "cache/" + self.prefix + "_data_"
        return pickle.load(open(dataset_file + str(batch) + ".pickle", "rb"))
    # to h5
    # rm NN., prefix
