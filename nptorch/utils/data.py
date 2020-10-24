from nptorch.tensor import *
import math


def default_collate_fn(all_data: list):
    data_list = []
    data_length = len(all_data[0])
    for i in range(data_length):
        data_list.append(array(np.r_[[data[i].data for data in all_data]]))
    return data_list


class DataSet:
    def __init__(self):
        pass

    def __repr__(self):
        return f'DataSet Name:{self.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class SubSet:
    def __init__(self, dataset: DataSet, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __repr__(self):
        return f'SubSet of DataSet:{self.dataset.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


class DataLoader:
    def __init__(self, dataset: DataSet or SubSet, batch_size=1, shuffle=True, collate_fn=default_collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        self.n_batches = math.ceil(len(dataset) / batch_size)
        self.indices = []

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if not self.indices:
            raise StopIteration
        if len(self.indices) <= self.batch_size:
            chosen_indices = self.indices
            self.indices = []
            return self.get_batch(chosen_indices)
        chosen_indices, self.indices = self.indices[:self.batch_size], self.indices[self.batch_size:]
        return self.get_batch(chosen_indices)

    def get_batch(self, indices):
        all_data = [self.dataset[index] for index in indices]
        return self.collate_fn(all_data)


def random_split(dataset: DataSet, split):
    if sum(split) != len(dataset):
        raise ValueError('Sum of input lengths does not equal the length of the input dataset!')
    indices = list(range(sum(split)))
    np.random.shuffle(indices)
    current_index = 0
    for s in split:
        yield SubSet(dataset=dataset, indices=indices[current_index: current_index + s])
        current_index += s


