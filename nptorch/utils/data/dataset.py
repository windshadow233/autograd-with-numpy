import numpy as np


class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        return f'DataSet Name:{self.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class Subset(object):
    def __init__(self, dataset: Dataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __repr__(self):
        return f'SubSet of DataSet:{self.dataset.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def random_split(dataset: Dataset, split):
    if sum(split) != len(dataset):
        raise ValueError('Sum of input lengths does not equal the length of the input dataset!')
    indices = list(range(sum(split)))
    np.random.shuffle(indices)
    current_index = 0
    for s in split:
        yield Subset(dataset=dataset, indices=indices[current_index: current_index + s])
        current_index += s