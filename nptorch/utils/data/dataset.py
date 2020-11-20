import numpy as np


class Dataset(object):
    def __repr__(self):
        return f'DataSet Name:{self.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: list):
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

    def __repr__(self):
        return f'SubSet of DataSet:{self.dataset.__class__.__name__}\nLength:{self.__len__()}'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]
    
    
class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = datasets
        self.cumulative_length = self.cumsum(datasets)

    def __repr__(self):
        return f'ConcatDataset of DataSets:\n{self.datasets}\nLength:{self.__len__()}'

    @staticmethod
    def cumsum(datasets):
        r, s = [], 0
        for dataset in datasets:
            length = len(dataset)
            s += length
            r.append(s)
        return r, s

    def get_dataset_data_idx(self, item):
        r = self.cumulative_length[0]
        for i, length in enumerate(r):
            if length <= item:
                item -= length
            else:
                return i, item

    def __len__(self):
        return self.cumulative_length[-1]

    def __getitem__(self, item):
        i, item = self.get_dataset_data_idx(item)
        return self.datasets[i][item]


def random_split(dataset: Dataset, split):
    if sum(split) != len(dataset):
        raise ValueError('Sum of input lengths does not equal the length of the input dataset!')
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    current_index = 0
    for s in split:
        yield Subset(dataset=dataset, indices=indices[current_index: current_index + s])
        current_index += s
