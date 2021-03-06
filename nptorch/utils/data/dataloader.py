import nptorch
from nptorch.tensor import *
from .dataset import Dataset
import math


def default_collate_fn(all_data: list):
    """
    默认的数据整理函数,将一批中的每个数据进行stack
    @param all_data: 一个列表,元素类型为tuple
    @return: 按以上规则生成的数据元组
    """
    if not isinstance(all_data[0], tuple):
        return nptorch.stack(all_data)
    data_length = len(all_data[0])
    data = tuple(nptorch.stack([data[i] for data in all_data]) for i in range(data_length))
    return data


class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=default_collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        self.n_batches = math.floor(len(dataset) / batch_size) if drop_last else math.ceil(len(dataset) / batch_size)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if len(self.indices) < self.batch_size:
            if self.drop_last or not self.indices:
                raise StopIteration
            chosen_indices = self.indices
            self.indices = []
            return self.get_batch(chosen_indices)
        chosen_indices, self.indices = self.indices[:self.batch_size], self.indices[self.batch_size:]
        return self.get_batch(chosen_indices)

    def get_batch(self, indices):
        all_data = [self.dataset[index] for index in indices]
        return self.collate_fn(all_data)
