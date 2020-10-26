from nptorch.tensor import *
from .dataset import Dataset, Subset
import math


def default_collate_fn(all_data: list):
    data_list = []
    data_length = len(all_data[0])
    for i in range(data_length):
        data_list.append(array(np.r_[[data[i].data for data in all_data]]))
    return data_list


class DataLoader(object):
    def __init__(self, dataset: Dataset or Subset, batch_size=1, shuffle=True, collate_fn=default_collate_fn):
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



