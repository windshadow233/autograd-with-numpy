from nptorch.tensor import *
import math


def default_collate_fn(all_data: list):
    data_list = []
    data_length = len(all_data[0])
    for i in range(data_length):
        data_list.append([data[i].data for data in all_data])
    for i, data in enumerate(data_list):
        data_list[i] = Tensor(np.r_[data])
    return data_list


class DataSet:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size=1, shuffle=True, collate_fn=default_collate_fn):
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
        return self

    def __next__(self):
        if not self.indices:
            raise StopIteration
        if len(self.indices) <= self.batch_size:
            return self.get_batches(self.indices)
        if self.shuffle:
            chosen_indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            chosen_indices = self.indices[:self.batch_size]
        self.indices = list(set(self.indices) - set(chosen_indices))
        return self.get_batches(chosen_indices)

    def get_batches(self, indices):
        all_data = [self.dataset[index] for index in indices]
        return self.collate_fn(all_data)


