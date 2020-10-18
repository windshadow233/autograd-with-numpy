import numpy as np


class DataSet:
    def __init__(self, data, labels, transform=lambda x: x):
        """
        初始化
        :param data: 数据矩阵,第0维度表示数据的索引,若为图片数据,则为四维张量
                     若为向量数据,则为二维矩阵。第一维度统一为数据索引
        :param labels: 标签
        :param transform: 数据变换函数
        """
        self.data = transform(data)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value[0]
        self.labels[key] = value[1]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class DataLoader:
    """
    用来批量打包数据
    """
    def __init__(self, dataset: DataSet, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(len(dataset) / self.batch_size))

        if shuffle:
            np.random.shuffle(self.dataset)

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        if self.counter >= self.n_batches:
            raise StopIteration
        if self.counter < self.n_batches - 1:
            data, labels = self.dataset[self.counter * self.batch_size: (self.counter + 1) * self.batch_size]
        else:
            data, labels = self.dataset[self.counter * self.batch_size:]
        self.counter += 1
        return data, labels


def random_split(dataset: DataSet, lens):
    if sum(lens) != len(dataset):
        raise RuntimeError('Sum of input lengths does not equal the length of the input dataset!')
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    datasets = []
    place = 0
    for l in lens:
        datasets.append(DataSet(*dataset[indices[place: place + l]]))
        place += l
    return datasets
