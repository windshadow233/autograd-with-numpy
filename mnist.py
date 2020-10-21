import struct
from tqdm import tqdm
import numpy as np
import nptorch
from nptorch import random
from nptorch import nn
from nptorch.optim import SGD
from nptorch.utils.data import DataSet, DataLoader


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: nptorch.Tensor):
        x = self.layers(x)
        return x


def test_model(model, test_loader: DataLoader):
    model.eval()
    count = 0
    for d, lb in test_loader:
        p = model(d).argmax(-1)
        count += (p == lb).float().sum()
    return count.item() / len(test_loader.dataset)


def load_mnist(img_path, label_path):
    with open(label_path, 'rb') as label:
        struct.unpack('>II', label.read(8))
        labels = np.fromfile(label, dtype=np.uint8)
    with open(img_path, 'rb') as img:
        _, num, rows, cols = struct.unpack('>IIII', img.read(16))
        images = np.fromfile(img, dtype=np.uint8).reshape(num, rows * cols)
    return nptorch.array(images, dtype=np.float32), nptorch.array(labels)


random.seed(0)
train_data, train_lbs = load_mnist('mnist/MNIST/raw/train-images-idx3-ubyte', 'mnist/MNIST/raw/train-labels-idx1-ubyte')
test_data, test_lbs = load_mnist('mnist/MNIST/raw/t10k-images-idx3-ubyte', 'mnist/MNIST/raw/t10k-labels-idx1-ubyte')
train_set = DataSet(train_data, train_lbs, transform=lambda x: (x / 255))
test_set = DataSet(test_data, test_lbs, transform=lambda x: (x / 255))
train_loader = DataLoader(train_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

dnn = DNN()
optimizer = SGD(dnn.parameters(), lr=5e-2, momentum=0.7)
loss_fcn = nn.CrossEntropyLoss()

for i in tqdm(range(5)):
    count = 0
    for n, data in enumerate(train_loader, 1):
        dnn.train()
        d, lb = data
        count += len(d)
        print(n)
        print(count)
        print('标签:', lb)
        y_hat = dnn(d)
        loss = loss_fcn(y_hat, lb)
        loss.backward()
        optimizer.step()
        dnn.eval()
        p = dnn(d).argmax(-1)
        print('优化后预测:', p)
        optimizer.zero_grad()
        print(f'优化后的准确比率:{(p == lb).float().sum().item() / len(d)}')


print(f'测试集准确率{test_model(dnn, test_loader)}')
print(f'训练集准确率{test_model(dnn, train_loader)}')
