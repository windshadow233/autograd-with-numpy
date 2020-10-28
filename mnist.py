import struct
from tqdm import tqdm
import numpy as np
import nptorch
from nptorch import random
from nptorch import nn
from nptorch.optim import SGD, Adam
from nptorch.transforms import Compose, ToTensor, Resize, ToPILImage
from nptorch.utils.data import Dataset, DataLoader

trans = Compose([ToPILImage(),
                 Resize((32, 32)),
                 ToTensor()])


class MNISTDataset(Dataset):
    def __init__(self, data_path, label_path):
        super(MNISTDataset, self).__init__()
        self.data, self.label = load_mnist(data_path, label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return trans(self.data[item]), self.label[item]


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
        )
        self.layers2 = nn.Sequential(
            nn.Linear(32 * 25, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10)
        )

    def forward(self, x: nptorch.Tensor):
        x = self.layers1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layers2(x)
        return x


def test_model(model, test_loader: DataLoader):
    with nptorch.no_grad():
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
        images = np.fromfile(img, dtype=np.uint8).reshape(num, rows, cols)
    return images, nptorch.array(labels)


random.seed(0)
train_set = MNISTDataset('mnist/MNIST/raw/train-images-idx3-ubyte', 'mnist/MNIST/raw/train-labels-idx1-ubyte')
test_set = MNISTDataset('mnist/MNIST/raw/t10k-images-idx3-ubyte', 'mnist/MNIST/raw/t10k-labels-idx1-ubyte')
train_loader = DataLoader(train_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=128)

model = LeNet()
optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.7)
loss_fcn = nn.CrossEntropyLoss()

for i in tqdm(range(5)):
    count = 0
    for n, data in enumerate(train_loader, 1):
        model.train()
        d, lb = data
        count += len(d)
        print(n)
        print(count)
        print('标签:', lb)
        y_hat = model(d)
        loss = loss_fcn(y_hat, lb)
        loss.backward()
        optimizer.step()
        model.eval()
        p = model(d).argmax(-1)
        print('优化后预测:', p)
        optimizer.zero_grad()
        print(f'优化后的准确比率:{(p == lb).float().sum().item() / len(d)}')


print(f'测试集准确率{test_model(model, test_loader)}')
print(f'训练集准确率{test_model(model, train_loader)}')
