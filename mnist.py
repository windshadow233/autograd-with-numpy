import struct
from tqdm import tqdm
import numpy as np
import nptorch
from nptorch import random
from nptorch import nn
from nptorch.optim import SGD
from nptorch import transforms as T
from nptorch.utils.data import Dataset, DataLoader

trans = T.Compose([T.ToPILImage(),
                   T.Resize((32, 32)),
                   T.ToTensor(),
                   T.Normalize([0.5], [0.5])])


def load_mnist(img_path, label_path):
    with open(label_path, 'rb') as label:
        struct.unpack('>II', label.read(8))
        labels = np.fromfile(label, dtype=np.uint8)
    with open(img_path, 'rb') as img:
        _, num, rows, cols = struct.unpack('>IIII', img.read(16))
        images = np.fromfile(img, dtype=np.uint8).reshape(num, rows, cols)
    return images, labels


class MNISTDataset(Dataset):
    def __init__(self, data_path, label_path):
        super(MNISTDataset, self).__init__()
        self.data, self.label = load_mnist(data_path, label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return trans(self.data[item]), nptorch.tensor(self.label[item])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, dilation=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, dilation=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32 * 25, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x: nptorch.Tensor):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        return x


@nptorch.no_grad()
def test_model(model, test_loader: DataLoader):
    model.eval()
    count = 0
    for d, lb in tqdm(test_loader):
        p = model(d).argmax(-1)
        count += (p == lb).float().sum()
    return count.item() / len(test_loader.dataset)


random.seed(0)
train_set = MNISTDataset('mnist/MNIST/raw/train-images-idx3-ubyte', 'mnist/MNIST/raw/train-labels-idx1-ubyte')
test_set = MNISTDataset('mnist/MNIST/raw/t10k-images-idx3-ubyte', 'mnist/MNIST/raw/t10k-labels-idx1-ubyte')
train_loader = DataLoader(train_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=128)

model = LeNet()
optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.7)
loss_fcn = nn.CrossEntropyLoss()

# for i in tqdm(range(5)):
#     count = 0
#     for d, lb in train_loader:
#         model.train()
#         count += len(d)
#         print(count)
#         print('标签:', lb)
#         y_hat = model(d)
#         loss = loss_fcn(y_hat, lb)
#         loss.backward()
#         optimizer.step()
#         model.eval()
#         with nptorch.no_grad():
#             p = model(d).argmax(-1)
#             print('优化后预测:', p)
#             print(f'优化后的准确比率:{(p == lb).float().sum().item() / len(d)}')
#         optimizer.zero_grad()


print(model.load_state_dict('LeNet.pkl'))
print(f'测试集准确率{test_model(model, test_loader)}')
print(f'训练集准确率{test_model(model, train_loader)}')
