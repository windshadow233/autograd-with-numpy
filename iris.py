from sklearn.datasets import load_iris
from tqdm import tqdm
import nptorch
from nptorch import random
from nptorch import nn
from nptorch.optim import SGD
from nptorch.utils.data import DataSet, DataLoader, random_split


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(4, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(30, 10)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        x = self.activate(x)
        x = self.linear3(x)
        return x


def test_model(model, test_dataset: DataLoader):
    count = 0
    for d, lb in test_dataset:
        p = model(d).argmax(-1)
        count += (p == lb).float().sum()
    return count.item() / len(test_dataset.dataset)


def load_iris_data():
    iris = load_iris()
    data = nptorch.array(iris['data'])
    labels = nptorch.array(iris['target'])
    return DataSet(data, labels)


random.seed(1)
train, test = random_split(load_iris_data(), [120, 30])
train = DataLoader(train, batch_size=10)
test = DataLoader(test, batch_size=10)
dnn = DNN()
optim = SGD(dnn.parameters(), lr=0.1, weight_decay=0.01)
loss_fcn = nn.CrossEntropyLoss()
for i in tqdm(range(120)):
    count = 0
    for n, data in enumerate(train, 1):
        d, lb = data
        count += len(d)
        print(count)
        y_hat = dnn(d)
        loss = loss_fcn(y_hat, lb)
        loss.backward()
        optim.step()
        p = dnn(d).argmax(-1)
        print('优化后预测:', p)
        optim.zero_grad()
        print(f'优化后的准确比率:{(p == lb).float().sum().item() / len(d)}')


print(f'测试集准确率{test_model(dnn, test)}')
print(f'训练集准确率{test_model(dnn, train)}')