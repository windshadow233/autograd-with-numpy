# autograd with numpy
本项目用于自己学习，旨在用numpy实现计算图，进行梯度的自动计算，从而能完成一些深度学习任务。

## 使用文档
本项目使用风格与numpy和pytorch类似，如果你会用pytorch，那么下面的基本不用看了。

### 基本使用
```python
import nptorch as nt
# 定义张量，可指定数据类型，返回Tensor类型
x = nt.array([1., 2., 3.], dtype=nt.float32)
y = nt.array([2., 3., 4.], dtype=nt.float32)
print(x)
print(type(x))
# array([1., 2., 3.], dtype=float32, requires_grad=True)
# <class 'nptorch.tensor.Tensor'>

# 转换数据类型
x = x.int()
x = x.long()

# 四则运算与矩阵运算，为了方便只定义了两矩阵乘法函数，
# matmul涵盖所有可能的矩阵乘，支持broadcast，当前例子中是点积
# outer可以进行向量外积，matmul也可以做到不过需要扩展维度
a = x + y
b = x.matmul(y)
b = x.outer(y)

# 随机函数调用，同样可以指定数据类型以及是否需要梯度
# 0-1上的均匀分布
x = nt.random.rand(size=(3, 4), dtype=nt.float32, requires_grad=True)
# 正态分布
x = nt.random.normal(size=(3, 4), mean=0., std=1.)
```
### 自动求导
1. 定义float类型的张量时，可指定参数requires_grad=True来声明需要对此张量求梯度。
2. 当得到标量结果时，可对该标量调用backward()方法，得到叶子节点的梯度。
3. 非叶子节点默认不会存储梯度，若需要其存储梯度，可调用其retain_grad()方法。
```python
import nptorch as nt
x = nt.array([1., 2., 3.], dtype=nt.float32, requires_grad=True)
y = x * 2
y.retain_grad()
y.sum().backward()
print(y.grad)
print(x.grad)
# array([1., 1., 1.], dtype=float32)
# array([2., 2., 2.], dtype=float32)
```
### 数据集
1. 项目封装了一个简单的data.py文件用以封装数据，功能比较简单。使用方法也类似pytorch，不过功能少了一些，以后再慢慢完善。
2. 自定义的数据集类需要继承DataSet类并完善__len__与__getitem__方法。
3. DataLoader初始化放入DataSet或SubSet类型的实例，可选参数有batch_size、shuffle与collate_fn，并通过生成一个迭代器的方式批量获取其中的数据。
4. random_split函数传入一个DataSet类型实例与一个长度列表，返回按长度随机分割后的数据集。
5. 具体使用实例可以看mnist.py文件。
### 模型搭建与使用
使用方法与pytorch高度相似，以卷积神经网络为例：
```python
from nptorch import nn
from nptorch.optim import SGD

# 模型创建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64 * 9, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        return x

cnn = CNN()
# 优化器，暂时只定义了SGD，支持动量与L1、L2正则化
# 通过.parameters()方法获得模型所有待训练参数
# 内置模型有默认待训练参数，若需要手动添加需训练参数，可使用nn.Parameter()类
# 该类输入一个Tensor类型，返回Parameter类，继承Tensor所有运算
optimizer = SGD(cnn.parameters(), lr=1e-2, momentum=0.9)
```
## 更新日志
### 2020/10/25
* 增加了BatchNorm2d，Dropout2d，2d部分基本写完了，开始补作业，隔段时间空了补充1d。
### 2020/10/24
* 更新了DataSet、SubSet、DataLoader类与random_split函数。
* 增加了NLLLoss、MSELoss。
### 2020/10/23
* 调整了反向传播算法。
* 增加了激活函数LeakyReLU，ELU。
* 增加模型保存方法，目前只保存完整模型，以后再写仅保存参数的。
### 2020/10/22
* 实现了一个简单的模型训练参数类Parameter。
### 2020/10/21
* 增加了二维的均值池化层和最大池化层。
### 2020/10/20
* 增加可做padding的二维卷积层Conv2d及其backward，使用卷积mnist准确率进一步提升，证明卷积没写错。
### 2020/10/19
* 实现Dropout层，完善SGD（增加动量，L1、L2正则化）
* 模仿pytorch对代码进行良好封装、模仿pytorch实现模型的可视化打印功能。
### 2020/10/18
* 实现了一个简单的SGD优化器。
* 首次用线性层跑通mnist数据集。
### 2020/10/13~2020/10/17
* 尝试了N种数据结构，最后借助numpy的ndarray类型封装了一个Tensor类，为计算图的节点，同时重写或补充了适用于Tensor类的上百个方法与函数，基本的运算均实现了backward。
* 实现线性层。
### 2020/10/12
产生“写个计算图玩玩”的idea。