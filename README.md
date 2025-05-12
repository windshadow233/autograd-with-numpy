# Autograd with Numpy
本项目用于自己学习,旨在用 Numpy 实现计算图,进行梯度的自动计算,从而能完成一些深度学习任务。

本项目所有运算都通过调用 Numpy 接口实现,其本身仅仅就是套了个壳。

项目依赖的第三方库：numpy==1.19.1,pillow==8.0.1
## 使用文档
本项目使用风格与 Numpy 和 PyTorch 类似（基本是后者的子集）,如果你会用 PyTorch, 那么下面的基本不用看了。

对于此项目的解读请[点击此处](https://blog.fyz666.xyz/blog/5640/)。

### 基本使用
```python
import nptorch as nt
# 定义张量,可指定数据类型,返回Tensor类型
x = nt.tensor([1., 2., 3.], dtype=nt.float32, requires_grad=True)
y = nt.tensor([2., 3., 4.], dtype=nt.float32)
print(x)
print(type(x))
# array([1., 2., 3.], dtype=float32, requires_grad=True)
# <class 'nptorch.tensor.Tensor'>

# 转换数据类型
x = x.int()
x = x.long()

# 四则运算与矩阵运算,为了方便只定义了两个矩阵乘法函数
# matmul是矩阵乘法运算,相当于`@`,支持broadcast,当前例子中是点积
# outer可以进行向量外积,matmul也可以做到不过需要扩展维度
a = x + y
b = x.matmul(y)
b = x.outer(y)

# 随机函数调用,同样可以指定数据类型以及是否需要梯度
# 0-1上的均匀分布
x = nt.random.rand(3, 4, dtype=nt.float32, requires_grad=True)
# 正态分布
y = nt.random.normal(mean=0., std=1., size=(3, 4))
```
### 自动求导
1. 定义 float 类型的张量时,可指定参数 `requires_grad=True` 来声明需要对此张量求梯度。
2. 当得到标量结果时,可对该标量调用 `backward` 方法,得到与该标量相关的计算图中所有叶子节点的梯度。
3. 为节约空间,非叶子节点默认不会存储梯度,若需要其存储梯度,可调用其 `retain_grad` 方法。
4. 上下文管理器 `no_grad` 可实现内部计算不进行计算图的构建与求导, 也可作为函数装饰器, 在不需要记录梯度时使用。
```python
import nptorch as nt
x = nt.tensor([1., 2., 3.], dtype=nt.float32, requires_grad=True)
y = x * 2
y.retain_grad()
y.sum().backward()
print(y.grad)
print(x.grad)
# array([1., 1., 1.], dtype=float32)
# array([2., 2., 2.], dtype=float32)
with nt.no_grad():
# 下面的所有运算将不构建计算图
    y = x * 2
    print(y.children)
    print(type(y.grad_fn))
# []
# <class 'NoneType'>
# 离开no.grad上下文,下面的运算将正常构建计算图
y = x * 2
print(y.children)
print(type(y.grad_fn))
# [(array([1., 2., 3.], dtype=float32, requires_grad=True), None), (2, None)]
# <class 'nptorch.backward.MulBackward'>

# 作为函数装饰器使用,函数内的运算将不构建计算图
@nt.no_grad()
def f():
    x = nt.tensor([1., 2, 3], requires_grad=True)
    y = x * 2
    print(type(y.grad_fn))
f()
# <class 'NoneType'>
```
### 数据集
1. 项目封装了一个简陋的 `data` 模块用以封装数据, 功能比较简单。使用方法也类似 PyTorch, 不过功能少了一些。
2. 自定义的数据集类需要继承 `Dataset` 类并完善 `__len__` 与 `__getitem__` 方法。
3. `DataLoader` 初始化放入 `Dataset` 或 `Subset` 类型的实例,可选参数有 `batch_size`、`shuffle` 与 `collate_fn`, 并通过生成一个迭代器的方式批量获取其中的数据。
4. `random_split` 函数传入一个 `Dataset` 类型实例与一个长度列表, 返回按长度随机分割后的数据集。
5. 具体使用实例可以看 `mnist.py` 文件。
### 模型搭建与优化
内置模型的使用方法与 `PyTorch` 高度相似(但功能差远了), 优化器目前实现了 `SGD` 与 `Adam` 两种。以一个简单的卷积神经网络和 `SGD` 优化器为例：
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
# 调用模型的parameters方法获得模型所有待训练参数,它将返回一个生成器。
# 内置模型有默认待训练参数,若需要手动添加需训练参数,可使用nn.Parameter类
# nn.Parameter类初始化需要一个Tensor类型,返回Parameter类,它继承Tensor的所有运算并将自动计算梯度
# 优化器必须参数是可迭代对象,迭代结果为全部待优化的参数,学习率等超参数有默认值,可自行修改
optimizer = SGD(cnn.parameters(), lr=1e-2, momentum=0.9)
```
## 更新日志
### 2020/11/24
* 增加了 `Conv1d` 类。
### 2020/11/22
* 增加了梯度截断函数 `clip_grad_norm_`。
### 2020/11/21
* 增加了 `Embedding` 类, 梯度截断函数 `clip_grad_value_`。
### 2020/11/20
* 增加了 `ConcatDataset` 类,用以 `Dataset` 类之间的加法。
### 2020/11/19
* 为 `Module` 类增加了 `state_dict`、`save_state_dict` 与 `load_state_dict` 方法。
### 2020/11/18
* 增加了 `GRU` 类。
### 2020/11/17
* 增加了 `DatasetFolder` 与 `ImageFolder` 类,用以读取数据文件。
* 训练了一个孪生神经网络模型,在 LFW 测试集上达到了100%准确率。
### 2020/11/13
* 增加了 `LSTM` 类。
### 2020/11/12
* 增加了 `ModuleList` 类。
### 2020/11/10
* 对 `norm` 函数及其 backward 进行了修改, 主要增加了 `axis` 参数。
* 增加了 `pairwise_distance` 函数、`PairwiseDistance` 类、`cosine_similarity` 函数、`CosineSimilarity` 类。
### 2020/11/8
* 增加了 `pad_sequence`、`sort` 与 `argsort` 函数。
### 2020/11/5
* 增加了 `RandomCrop`、`RandomMask` 图像变换。
### 2020/11/4
* 增加了 `RandomHorizontalFlip`、`RandomVerticalFlip` 与 `CenterCrop` 三个图像变换。
### 2020/11/3
* 增加了 `BatchNorm1d`, 统一了 `BatchNorm` 的 forward 与 backward。
### 2020/10/30
* `Conv2d` 增加了参数 `dilation`,默认为(0, 0), 由于个人理解出现问题，这里的 `dilation` 相当于 PyTorch 的 `dilation - 1`。
* 实现了一个简单的 `RNN`。
### 2020/10/29
* 增加了 `Normalize` 变换。
### 2020/10/28
* 实现了一个简单的上下文管理器 `no_grad`, 在该上下文中的所有运算将不构造计算图。
* 增加了 `Softplus` 激活函数。
### 2020/10/27
* 增加了 `ToPILImage` 变换。
* 增加了 `MaxPool1d` 与 `AvgPool1d`。
### 2020/10/26
* 增加了 `Adam` 优化器。
* 增加了 `ToTensor`、`Reshape` 与 `GaryScale` 三个图像变换。
### 2020/10/25
* 增加了 `BatchNorm2d`、`Dropout2d`, 2d 部分基本写完了, 开始补作业, 隔段时间空了补充 1d。
### 2020/10/24
* 更新了 `Dataset`、`Subset`、`DataLoader` 类与 `random_split` 函数。
* 增加了 `NLLLoss`、`MSELoss`。
### 2020/10/23
* 调整了反向传播算法。
* 增加了激活函数 `LeakyReLU`、`ELU`。
* 增加模型保存方法, 目前只保存完整模型, 以后再写仅保存参数的。
### 2020/10/22
* 实现了一个简单的模型待训练参数类 `Parameter`。
### 2020/10/21
* 增加了二维的均值池化层和最大池化层。
### 2020/10/20
* 增加可做 padding 的二维卷积层 `Conv2d` 及其 backward,使用卷积 mnist 准确率进一步提升, ~~证明卷积没写错~~。
### 2020/10/19
* 实现 `Dropout` 层, 完善 `SGD`（增加动量, L1、L2正则化）。
* 模仿 PyTorch 对代码进行封装、模仿 PyTorch 实现模型的结构化打印功能。
### 2020/10/18
* 实现了一个简单的 `SGD` 优化器。
* 首次用线性层跑通 mnist 数据集。
### 2020/10/13~2020/10/17
* 尝试了 N 种封装方法,最后借助 Numpy 的 `ndarray` 类型封装了一个 `Tensor` 类, 作为计算图的节点类, 同时重写或补充了适用于 `Tensor` 类的上百个方法与函数, 基本的运算均实现了 backward。
* 实现线性层。
### 2020/10/12
产生“写个计算图玩玩”的 idea。
