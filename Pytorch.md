## Pytorch
### 前置知识
参考[Perceptron_PytorchLib 项目目录](https://github.com/RobinElysia/PythonLibAction/tree/main/pytorch/Perceptron_PytorchLib)，包含手操微分、手写各种损失函数、激活函数、各种梯度算法、三层网络和BP反向传播等等。

### Pytorch Lib
#### CreateTensor（Tensor基本使用）
```python
import torch
import numpy as np

def Easy_tensor():
    """
    简单创建
    :return:
    """
    x = torch.tensor(1.0)
    x2 = torch.tensor([1.0, 2.0])
    print(x)
    print(x.dtype) # 默认为float32, numpy默认是float64，整数默认是int64（torch）
    print(x.size()) # 默认为空[]，因为是标量
    """
    tensor(1.)
    torch.float32
    torch.Size([])
    """

    print(x2)
    print(x2.dtype)
    print(x2.size())
    """
    tensor([1., 2.])
    torch.float32
    torch.Size([2])
    """

    x3 = torch.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    print(x3)
    print(x3.dtype)
    print(x3.size())
    """
    tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
    torch.float64
    torch.Size([2, 2])
    """

    # GPU数据
    cuda = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float64, device='cuda')
    # 转换数据
    x.to(device='cuda')
    print(cuda.device)
    print(x.device)


def Easy_Tensor():
    """
    指定维度大小预先创建，不管元素是什么，始终是float32
    且创建始为向量/矩阵，不可能是标量
    :return:
    """
    x = torch.Tensor(2, 3, 4)
    print(x)
    print(x.size())
    print(x.dtype)
    """
    tensor([[[-3.8086e-01,  1.4041e-42,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],

        [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]])
    torch.Size([2, 3, 4])
    torch.float32
    """

    x2 = torch.Tensor([[1, 2], [2, 3]])
    print(x2)
    print(x2.size())
    print(x2.dtype)
    """
    tensor([[1., 2.],
        [2., 3.]])
    torch.Size([2, 2])
    torch.float32
    """

    x3 = torch.Tensor(10)
    print(x3)
    print(x3.size())
    print(x3.dtype)
    """
    注意，是向量，不再是标量了！！！
    tensor([9.3592e+15, 1.7166e-42, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
    torch.Size([10])
    torch.float32
    """

def Type_Tensor_or_tensor():
    """
    类型前缀和属性指定类型
    :return:
    """
    x = torch.FloatTensor(2, 3, 4) # 形状生成
    # LongTensor、DoubleTensor、IntTensor、ShortTensor、ByteTensor、HalfTensor、BoolTensor
    print(x)
    print(x.size())
    print(x.dtype)
    """
    tensor([[[3.8707e+25, 1.7628e-42, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]],

        [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]])
    torch.Size([2, 3, 4])
    torch.float32
    """
    x1 = torch.tensor([1, 2, 3], dtype=torch.float) # 内容生成
    print(x1)
    print(x1.size())
    print(x1.dtype)
    """
    tensor([1., 2., 3.])
    torch.Size([3])
    torch.float32
    """

def Interval_tensor():
    """
    生成指定区间的tensor
    :return:
    """
    x = torch.arange(0, 10, 2) # 0-10，步长为2，左闭右开
    print(x)

    x1 = torch.linspace(0, 10, 5) # 0-10，生成5个，左闭右闭
    print(x1)

    x2 = torch.logspace(0, 10, 5, base=2) # 0-10，生成5个，左闭右闭，底数是2
    print(x2)

def full_tensor():
    """
    生成全0/1/x的tensor
    :return:
    """
    x = torch.zeros(2, 3, 4) # 生成全0的tensor
    print(x)
    x1 = torch.ones(2, 3, 4) # 生成全1的tensor
    print(x1)
    x2 = torch.full((2, 3, 4), 5) # 生成全5的tensor
    print(x2)
    x3 = torch.empty(2, 3, 4) # 随机生成指定大小的tensor
    print(x3)

    # 根据已知张量生成
    x4 = torch.zeros_like(x) # 根据x生成全0的tensor
    print(x4)
    x5 = torch.ones_like(x) # 根据x生成全1的tensor
    print(x5)
    x6 = torch.full_like(x, 5) # 根据x生成全5的tensor
    print(x6)
    x7 = torch.empty_like(x) # 根据x生成随机的tensor
    print(x7)

    # 单位阵
    x8 = torch.eye(3) # 单位阵
    print(x8)
    x9 = torch.eye(3, 4) # 3行4列
    print(x9)
    """
    tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
    tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]])
    """

def random_tensor():
    """
    生成随机数
    :return:
    """
    x = torch.rand(2, 3, 4) # 均匀分布随机生成三维[0,1)的 tensor
    print(x)
    x1 = torch.randint(0, 10, (2, 3, 4)) # 均匀分布随机生成三维[0,10)的 int tensor
    print(x1)
    x2 = torch.randn(2, 3, 4) # 标准正态分布随机生成三维[-1,1)的 tensor
    print(x2)
    x3 = torch.normal(0, 1, (2, 3, 4)) # 正态分布随机生成三维均值为 0，方差为 1 的 tensor

    # 根据已有张量生成
    x4 = torch.rand_like(x)
    print(x4)
    x5 = torch.randint_like(x, 0, 10) # 根据已有张量生成，在[0,10)内
    print(x5)
    x6 = torch.randn_like(x) # 根据已有张量生成，标准正态分布

def perm_tensor():
    """
    对tensor洗牌
    :return:
    """
    x = torch.randperm(10) # 0-9，随机排序
    print(x)
    print(torch.random.initial_seed()) # 获取当前种子
    torch.random.manual_seed(42) # 设置种子

if __name__ == '__main__':
    # Easy_tensor()
    print("==========")
    # Easy_Tensor()
    print("==========")
    # Type_Tensor_or_tensor()
    print("==========")
    # Interval_tensor()
    print("==========")
    # full_tensor()
    print("==========")
    # random_tensor()
    print("==========")
    # perm_tensor()
```

#### TransAndAlgTensor（Tensor的转换与运算）
```python
import torch
import numpy as np

def trans_type_tensor():
    """
    张量的改变类型
    :return:
    """
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    print(x.dtype)
    # 类型转换
    x1 = x.type(torch.int)
    print(x1.dtype)
    # 或者直接调用函数
    x2 = x.half()
    print(x2.dtype)
    # 再或者
    x3 = x.to(torch.complex64) # 64位复数
    print(x3.dtype)

def trans_np_tensor():
    """
    ndarray与tensor的转换
    :return:
    """
    # 设置打印选项
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    x = np.array([1, 2, 3])
    x1 = torch.from_numpy(x) # ndarray 转 tensor
    print(x1.dtype)
    x2 = x1.numpy() # tensor 转 ndarray
    # 上述转换是浅拷贝，即 x 和 x2 都指向同一块内存，如果 x1 改变，x2 也会改变

    # 取消内存共享
    # x2 = x1.numpy().copy() # 创建新的数组
    # x1 = torch.from_numpy(x2.copy()) # 创建新的张量

def Scalar_to_Tensor():
    """
    标量与张量的转换
    :return:
    """
    x = torch.tensor(10) # 创建张量
    # x = torch.tensor([10]) 也可以
    # x = torch.tensor([[10]]) # 也也可以
    print(x)
    # 转换为标量
    print(x.item())

def Alg_tensor():
    """
    张量的运算
    :return:
    """
    x = torch.randint(10, 100, (2, 3))
    print(x + 10)
    print(x - 10)
    print(-x)
    print(x * 10)
    print(x / 10)
    print(x ** 2)
    print(x // 10)

    # 函数
    print(x.add(10)) # 加
    print(x.neg()) # 取负数
    print(x.sub(10)) # 减
    print(x.mul(10)) # 乘，哈达玛积，即对应位置相乘
    print(x.div(10)) # 除
    print(x.pow(2)) # 幂
    print(x.sqrt()) # 平方根
    print(x.exp()) # 指数

    # 取代原对象数据（注意类型不一不可进行该操作）
    # print(x.add_(10))  # 加
    # print(x.neg_())  # 取负数
    # print(x.sub_(10))  # 减
    # print(x.mul_(10))  # 乘，哈达玛积，即对应位置相乘
    # print(x.div_(10))  # 除
    # print(x.pow_(2))  # 幂，可以传分数实现开方
    # print(x.sqrt_())  # 平方根
    # ...

    # 矩阵乘法
    x = torch.randint(10, 100, (2, 3))
    y = torch.randint(10, 100, (3, 4))
    print(x @ y) # 语法糖
    print(torch.matmul(x, y)) # 多维矩阵乘法
    print(torch.mm(x, y)) # 二维专用矩阵乘法

    # 注意：x += 10和x = x + 10的区别。两者是不同的，前者是原地操作，后者是创建新的张量。
    # x += 10类似于x = x.add_(10)，x = x + 10类似于x = x.add(10)。
    # 其他的一样，@=不同上的原理，张量运算的形状发生变化不可以赋值回去，ndarray也是一样的，形状不同直接报错
    # 你可以使用以下操作来节省内存
    # x[0:0] = x @ y，但是形状会出问题，缺的数据会被有的数据广播覆盖，所以这么用得符合广播的规则

def Alg_Func_tensor():
    """
    函数统计运算
    """
    x = torch.randint(10, 100, (2, 3, 4))
    print(x)
    print(x.sum(dim=0)) # 矩阵去第三维求和
    print(x.sum(dim=1)) # 矩阵去第二维求和
    print(x.sum(dim=2)) # 矩阵去第一维求和

    # 均值
    # 只能是复数或者浮点数
    print(x.mean(dim=0))
    print(x.mean(dim=1))
    print(x.mean(dim=2))

    # 标准差
    print(x.std(dim=0))
    print(x.std(dim=1))
    print(x.std(dim=2))

    # 最大值
    print(x.max(dim=0))
    print(x.max(dim=1))
    print(x.max(dim=2))
    # 最小值
    print(x.min(dim=0))
    print(x.min(dim=1))
    print(x.min(dim=2))
    # 得到的是最大值和最小值的索引和值

    # 或者直接得到索引
    print(x.argmax(dim=0))
    print(x.argmin(dim=0))

    # 去重
    print(x.unique()) # 也可以加维度

    # 排序
    print(x.sort()) # 默认升序，有值和索引，可以加维度


if __name__ == '__main__':
    """
    转换和计算tensor
    """
    # trans_type_tensor()
    print("==========")
    # trans_np_tensor()
    print("==========")
    # Scalar_to_Tensor()
    print("==========")
    # Alg_tensor()
    print("==========")
    # Alg_Func_tensor()
```

#### IndexAndChangeTensor（Tensor维度形状改变）
```python
import torch
import numpy as np

def index_tensor():
    """
    索引张量
    """
    x = torch.randint(10, 100, (2, 3, 4))
    print(x)
    print(x[1, 1, 1]) # 索引(2,2,2)
    print(x[1]) # 索引第一个矩阵
    print(x[:, 1]) # 索引所有矩阵的第二行
    print(x[1:]) # 索引除了第一个的所有矩阵
    print(x[-1:, 1:3, 1:3]) # 索引最后一个矩阵的第一、二列第一、二行
    # 你也可以加步长 起始:结束:步长，[起始，结束)步长
    # tensor不可负数做步长
    """
    tensor([[[88, 16, 80, 20],
         [75, 75, 45, 56],
         [61, 87, 59, 55]],

        [[61, 45, 84, 49],
         [83, 39, 63, 54],
         [43, 49, 14, 92]]])
    """

    # 列表索引
    print(x[[1, 0], [2, 0], [1, 0]])
    # 1，0
    # 2，0
    # 1，0
    # 第二个矩阵的第三行的第二列，第一个矩阵的第一行的第一列
    print(x[[[1], [0]], [2, 0]])
    # 1，1，0，0
    # 2，0，2，0
    # 索引((1,2),(1,0)),((0,2),(0,0))
    # 这是一个三维的索引

    #布尔索引
    print(x[x > 50]) # 索引大于50的元素

    # 转置操作
    # print(x.T) 全部转置
    # print(x.mT) 内部矩阵转置
    # print(x.permute(1, 2, 0)) # 指定维度转置

def change_tensor():
    """
    改变张量
    """
    # 交换维度
    x = torch.randint(10, 100, (2, 3, 4))
    print(x.transpose(1, 2)) # 交换0，1，2中，1，2的维度

    # 维度重排
    print(x.permute(1, 2, 0)) # 改变维度的顺序为1，2，0

    # 完全改变
    print(x.reshape(12, 2))
    print(x.reshape(6, 4)) # 6*4=2*3*4
    print(x.reshape(-1)) # 改变维度为1维

    # view试图翻转
    tensor1 = torch.randint(1, 9, (3, 5, 4))
    print(tensor1)
    print(tensor1.is_contiguous())  # is_contiguous()判断是否内存连续
    print(tensor1.contiguous().view(-1))  # contiguous()强制内存连续
    print(tensor1.view(-1, 10))
    # 转置后不是连续的了，所以view()会报错
    tensor1 = tensor1.T
    print(tensor1.is_contiguous())  # is_contiguous()判断是否内存连续
    print(tensor1.contiguous().view(-1))  # contiguous()强制内存连续

def add_div_tensor():
    """
    张量维度的加减
    """
    x = torch.tensor([1,2,3,4])
    print(x.unsqueeze(dim = 0)) # 增加一个0维度
    print(x.unsqueeze(dim = 1)) # 添加一个1维度
    # 或者直接原地改：x.unsqueeze_(dim = 0)

    # 删除一个维度
    print(x.squeeze(dim = 0))
    print(x.squeeze(dim = 1))
    # 或者直接原地改：x.squeeze_(dim = 0)

def Splicing_tensor():
    """
    拼接张量
    :return:
    """
    # 拼接要求其他维度大小必须一样
    x = torch.randint(10, 100, (2, 3))
    y = torch.randint(10, 100, (2, 3))
    print(torch.cat([x, y], dim = 0))
    # 堆叠，所有的维度大小必须一样
    print(torch.stack([x, y], dim = 0))

if __name__ == '__main__':
    """
    索引和改变张量
    """
    # index_tensor()
    print("==========")
    # change_tensor()
    print("==========")
    # add_div_tensor()
    print("==========")
    # Splicing_tensor()
```

#### AutoGrad（动态计算图微分入门）
```python
import torch

def grad_tensor():
    """
    简单的自动梯度计算与反向传播
    :return:
    """
    x = torch.tensor(1)
    y = torch.tensor([[2]])
    w = torch.rand(1,1,requires_grad=True)
    b = torch.rand(1,1,requires_grad=True)
    z = w * x + b
    print(x.is_leaf)
    print(w.is_leaf)
    print(z.is_leaf) # 不是叶子节点
    print(b.is_leaf)
    print(y.is_leaf)
    # 叶子节点计算梯度会自动保存，非叶子节点不会保存梯度
    # 开启return_graph=True，非叶子节点会保存梯度

    loss = torch.nn.MSELoss()
    loss_v = loss(z, y)
    print(loss_v)
    print(loss_v.is_leaf)

    # 反向传播
    loss_v.backward()
    print(w.grad)
    print(b.grad)

def other_detach_grad_tensor():
    """
    节点分支单独运算
    :return:
    """
    x = torch.tensor(1, requires_grad=True)
    y = x.detach()
    print(y.requires_grad)
    print(x.requires_grad)
    print(id(x)) # id不同
    print(id(y))
    print(x.untyped_storage()) # 数据共享
    print(y.untyped_storage())

def data_vs_detach():
    """
    张量直接data和detach的区别
    :return:
    """
    # 当你试图使用data对节点进行修改时，妄图对修改后的节点做梯度，会成功，并且会直接影响梯度运算，这不是我们希望的
    # 而使用detach()时，张量的数据会进行复制，不会对原张量进行修改，但是对梯度运算没有影响


if __name__ == "__main__":
    grad_tensor()
    print("------------------")
    other_detach_grad_tensor()
```

#### ActFunc（Torch的激活函数）
```python
import torch

def NN_Sigmoid():
    x = torch.randn(1, 1)
    print(x.sum().sigmoid())

def NN_Tanh():
    x = torch.randn(1, 1)
    print(x.sum().tanh())

def NN_ReLU():
    x = torch.randn(1, 1)
    print(x.sum().relu())

def NN_Softmax():
    x = torch.randn(1, 2)
    print(x.softmax(dim=1))

if __name__ == '__main__':
    """
    torch实现的激活函数
    """
    NN_Sigmoid()
```

#### ValueInitAndDropout（w与b的初始化与正则化）
```python
import torch.nn as nn
import torch

def con_init():
    """
    常数初始化
    :return:
    """
    linear = nn.Linear(5, 2)

    # 全部参数初始化为0
    nn.init.zeros_(linear.weight)
    print(linear.weight)

    # 全部参数初始化为1
    nn.init.ones_(linear.weight)
    print(linear.weight)

    # 全部参数初始化为一个常数
    nn.init.constant_(linear.weight, 10)
    print(linear.weight)

def eye_init():
    """
    秩初始化
    :return:
    """
    linear = nn.Linear(5, 2)

    # 参数初始化为单位矩阵
    nn.init.eye_(linear.weight)
    print(linear.weight)

def normal_init():
    """
    正态分布初始化
    :return:
    """
    linear = nn.Linear(5, 2)

    # 参数初始化为按指定均值与标准差正态分布
    nn.init.normal_(linear.weight, mean=0.0, std=1.0)
    print(linear.weight)

def uniform_init():
    """
    均匀分布初始化
    :return:
    """
    linear = nn.Linear(5, 2)

    # 参数初始化为在区间内均匀分布
    nn.init.uniform_(linear.weight, a=0, b=10)
    print(linear.weight)

def xavier_init():
    """
    Xavier初始化
    :return:
    """
    linear = nn.Linear(5, 2)

    # Xavier正态分布初始化
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight)

    # Xavier均匀分布初始化
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight)

def kaiming_init():
    """
    何凯明初始化
    """
    linear = nn.Linear(5, 2)

    # Kaiming正态分布初始化
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight)

    # Kaiming均匀分布初始化
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight)

def Dropout_init():
    """
    Dropout初始化
    """
    dropout = torch.nn.Dropout(p=0.5)
    x = torch.randint(1, 10, (10,), dtype=torch.float32)
    print("Dropout前：", x)
    print("Dropout后：", dropout(x))

if __name__ == '__main__':
    """
    常见的W和b初始化与正则化（随机失活）
    """
    con_init()
    print("--------------------")
    eye_init()
    print("--------------------")
    normal_init()
    print("--------------------")
    uniform_init()
    print("--------------------")
    xavier_init()
    print("--------------------")
    kaiming_init()
```

#### ClassificationLoss（分类）
```python
import torch
import torch.nn as nn

def Classification_NN():
    # 真实值
    target = torch.tensor([[1], [0], [0]], dtype=torch.float32)
    # 预测值
    input = torch.randn((3, 1))
    prediction = torch.sigmoid(input) # 经过激活函数进行 0 1 预测
    # 实例化损失函数
    loss = nn.BCELoss() # 二分类损失函数
    print(loss(prediction, target)) # 计算损失

def Regression_NN():
    # 真实值为标签
    target = torch.tensor([1, 0, 3, 2, 5, 4])  # 真实值
    input = torch.randn((6, 8))  # 预测值
    loss = nn.CrossEntropyLoss()  # 实例化损失函数
    print(loss(input, target))

    # 真实值为概率
    target = torch.randn(6, 8).softmax(dim=1)  # 真实值
    input = torch.randn((6, 8))  # 预测值
    loss = nn.CrossEntropyLoss()  # 分类交叉熵损失函数
    print(loss(input, target))


if __name__ == "__main__":
    """
    简单的了解分类任务和回归任务如何实现
    """
    Classification_NN()
    print("----------------------")
    Regression_NN()
```

#### AndRegressionLoss（回归）
```python
import torch.nn as nn
import torch

def get_loss():
    """
    回归任务常用损失
    :return:
    """
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss_L2 = nn.MSELoss()
    print(loss_L2(tensor, tensor))

    # L1 损失
    loss_L1 = nn.L1Loss()
    print(loss_L1(tensor, tensor))

    # smooth L1 损失
    loss_smoothL1 = nn.SmoothL1Loss()
    print(loss_smoothL1(tensor, tensor))


if __name__ == '__main__':
    get_loss()
```

### optim优化器

#### 动量优化器
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 1. 创建简单的训练数据
# 生成一个简单的线性关系：y = 2x + 1 + 噪声
n_samples = 100
x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
y = 2 * x + 1 + torch.randn(n_samples, 1) * 0.5


# 2. 定义简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)


# 3. 创建模型、损失函数和优化器
model = LinearModel()

# 使用带动量的SGD优化器
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 作为对比，也创建一个没有动量的优化器
model_no_momentum = LinearModel()
optimizer_no_momentum = optim.SGD(model_no_momentum.parameters(), lr=0.01, momentum=0)

criterion = nn.MSELoss()  # 均方误差损失


# 4. 训练函数
def train_model(model, optimizer, epochs=200):
    losses = []
    for epoch in range(epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


# 5. 训练两个模型进行比较
print("使用动量的训练过程:")
losses_momentum = train_model(model, optimizer_momentum)

print("\n不使用动量的训练过程:")
losses_no_momentum = train_model(model_no_momentum, optimizer_no_momentum)

# 6. 可视化结果
plt.figure(figsize=(12, 4))

# 损失曲线对比
plt.subplot(1, 2, 1)
plt.plot(losses_momentum, 'b-', label='With Momentum (0.9)')
plt.plot(losses_no_momentum, 'r-', label='Without Momentum')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 拟合结果对比
plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label='Data')

# 使用动量的预测结果
with torch.no_grad():
    y_pred_momentum = model(x)
    y_pred_no_momentum = model_no_momentum(x)

plt.plot(x.numpy(), y_pred_momentum.numpy(), 'b-', linewidth=2, label='With Momentum')
plt.plot(x.numpy(), y_pred_no_momentum.numpy(), 'r-', linewidth=2, label='Without Momentum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting Results')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 查看最终参数
print("\n最终参数比较:")
print("使用动量的模型参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

print("\n不使用动量的模型参数:")
for name, param in model_no_momentum.named_parameters():
    print(f"{name}: {param.data}")

# 8. 动量的更详细示例
print("\n" + "=" * 50)
print("动量的工作原理示例:")


# 创建一个简单的二次函数优化问题
def f(x):
    return x ** 2 + 10 * x + 25


# 手动实现带Momentum的梯度下降
def momentum_optimization(start_x=0.0, lr=0.1, momentum=0.9, n_iter=30):
    x = torch.tensor([start_x], requires_grad=True)
    velocity = torch.zeros(1)
    positions = [x.item()]

    for i in range(n_iter):
        # 计算梯度
        y = f(x)
        y.backward()

        # Momentum更新
        velocity = momentum * velocity + lr * x.grad
        x.data = x.data - velocity

        # 重置梯度
        x.grad.zero_()

        positions.append(x.item())

    return positions


# 比较不同动量值的效果
for momentum in [0, 0.5, 0.9]:
    positions = momentum_optimization(start_x=10.0, momentum=momentum)
    print(f"Momentum={momentum}: 最终位置={positions[-1]:.6f}, 迭代次数={len(positions)}")
```

#### 学习率衰减优化器
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, MultiStepLR, LambdaLR
)

"""
StepLR：固定步长衰减，每隔固定epoch衰减一次
ExponentialLR：指数衰减，每个epoch都衰减
CosineAnnealingLR：余弦退火，学习率按余弦函数变化
ReduceLROnPlateau：基于验证指标衰减，当指标不再改善时衰减
MultiStepLR：多步长衰减，在指定epoch处衰减
LambdaLR：自定义衰减函数，灵活度高
CosineAnnealingWarmRestarts：余弦退火热重启，周期性重置学习率
"""

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 1. 创建训练数据 - 一个更复杂的非线性问题
n_samples = 200
x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
y = torch.sin(x) + 0.3 * torch.randn(n_samples, 1)  # 正弦函数加噪声


# 2. 定义更复杂的神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# 3. 创建多个相同初始状态的模型，用于比较不同的学习率调度器
def create_model_and_optimizer():
    model = NeuralNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


# 4. 训练函数，记录学习率和损失
def train_with_scheduler(model, optimizer, scheduler, epochs=200):
    losses = []
    learning_rates = []

    for epoch in range(epochs):
        # 前向传播
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        if scheduler is not None:
            # ReduceLROnPlateau需要传入验证指标
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        # 记录
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 40 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return losses, learning_rates


# 5. 定义不同的学习率调度策略
print("=" * 60)
print("不同学习率衰减策略比较")
print("=" * 60)

# 策略1: 固定学习率（无调度器）
print("\n1. 固定学习率 (无调度器)")
model1, optimizer1 = create_model_and_optimizer()
losses1, lrs1 = train_with_scheduler(model1, optimizer1, None)

# 策略2: 步长衰减
print("\n2. 步长衰减 (每50个epoch衰减为原来的0.5倍)")
model2, optimizer2 = create_model_and_optimizer()
scheduler2 = StepLR(optimizer2, step_size=50, gamma=0.5)
losses2, lrs2 = train_with_scheduler(model2, optimizer2, scheduler2)

# 策略3: 指数衰减
print("\n3. 指数衰减 (每个epoch衰减为原来的0.995倍)")
model3, optimizer3 = create_model_and_optimizer()
scheduler3 = ExponentialLR(optimizer3, gamma=0.995)
losses3, lrs3 = train_with_scheduler(model3, optimizer3, scheduler3)

# 策略4: 余弦退火
print("\n4. 余弦退火 (从初始学习率退火到0)")
model4, optimizer4 = create_model_and_optimizer()
scheduler4 = CosineAnnealingLR(optimizer4, T_max=200, eta_min=0)
losses4, lrs4 = train_with_scheduler(model4, optimizer4, scheduler4)

# 策略5: 基于验证损失的衰减
print("\n5. 基于验证损失的衰减 (当损失不再下降时衰减)")
model5, optimizer5 = create_model_and_optimizer()
scheduler5 = ReduceLROnPlateau(optimizer5, mode='min', factor=0.5, patience=10, verbose=False)
losses5, lrs5 = train_with_scheduler(model5, optimizer5, scheduler5)

# 策略6: 多步长衰减
print("\n6. 多步长衰减 (在epoch 30, 80, 120时衰减)")
model6, optimizer6 = create_model_and_optimizer()
scheduler6 = MultiStepLR(optimizer6, milestones=[30, 80, 120], gamma=0.5)
losses6, lrs6 = train_with_scheduler(model6, optimizer6, scheduler6)

# 策略7: 自定义Lambda衰减
print("\n7. 自定义Lambda衰减 (学习率 = 初始学习率 * (1 + 10*epoch/总epochs)^-0.75)")
model7, optimizer7 = create_model_and_optimizer()
lambda_func = lambda epoch: (1 + 10 * epoch / 200) ** -0.75
scheduler7 = LambdaLR(optimizer7, lr_lambda=lambda_func)
losses7, lrs7 = train_with_scheduler(model7, optimizer7, scheduler7)

# 6. 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 学习率变化曲线
ax1 = axes[0, 0]
epochs = range(200)
ax1.plot(epochs, lrs1, label='固定学习率')
ax1.plot(epochs, lrs2, label='步长衰减')
ax1.plot(epochs, lrs3, label='指数衰减')
ax1.plot(epochs, lrs4, label='余弦退火')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate')
ax1.set_title('学习率变化曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 损失变化曲线
ax2 = axes[0, 1]
ax2.plot(epochs, losses1, label='固定学习率')
ax2.plot(epochs, losses2, label='步长衰减')
ax2.plot(epochs, losses3, label='指数衰减')
ax2.plot(epochs, losses4, label='余弦退火')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('损失变化曲线')
ax2.set_yscale('log')  # 对数坐标更清晰
ax2.legend()
ax2.grid(True, alpha=0.3)

# 其他调度策略的学习率变化
ax3 = axes[1, 0]
ax3.plot(epochs, lrs5, label='基于验证损失衰减')
ax3.plot(epochs, lrs6, label='多步长衰减')
ax3.plot(epochs, lrs7, label='自定义Lambda衰减')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('其他调度策略的学习率变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 拟合结果对比
ax4 = axes[1, 1]
ax4.scatter(x.numpy(), y.numpy(), alpha=0.3, label='原始数据', s=10)

with torch.no_grad():
    y_pred1 = model1(x)
    y_pred4 = model4(x)
    y_pred7 = model7(x)

ax4.plot(x.numpy(), y_pred1.numpy(), 'r-', linewidth=2, label='固定学习率')
ax4.plot(x.numpy(), y_pred4.numpy(), 'g-', linewidth=2, label='余弦退火')
ax4.plot(x.numpy(), y_pred7.numpy(), 'b-', linewidth=2, label='Lambda衰减')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('不同调度策略的拟合结果')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 详细展示一种调度器的使用（StepLR）
print("\n" + "=" * 60)
print("StepLR调度器详细示例")
print("=" * 60)

# 重新创建一个模型
model_demo, optimizer_demo = create_model_and_optimizer()

# 创建StepLR调度器：每25个epoch学习率乘以0.8
scheduler_demo = StepLR(optimizer_demo, step_size=25, gamma=0.8)

print("初始学习率:", optimizer_demo.param_groups[0]['lr'])

# 模拟训练过程
for epoch in range(100):
    # 模拟训练步骤
    optimizer_demo.step()  # 实际训练中这里会有loss.backward()

    # 更新学习率
    scheduler_demo.step()

    # 每10个epoch打印一次学习率
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: 学习率 = {optimizer_demo.param_groups[0]["lr"]:.6f}')

# 8. 结合动量的学习率衰减示例
print("\n" + "=" * 60)
print("结合动量Momentum和学习率衰减的综合示例")
print("=" * 60)

# 创建模型
model_combined = NeuralNet()

# 使用带动量的SGD优化器
optimizer_combined = optim.SGD(
    model_combined.parameters(),
    lr=0.1,
    momentum=0.9,  # 添加动量
    weight_decay=1e-4  # L2正则化
)

# 使用余弦退火结合热重启的调度器
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler_combined = CosineAnnealingWarmRestarts(
    optimizer_combined,
    T_0=50,  # 第一个周期的长度
    T_mult=2,  # 每个周期长度加倍
    eta_min=0.001  # 最小学习率
)

print("训练过程:")
for epoch in range(150):
    # 模拟训练步骤
    loss = torch.randn(1).item() * 0.1  # 模拟损失

    # 优化步骤
    optimizer_combined.zero_grad()
    # loss.backward()  # 实际训练中会有反向传播
    optimizer_combined.step()

    # 更新学习率
    scheduler_combined.step()

    if (epoch + 1) % 25 == 0:
        print(f'Epoch {epoch + 1}: LR={optimizer_combined.param_groups[0]["lr"]:.6f}, '
              f'Momentum={optimizer_combined.param_groups[0]["momentum"]}')

# 9. 学习率调度器的状态保存与恢复示例
print("\n" + "=" * 60)
print("学习率调度器状态保存与恢复示例")
print("=" * 60)

model_checkpoint = NeuralNet()
optimizer_checkpoint = optim.SGD(model_checkpoint.parameters(), lr=0.1)
scheduler_checkpoint = StepLR(optimizer_checkpoint, step_size=20, gamma=0.7)

# 训练几个epoch
for epoch in range(10):
    scheduler_checkpoint.step()
    print(f'Epoch {epoch + 1}: LR={optimizer_checkpoint.param_groups[0]["lr"]:.6f}')

# 保存调度器状态
checkpoint = {
    'model_state_dict': model_checkpoint.state_dict(),
    'optimizer_state_dict': optimizer_checkpoint.state_dict(),
    'scheduler_state_dict': scheduler_checkpoint.state_dict(),
    'epoch': 10
}

print(f"\n保存检查点，当前epoch={checkpoint["epoch"]}, LR={optimizer_checkpoint.param_groups[0]['lr']:.6f}")

# 创建新模型和调度器
model_resume = NeuralNet()
optimizer_resume = optim.SGD(model_resume.parameters(), lr=0.1)
scheduler_resume = StepLR(optimizer_resume, step_size=20, gamma=0.7)

# 恢复状态
model_resume.load_state_dict(checkpoint['model_state_dict'])
optimizer_resume.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler_resume.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']

print(f"恢复检查点，从epoch={start_epoch}继续训练")

# 继续训练
for epoch in range(start_epoch, start_epoch + 5):
    scheduler_resume.step()
    print(f'Epoch {epoch + 1}: LR={optimizer_resume.param_groups[0]["lr"]:.6f}')

print("\n总结：学习率衰减策略的选择取决于具体任务和数据集特性！")
```

#### 自适应学习率衰减优化器
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出维度也为1

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.1)

# 准备一些简单的数据
# 输入特征
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 目标值
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空过往梯度
    loss.backward()        # 反向传播，计算当前梯度
    optimizer.step()       # 根据梯度更新网络参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印训练后的模型参数
print(f'Model parameters: {list(model.parameters())}')
```

#### 历史衰减自适应学习率衰减优化器
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出维度也为1

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# 准备一些简单的数据
# 输入特征
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 目标值
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空过往梯度
    loss.backward()        # 反向传播，计算当前梯度
    optimizer.step()       # 根据梯度更新网络参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印训练后的模型参数
print(f'Model parameters: {list(model.parameters())}')
```

#### 自适应矩估计
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出维度也为1

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 准备一些简单的数据
# 输入特征
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 目标值
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空过往梯度
    loss.backward()        # 反向传播，计算当前梯度
    optimizer.step()       # 根据梯度更新网络参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印训练后的模型参数
print(f'Model parameters: {list(model.parameters())}')
```

### practice（实践）
#### LinearDemo（线性回归简单实现）
```python
import torch
import matplotlib.pyplot as plt
from torch import nn, optim  # 模型、损失函数和优化器
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器

"""
基于torch的简单的线性回归
"""

# 构建数据集
X = torch.randn(100, 1)  # 输入
w = torch.tensor([2.5])  # 权重
b = torch.tensor([5.2])  # 偏置
noise = torch.randn(100, 1) * 0.1  # 噪声
y = w * X + b + noise  # 目标
dataset = TensorDataset(X, y)  # 构造数据集对象
dataloader = DataLoader(
    dataset, batch_size=10, shuffle=True
)  # 构造数据加载器对象，batch_size为每次训练的样本数，shuffle为是否打乱数据

# 构造模型
model = nn.Linear(in_features=1, out_features=1)  # 线性回归模型，1个输入，1个输出

# 损失函数和优化器
loss = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 随机梯度下降，学习率0.001

# 模型训练
loss_list = []
for epoch in range(1000):
    total_loss = 0
    train_num = 0
    for x_train, y_train in dataloader:
        # 每次训练一个batch大小的数据
        y_pred = model(x_train)  # 模型预测
        loss_value = loss(y_pred, y_train)  # 计算损失
        total_loss += loss_value.item()
        train_num += len(y_train)
        optimizer.zero_grad()  # 梯度清零
        loss_value.backward()  # 反向传播
        optimizer.step()  # 更新参数
    loss_list.append(total_loss / train_num)

print(model.weight, model.bias)  # 打印权重和偏置
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

#### 手动实现自定义神经网络
```python
import torch
import torch.nn as nn
from torchsummary import summary

class ComNN(nn.Module):
    """
    自定义网络层数
    """
    def __init__(self, input_dim, hidden_dim: list, output_dim):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.out = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    # 测试网络
    # 创建一个具有2个输入节点、隐藏层为[10, 8, 6,4]、3个输出节点的网络
    net = ComNN(input_dim=2, hidden_dim=[10, 8, 6, 4], output_dim=3).to(device="cuda")
    
    # 创建一些测试数据
    test_input = torch.randn(5, 2, device="cuda")  # 5个样本，每个样本有2个特征
    
    # 前向传播
    output = net(test_input)
    
    print("网络结构:")
    print(net)
    print("\n输入数据:")
    print(test_input)
    print("\n输出数据:")
    print(output)
    print("\n输出数据的形状:")
    print(output.shape)
    print("\n输出每行的和（应接近1，因为使用了softmax）:")
    print(torch.sum(output, dim=1))

    print()

    # for name, param in net.named_parameters(): # 查看信息
    #     print(f"参数名称: {name}")
    #     print(f"参数形状: {param.shape}")
    #     print(f"参数值: {param}")
    #     print()
    #
    # print(net.state_dict()) # 查看状态

    summary(net, (2, 2), batch_size=10, device="cuda")
```

#### torch自带实现自定义神经网络
```python
import torch.nn as nn
import torch

"""
使用pytorch自带的网络定义Sequential创建深度神经网络模型
"""

# 构建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1),
)

# 初始化参数
def init_weights(m):
    # 对Linear层进行初始化
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)  # apply会遍历所有子模块并依次调用函数

output = model(torch.randn(10, 3))
print("输出：\n", output)

```