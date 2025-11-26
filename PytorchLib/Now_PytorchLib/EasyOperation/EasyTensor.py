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