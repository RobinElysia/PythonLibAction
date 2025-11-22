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
    """

if __name__ == '__main__':
    Easy_tensor()
    print("==========")
    Easy_Tensor()