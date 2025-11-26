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