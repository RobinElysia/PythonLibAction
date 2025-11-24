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