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