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