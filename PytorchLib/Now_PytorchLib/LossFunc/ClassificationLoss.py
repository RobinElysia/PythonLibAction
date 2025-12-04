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