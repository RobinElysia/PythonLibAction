import numpy as np

from PytorchLib.utils.ActFunc import Softmax # 激活函数
from PytorchLib.utils.LossFunc import CEE # 损失函数
from PytorchLib.utils.gd import gradient


# 定义一个简单神经网络类（只有输入层输出层）
class SimpleNN:
    def __init__(self):
        self.W = np.random.randn(2,3) # 输入层input_size个神经元，输出层hidden_size个神经元

    # 前向传播
    def forward(self, X):
        """
        :param X: 输入
        :return: 运算soft的y
        """
        a = X @ self.W # 矩阵乘法
        y = Softmax.complex_softmax(a)
        return y

    # 损失
    def loss(self, X, t):
        """
        :param X: 输入
        :param t: 真实值
        :return:
        """
        y = self.forward(X)
        return CEE.cross_entropy_error(y, t)

# 主流程
if __name__ == "__main__":
    X = np.array([0.6, 0.9])
    t = np.array([0, 1, 0])
    network = SimpleNN()
    # 损失
    f = lambda W: network.loss(X, t)
    # 梯度
    W_grad = gradient.numerical_matrix_gradient(f, network.W)
    print(W_grad)