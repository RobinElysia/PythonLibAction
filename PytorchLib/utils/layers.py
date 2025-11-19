from PytorchLib.utils.ActFunc import sigmoid, Softmax
from PytorchLib.utils.LossFunc import CEE

import numpy as np

class ReLu:
    """
    ReLu激活函数
    前向传播
    反向传播
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dy):
        dy[self.mask] = 0
        dx = dy
        return dx

class Sigmoid:
    """
    Sigmoid激活函数
    前向传播
    反向传播
    """
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = sigmoid.sigmoid(x)
        self.y = y
        return y

    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx

class affine:
    """
    前置仿射
    前向
    反向
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X进行保存
        self.X = None
        # 形状
        self.o_shape = None
        # 权重和偏执参数的梯度值保存
        self.dw = None
        self.db = None

    def forward(self, X):
        self.o_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = X @ self.W + self.b
        return y

    def backward(self, dy):
        dX = dy @ self.W.T
        # 转换
        dX = dX.reshape(*self.o_shape)
        self.dw = self.X.T @ dy
        self.db = np.sum(dy, axis=0)
        return dX

class softmax_with_loss:
    """
    输出层
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, X, t):
        self.t = t
        self.y = Softmax.complex_softmax(X)
        self.loss = CEE.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是独热编码的标签，就直接代入公式计算
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果是标签的索引，就根据索引进行计算
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
            dx = dx / n
        return dx

# 输入层 → Affine → 激活函数 → Affine → 激活函数 → ... → Affine → SoftmaxWithLoss