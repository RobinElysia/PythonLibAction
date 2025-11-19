from PytorchLib.utils.ActFunc import sigmoid
import numpy as np

class ReLu:
    """
    ReLu层的细分
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
    Sigmoid层的细分
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
    仿射
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