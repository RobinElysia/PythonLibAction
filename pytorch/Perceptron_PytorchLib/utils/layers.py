from pytorch.Perceptron_PytorchLib.utils.ActFunc import sigmoid
from pytorch.Perceptron_PytorchLib.utils.ActFunc import Softmax
from pytorch.Perceptron_PytorchLib.utils.LossFunc import CEE

import numpy as np

class ReLu_:
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

class Sigmoid_:
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
        self.original_shape = None
        # 权重和偏执参数的梯度值保存
        self.dW = None
        self.db = None

    def forward(self, X):
        self.original_shape = X.shape
        self.X = X
        y = X @ self.W + self.b
        return y

    def backward(self, dy):
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)
        dx = np.dot(dy, self.W.T)
        return dx


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
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            # 确保索引不会超出范围
            indices = np.arange(batch_size)
            t_indices = self.t.astype(int)  # 确保标签是整数类型
            # 检查索引是否有效，同时检查两个维度
            valid_indices = (t_indices < self.y.shape[1]) & (indices < self.y.shape[0])
            dx[indices[valid_indices], t_indices[valid_indices]] -= 1
            dx = dx / batch_size
        # 保持与输入数据相同的形状
        return dx  # 不再需要转置