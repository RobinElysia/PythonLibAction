import numpy as np
from PytorchLib.utils.ActFunc import Softmax, sigmoid
from PytorchLib.utils.LossFunc import CEE
from PytorchLib.utils.gd import gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param weight_init_std: 权重标准差
        """
        self.params = {} # 参数变量
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X):
        """
        前向传播
        :param X: 输入
        :return: 输出
        """
        # 获取参数
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # 开始前向传播
        a1 = X @ W1 + b1 # 第一层矩阵乘法
        z1 = sigmoid.sigmoid(a1) # 二层激活函数
        a2 = z1 @ W2 + b2 # 第二层矩阵乘法
        y = Softmax.complex_softmax(a2) # 三层输出计算
        return y

    def loss(self, x, t):
        """
        计算损失
        :param x: 输入
        :param t: 真实值
        :return: 损失
        """
        y = self.forward(x) # 计算预测y值
        return CEE.cross_entropy_error(y, t) # 计算预测y与真实t的交叉熵损失

    def accuracy(self, x, t):
        y_proba = self.forward(x) # 计算y值
        y = np.argmax(y_proba, axis=1) # 获取最大概率的索引
        accuracy = np.sum(y == t) / len(t) # 计算准确率
        return accuracy

    def numerical_gradient(self, x, t):
        # 定义目标函数
        loos_ = lambda w: self.loss(x, t) # 损失函数实际上就是上述，但是这里需要构建一下
        # 对每个参数使用微分计算梯度
        g = {}
        g['W1'] = gradient.numerical_matrix_gradient(loos_, self.params['W1']) # 调用写好的梯度函数
        g['b1'] = gradient.numerical_matrix_gradient(loos_, self.params['b1'])
        g['W2'] = gradient.numerical_matrix_gradient(loos_, self.params['W2'])
        g['b2'] = gradient.numerical_matrix_gradient(loos_, self.params['b2'])
        return g # 返回梯度