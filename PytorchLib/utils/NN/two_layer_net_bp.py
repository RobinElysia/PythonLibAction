"""
手写bp反向传播算法的单隐层网络
"""

import numpy as np

from PytorchLib.utils.ActFunc import sigmoid, Softmax # 激活函数
from PytorchLib.utils.LossFunc import CEE # 交叉熵损失
from PytorchLib.utils.gd import gradient # 梯度计算
from PytorchLib.utils.layers import ReLu_, Sigmoid_, affine, softmax_with_loss # bp
from collections import OrderedDict # 定义一个层顺序和相关的名称，有序字典

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param weight_init_std: 权重标准差，改为较小值以避免梯度爆炸
        """
        self.params = {} # 参数变量
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 定义层顺序
        self.layers = OrderedDict()
        self.layers['Affine1'] = affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu_()
        self.layers['Affine2'] = affine(self.params['W2'], self.params['b2'])
        # 单独定义输出层
        self.lastLayer = softmax_with_loss()

    def forward(self, X):
        """
        前向传播
        :param X: 输入
        :return: 输出
        """
        # 对于每一层调用forward
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def loss(self, x, t):
        """
        计算损失
        :param x: 输入
        :param t: 真实值
        :return: 损失
        """
        # 先进行前向传播
        y = self.forward(x)
        # 再计算损失
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y_pred = self.forward(x) # 计算y值
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        y = np.argmax(y_pred, axis=1) # 获取最大概率的索引
        accuracy = np.sum(y == t) / float(x.shape[0]) # 计算准确率
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

    # bp优化后的numerical_gradient
    def gradient_bp(self, x, t):
        # 前向传播
        self.forward(x)
        # 计算损失（不需要再次前向传播）
        self.lastLayer.forward(self.layers['Affine2'].X @ self.params['W2'] + self.params['b2'], t)
        # 反向传播
        dy = self.lastLayer.backward(1)
        # 注意：需要按相反顺序进行反向传播
        layers = list(self.layers.values())
        for i in range(len(layers) - 1, -1, -1): # 翻转
            dy = layers[i].backward(dy)
        # 提取各层参数梯度
        grads = {}
        grads['W1'] , grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'] , grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads