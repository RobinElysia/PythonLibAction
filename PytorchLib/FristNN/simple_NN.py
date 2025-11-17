import numpy as np
from utils.ActFunc import otherActFunc, sigmoid


# 初始化网络
def init_network():
    network = {}
    # 第一层
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 输入层3个神经元，隐藏层2个神经元
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 第二层
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 隐藏层2个神经元，输出层2个神经元
    network['b2'] = np.array([0.1, 0.2])
    # 第三层
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 隐藏层2个神经元，输出层2个神经元
    network['b3'] = np.array([0.1, 0.2])
    return network

# 前向传播
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = otherActFunc.identity(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = otherActFunc.identity(a3)
    y = otherActFunc.identity(y)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)