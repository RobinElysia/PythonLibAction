import numpy as np

# Softmax 函数
def navi_softmax(x):
    # 溢出处理
    x = x-np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 复杂的 Softmax 函数
def complex_softmax(x):
    x = x.T
    # 溢出处理
    x = x-np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0).T

# Softmax 导函数
def softmax_derivative(x):
    return complex_softmax(x) * (1 - complex_softmax(x))

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(complex_softmax(x))