import numpy as np

# Softmax 函数
def navi_softmax(x):
    # 溢出处理
    x = x-np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 复杂的 Softmax 函数
def complex_softmax(x):
    # 溢出处理
    x = x-np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Softmax 导函数
def softmax_derivative(x):
    s = complex_softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(complex_softmax(x))