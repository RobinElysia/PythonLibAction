import numpy as np

# Tanh 函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Tanh 函数求导
def tanh_derivative(x):
    return 1 - tanh(x) ** 2

# Tanh 函数和Sigmoid函数相似，都是非线性函数，但是Tanh函数的输出范围更广，[-1, 1]，Sigmoid函数的输出范围是[0, 1]，
# 但是x的取值更小，因此Tanh函数更适合处理输入值较小的情况，而Sigmoid函数更适合处理输入值较大的情况（在[-3,3]）。
# 是由Sigmoid函数乘2减1得到的，双曲正切函数
# Tanh函数关于关于x轴对称，但同样有梯度消失
# 对应numpy中的 tanh() 函数