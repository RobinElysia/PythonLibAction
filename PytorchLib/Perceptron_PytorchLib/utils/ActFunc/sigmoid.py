import numpy as np

# 阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# 多元阶跃函数
def vector_step_function(x):
    return np.array(x > 0, dtype=int) # 创建一个布尔数组，并转换为整数

# 阶跃函数数学性质不好

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid函数求导
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# sigmoid函数数学性质比较阶跃函数好，可导连续，可以表示概率值（全为正数）
# 但是它的实际有效值数值域范围有限（在[-6,6]），输入值过小或过大会无法计算或者直接数值消失，导致无法计算
# 可能影响梯度计算，导致梯度消失