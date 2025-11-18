import numpy as np
from PytorchLib.utils.gradent import gradient

# 梯度下降函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x # 初始值
    x_history = [] # 存储每一步的参数
    for i in range(step_num): # 迭代次数
        x_history.append(x.copy()) # 存储参数
        grad = gradient.numerical_gradient(f, x) # 计算梯度
        x -= lr * grad # 更新参数
    return x, np.array(x_history) # 返回参数和参数历史