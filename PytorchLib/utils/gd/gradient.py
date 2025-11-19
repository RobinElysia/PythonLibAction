import numpy as np

# 数值微分
def diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h) # 中心差分，而不是除以h

# 向量梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        # 保存当前的值
        tmp_val = x[idx]
        # 加减h计算中心差分
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # 恢复原值
        x[idx] = tmp_val
    return grad

# 矩阵梯度
def numerical_matrix_gradient(f, X):
    # 判断是否为向量
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else: # 矩阵
        grad = np.zeros_like(X)
        # 遍历每一行元素进行梯度
        for i ,x in enumerate(X):
            grad[i] = numerical_gradient(f, x)
        return grad