import numpy as np

# ReLU函数
def ReLU(x):
    return np.maximum(0, x)

# ReLU函数求导，就是0或1
def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

# 1. 为了解决梯度消失问题，我们有了ReLU函数，它可以作用于隐层作为激活函数，也可以作为输出层作为激活函数。
# 2. 仅有一个不可导点，但是无伤大雅
# 3. 没有激活就是平的，激活了就是一条直线
# 4. 当x小于等于0时，ReLU函数输出0，当x大于0时，ReLU函数输出x
# 5. 它的导函数就是阶跃函数
# 6. 不存在梯度消失，且当x小于零时，部分神经元是活跃的，可以减少计算量，但是一直为小于等于0的x，会导致神经元死亡问题

# leaky ReLU函数
def leaky_ReLU(alpha,x):
    return np.maximum(alpha * x, x)

# 反向传播计算
def leaky_ReLU_derivative(alpha,x):
    return np.where(x > 0, 1, alpha)

# 1. ReLU 激活函数的改进，当x小于0时，输出$\alpha$x，当x大于0时，输出x
# 2. 避免神经元死亡