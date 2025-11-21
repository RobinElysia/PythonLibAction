import numpy as np

# 随机梯度下降
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        :param params: 参数字典
        :param grads: 梯度字典
        :return:
        """
        # 循环更新参数
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 梯度下降算法容易陷入局部最优解，因此，为了避免陷入局部最优解，
# 我们可以使用随机梯度下降算法。随机梯度下降算法的思路是每次只更新一个样本的梯度，而不是更新所有样本的梯度。
# 动量法，随机梯度下降算法的一个改进版本，可以避免陷入局部最优解。

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # 对v进行初始化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 参数更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key] # v = momentum * v - lr * grads
            params[key] += self.v[key] # params = params + v

# 学习率衰减
    # 等间隔衰减：步长衰减，按照训练周期进行某比例衰减
    # 指定间隔衰减：步长衰减，按照训练周期进行某系数衰减
    # 指数衰减：步长衰减，按照训练周期进行指数为底进行衰减

    # 有点死板
# 看看AdaGrad
    # 自适应梯度
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        # 对h进行初始化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 梯度更新
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 哈达马积
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
# 动量法的初始化比AdaGrad好，因为随着v参数的更新，v的变化率会趋于稳定
# 而AdaGrad的h不会，它的初始化参数是哈达马积，不会像动量法的v参数更新那样趋于稳定
# 于是为了引入，我们又从AdaGrad到了RMSProp
    # 均方根传播，对于过早的历史信息进行遗忘，权重变小，新的信息权重变大（使用指数移动加权平均，EMA）
    # 多了一个α超参数
class RMSProp:
    def __init__(self, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.h = None

    def update(self, params, grads):
        # 对h进行初始化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 梯度更新
        for key in params.keys():
            self.h[key] *= self.gamma
            self.h[key] += (1 - self.gamma) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# Adam它把两条非常成功的改进 SGD 的路线直接“拼”在了一起：
    # AdaGrad——给每个参数配一个“历史梯度平方和”做分母，实现学习率自适应。
    # RMSProp——把 AdaGrad 的“历史平方和”改成“指数移动平均”，解决后期学习率过早降到 0 的问题。
    # Adam 在这两条算法的基础上再叠加一层“动量”（Momentum，即对梯度本身也做指数移动平均），于是同时拥有了
        # 动量法的一阶矩估计（mean）
        # RMSProp 的二阶矩估计（uncentered variance）
    # 自适应矩估计
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        # 对m和v进行初始化
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # 迭代次数
        self.iter += 1
        # 更改当前轮次的学习率
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        # 更新参数
        for key in params.keys():
            # # m = beta2 * m + (1 - beta2) * grads**2
            # self.m[key] += self.beta2 * self.m[key] + (1 - self.beta2) * (grads[key] ** 2)
            # # v = beta1 * v + (1 - beta1) * grads
            # self.v[key] += self.beta1 * self.v[key] + (1 - self.beta1) * grads[key]
            # 上述移向得到，简化运算效率
            self.v[key] += (1 - self.beta1) * (grads[key] - self.v[key])
            self.m[key] += (1 - self.beta2) * (grads[key] - self.m[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.m[key]) + 1e-7)