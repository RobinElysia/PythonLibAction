import numpy as np

# CEE
def cross_entropy_error(y, t):
    # y: 预测值, t: 真实值
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 1-hot向量
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size