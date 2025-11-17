import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 情况1: 标准欧氏距离 (W = I)
W1 = np.diag([1, 1])  # 单位矩阵
print("标准欧氏距离的等距离线是圆形")

# 情况2: 加权欧氏距离 (W ≠ I)
W2 = np.diag([2, 0.5])  # x方向权重2，y方向权重0.5
print("加权欧氏距离的等距离线是椭圆")


# 绘制等距离线
def plot_distance_contour(ax, W, title):
    # 生成网格点
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # 计算每个点的距离值 (以原点为参考)
    distances = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[i, j], Y[i, j]])
            distances[i, j] = point.T @ W @ point

    # 绘制等值线
    contours = ax.contour(X, Y, distances, levels=[0.5, 1, 2, 3], colors='blue', alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8)

    # 绘制坐标轴
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


plot_distance_contour(ax1, W1, '标准欧氏距离: W = [[1,0],[0,1]]\n等距离线为圆形')
plot_distance_contour(ax2, W2, '加权欧氏距离: W = [[2,0],[0,0.5]]\n等距离线为椭圆')

plt.tight_layout()
plt.show()