import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 1. 创建简单的训练数据
# 生成一个简单的线性关系：y = 2x + 1 + 噪声
n_samples = 100
x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
y = 2 * x + 1 + torch.randn(n_samples, 1) * 0.5


# 2. 定义简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)


# 3. 创建模型、损失函数和优化器
model = LinearModel()

# 使用带动量的SGD优化器
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 作为对比，也创建一个没有动量的优化器
model_no_momentum = LinearModel()
optimizer_no_momentum = optim.SGD(model_no_momentum.parameters(), lr=0.01, momentum=0)

criterion = nn.MSELoss()  # 均方误差损失


# 4. 训练函数
def train_model(model, optimizer, epochs=200):
    losses = []
    for epoch in range(epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


# 5. 训练两个模型进行比较
print("使用动量的训练过程:")
losses_momentum = train_model(model, optimizer_momentum)

print("\n不使用动量的训练过程:")
losses_no_momentum = train_model(model_no_momentum, optimizer_no_momentum)

# 6. 可视化结果
plt.figure(figsize=(12, 4))

# 损失曲线对比
plt.subplot(1, 2, 1)
plt.plot(losses_momentum, 'b-', label='With Momentum (0.9)')
plt.plot(losses_no_momentum, 'r-', label='Without Momentum')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 拟合结果对比
plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label='Data')

# 使用动量的预测结果
with torch.no_grad():
    y_pred_momentum = model(x)
    y_pred_no_momentum = model_no_momentum(x)

plt.plot(x.numpy(), y_pred_momentum.numpy(), 'b-', linewidth=2, label='With Momentum')
plt.plot(x.numpy(), y_pred_no_momentum.numpy(), 'r-', linewidth=2, label='Without Momentum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting Results')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 查看最终参数
print("\n最终参数比较:")
print("使用动量的模型参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

print("\n不使用动量的模型参数:")
for name, param in model_no_momentum.named_parameters():
    print(f"{name}: {param.data}")

# 8. 动量的更详细示例
print("\n" + "=" * 50)
print("动量的工作原理示例:")


# 创建一个简单的二次函数优化问题
def f(x):
    return x ** 2 + 10 * x + 25


# 手动实现带Momentum的梯度下降
def momentum_optimization(start_x=0.0, lr=0.1, momentum=0.9, n_iter=30):
    x = torch.tensor([start_x], requires_grad=True)
    velocity = torch.zeros(1)
    positions = [x.item()]

    for i in range(n_iter):
        # 计算梯度
        y = f(x)
        y.backward()

        # Momentum更新
        velocity = momentum * velocity + lr * x.grad
        x.data = x.data - velocity

        # 重置梯度
        x.grad.zero_()

        positions.append(x.item())

    return positions


# 比较不同动量值的效果
for momentum in [0, 0.5, 0.9]:
    positions = momentum_optimization(start_x=10.0, momentum=momentum)
    print(f"Momentum={momentum}: 最终位置={positions[-1]:.6f}, 迭代次数={len(positions)}")