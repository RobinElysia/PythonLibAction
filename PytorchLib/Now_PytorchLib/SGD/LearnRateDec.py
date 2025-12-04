import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, MultiStepLR, LambdaLR
)

"""
StepLR：固定步长衰减，每隔固定epoch衰减一次
ExponentialLR：指数衰减，每个epoch都衰减
CosineAnnealingLR：余弦退火，学习率按余弦函数变化
ReduceLROnPlateau：基于验证指标衰减，当指标不再改善时衰减
MultiStepLR：多步长衰减，在指定epoch处衰减
LambdaLR：自定义衰减函数，灵活度高
CosineAnnealingWarmRestarts：余弦退火热重启，周期性重置学习率
"""

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 1. 创建训练数据 - 一个更复杂的非线性问题
n_samples = 200
x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
y = torch.sin(x) + 0.3 * torch.randn(n_samples, 1)  # 正弦函数加噪声


# 2. 定义更复杂的神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# 3. 创建多个相同初始状态的模型，用于比较不同的学习率调度器
def create_model_and_optimizer():
    model = NeuralNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


# 4. 训练函数，记录学习率和损失
def train_with_scheduler(model, optimizer, scheduler, epochs=200):
    losses = []
    learning_rates = []

    for epoch in range(epochs):
        # 前向传播
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        if scheduler is not None:
            # ReduceLROnPlateau需要传入验证指标
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        # 记录
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 40 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return losses, learning_rates


# 5. 定义不同的学习率调度策略
print("=" * 60)
print("不同学习率衰减策略比较")
print("=" * 60)

# 策略1: 固定学习率（无调度器）
print("\n1. 固定学习率 (无调度器)")
model1, optimizer1 = create_model_and_optimizer()
losses1, lrs1 = train_with_scheduler(model1, optimizer1, None)

# 策略2: 步长衰减
print("\n2. 步长衰减 (每50个epoch衰减为原来的0.5倍)")
model2, optimizer2 = create_model_and_optimizer()
scheduler2 = StepLR(optimizer2, step_size=50, gamma=0.5)
losses2, lrs2 = train_with_scheduler(model2, optimizer2, scheduler2)

# 策略3: 指数衰减
print("\n3. 指数衰减 (每个epoch衰减为原来的0.995倍)")
model3, optimizer3 = create_model_and_optimizer()
scheduler3 = ExponentialLR(optimizer3, gamma=0.995)
losses3, lrs3 = train_with_scheduler(model3, optimizer3, scheduler3)

# 策略4: 余弦退火
print("\n4. 余弦退火 (从初始学习率退火到0)")
model4, optimizer4 = create_model_and_optimizer()
scheduler4 = CosineAnnealingLR(optimizer4, T_max=200, eta_min=0)
losses4, lrs4 = train_with_scheduler(model4, optimizer4, scheduler4)

# 策略5: 基于验证损失的衰减
print("\n5. 基于验证损失的衰减 (当损失不再下降时衰减)")
model5, optimizer5 = create_model_and_optimizer()
scheduler5 = ReduceLROnPlateau(optimizer5, mode='min', factor=0.5, patience=10, verbose=False)
losses5, lrs5 = train_with_scheduler(model5, optimizer5, scheduler5)

# 策略6: 多步长衰减
print("\n6. 多步长衰减 (在epoch 30, 80, 120时衰减)")
model6, optimizer6 = create_model_and_optimizer()
scheduler6 = MultiStepLR(optimizer6, milestones=[30, 80, 120], gamma=0.5)
losses6, lrs6 = train_with_scheduler(model6, optimizer6, scheduler6)

# 策略7: 自定义Lambda衰减
print("\n7. 自定义Lambda衰减 (学习率 = 初始学习率 * (1 + 10*epoch/总epochs)^-0.75)")
model7, optimizer7 = create_model_and_optimizer()
lambda_func = lambda epoch: (1 + 10 * epoch / 200) ** -0.75
scheduler7 = LambdaLR(optimizer7, lr_lambda=lambda_func)
losses7, lrs7 = train_with_scheduler(model7, optimizer7, scheduler7)

# 6. 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 学习率变化曲线
ax1 = axes[0, 0]
epochs = range(200)
ax1.plot(epochs, lrs1, label='固定学习率')
ax1.plot(epochs, lrs2, label='步长衰减')
ax1.plot(epochs, lrs3, label='指数衰减')
ax1.plot(epochs, lrs4, label='余弦退火')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate')
ax1.set_title('学习率变化曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 损失变化曲线
ax2 = axes[0, 1]
ax2.plot(epochs, losses1, label='固定学习率')
ax2.plot(epochs, losses2, label='步长衰减')
ax2.plot(epochs, losses3, label='指数衰减')
ax2.plot(epochs, losses4, label='余弦退火')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('损失变化曲线')
ax2.set_yscale('log')  # 对数坐标更清晰
ax2.legend()
ax2.grid(True, alpha=0.3)

# 其他调度策略的学习率变化
ax3 = axes[1, 0]
ax3.plot(epochs, lrs5, label='基于验证损失衰减')
ax3.plot(epochs, lrs6, label='多步长衰减')
ax3.plot(epochs, lrs7, label='自定义Lambda衰减')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('其他调度策略的学习率变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 拟合结果对比
ax4 = axes[1, 1]
ax4.scatter(x.numpy(), y.numpy(), alpha=0.3, label='原始数据', s=10)

with torch.no_grad():
    y_pred1 = model1(x)
    y_pred4 = model4(x)
    y_pred7 = model7(x)

ax4.plot(x.numpy(), y_pred1.numpy(), 'r-', linewidth=2, label='固定学习率')
ax4.plot(x.numpy(), y_pred4.numpy(), 'g-', linewidth=2, label='余弦退火')
ax4.plot(x.numpy(), y_pred7.numpy(), 'b-', linewidth=2, label='Lambda衰减')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('不同调度策略的拟合结果')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 详细展示一种调度器的使用（StepLR）
print("\n" + "=" * 60)
print("StepLR调度器详细示例")
print("=" * 60)

# 重新创建一个模型
model_demo, optimizer_demo = create_model_and_optimizer()

# 创建StepLR调度器：每25个epoch学习率乘以0.8
scheduler_demo = StepLR(optimizer_demo, step_size=25, gamma=0.8)

print("初始学习率:", optimizer_demo.param_groups[0]['lr'])

# 模拟训练过程
for epoch in range(100):
    # 模拟训练步骤
    optimizer_demo.step()  # 实际训练中这里会有loss.backward()

    # 更新学习率
    scheduler_demo.step()

    # 每10个epoch打印一次学习率
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: 学习率 = {optimizer_demo.param_groups[0]["lr"]:.6f}')

# 8. 结合动量的学习率衰减示例
print("\n" + "=" * 60)
print("结合动量Momentum和学习率衰减的综合示例")
print("=" * 60)

# 创建模型
model_combined = NeuralNet()

# 使用带动量的SGD优化器
optimizer_combined = optim.SGD(
    model_combined.parameters(),
    lr=0.1,
    momentum=0.9,  # 添加动量
    weight_decay=1e-4  # L2正则化
)

# 使用余弦退火结合热重启的调度器
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler_combined = CosineAnnealingWarmRestarts(
    optimizer_combined,
    T_0=50,  # 第一个周期的长度
    T_mult=2,  # 每个周期长度加倍
    eta_min=0.001  # 最小学习率
)

print("训练过程:")
for epoch in range(150):
    # 模拟训练步骤
    loss = torch.randn(1).item() * 0.1  # 模拟损失

    # 优化步骤
    optimizer_combined.zero_grad()
    # loss.backward()  # 实际训练中会有反向传播
    optimizer_combined.step()

    # 更新学习率
    scheduler_combined.step()

    if (epoch + 1) % 25 == 0:
        print(f'Epoch {epoch + 1}: LR={optimizer_combined.param_groups[0]["lr"]:.6f}, '
              f'Momentum={optimizer_combined.param_groups[0]["momentum"]}')

# 9. 学习率调度器的状态保存与恢复示例
print("\n" + "=" * 60)
print("学习率调度器状态保存与恢复示例")
print("=" * 60)

model_checkpoint = NeuralNet()
optimizer_checkpoint = optim.SGD(model_checkpoint.parameters(), lr=0.1)
scheduler_checkpoint = StepLR(optimizer_checkpoint, step_size=20, gamma=0.7)

# 训练几个epoch
for epoch in range(10):
    scheduler_checkpoint.step()
    print(f'Epoch {epoch + 1}: LR={optimizer_checkpoint.param_groups[0]["lr"]:.6f}')

# 保存调度器状态
checkpoint = {
    'model_state_dict': model_checkpoint.state_dict(),
    'optimizer_state_dict': optimizer_checkpoint.state_dict(),
    'scheduler_state_dict': scheduler_checkpoint.state_dict(),
    'epoch': 10
}

print(f"\n保存检查点，当前epoch={checkpoint["epoch"]}, LR={optimizer_checkpoint.param_groups[0]['lr']:.6f}")

# 创建新模型和调度器
model_resume = NeuralNet()
optimizer_resume = optim.SGD(model_resume.parameters(), lr=0.1)
scheduler_resume = StepLR(optimizer_resume, step_size=20, gamma=0.7)

# 恢复状态
model_resume.load_state_dict(checkpoint['model_state_dict'])
optimizer_resume.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler_resume.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']

print(f"恢复检查点，从epoch={start_epoch}继续训练")

# 继续训练
for epoch in range(start_epoch, start_epoch + 5):
    scheduler_resume.step()
    print(f'Epoch {epoch + 1}: LR={optimizer_resume.param_groups[0]["lr"]:.6f}')

print("\n总结：学习率衰减策略的选择取决于具体任务和数据集特性！")