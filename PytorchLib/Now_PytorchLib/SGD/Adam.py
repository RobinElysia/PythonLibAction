import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出维度也为1

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 准备一些简单的数据
# 输入特征
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 目标值
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空过往梯度
    loss.backward()        # 反向传播，计算当前梯度
    optimizer.step()       # 根据梯度更新网络参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印训练后的模型参数
print(f'Model parameters: {list(model.parameters())}')