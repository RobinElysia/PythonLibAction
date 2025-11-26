import torch
import torch.nn as nn
from torchsummary import summary

class ComNN(nn.Module):
    """
    自定义网络层数
    """
    def __init__(self, input_dim, hidden_dim: list, output_dim):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.out = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    # 测试网络
    # 创建一个具有2个输入节点、隐藏层为[10, 8]、3个输出节点的网络
    net = ComNN(input_dim=2, hidden_dim=[10, 8], output_dim=3)
    
    # 创建一些测试数据
    test_input = torch.randn(5, 2)  # 5个样本，每个样本有2个特征
    
    # 前向传播
    output = net(test_input)
    
    print("网络结构:")
    print(net)
    print("\n输入数据:")
    print(test_input)
    print("\n输出数据:")
    print(output)
    print("\n输出数据的形状:")
    print(output.shape)
    print("\n输出每行的和（应接近1，因为使用了softmax）:")
    print(torch.sum(output, dim=1))

    print()

    # for name, param in net.named_parameters(): # 查看信息
    #     print(f"参数名称: {name}")
    #     print(f"参数形状: {param.shape}")
    #     print(f"参数值: {param}")
    #     print()
    #
    # print(net.state_dict()) # 查看状态

    summary(net, (2, 2), batch_size=10, device="cpu")