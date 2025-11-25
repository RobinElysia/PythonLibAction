import torch

def grad_tensor():
    """
    简单的自动梯度计算与反向传播
    :return:
    """
    x = torch.tensor(1)
    y = torch.tensor([[2]])
    w = torch.rand(1,1,requires_grad=True)
    b = torch.rand(1,1,requires_grad=True)
    z = w * x + b
    print(x.is_leaf)
    print(w.is_leaf)
    print(z.is_leaf) # 不是叶子节点
    print(b.is_leaf)
    print(y.is_leaf)
    # 叶子节点计算梯度会自动保存，非叶子节点不会保存梯度
    # 开启return_graph=True，非叶子节点会保存梯度

    loss = torch.nn.MSELoss()
    loss_v = loss(z, y)
    print(loss_v)
    print(loss_v.is_leaf)

    # 反向传播
    loss_v.backward()
    print(w.grad)
    print(b.grad)

def other_detach_grad_tensor():
    """
    节点分支单独运算
    :return:
    """
    x = torch.tensor(1, requires_grad=True)
    y = x.detach()
    print(y.requires_grad)
    print(x.requires_grad)
    print(id(x)) # id不同
    print(id(y))
    print(x.untyped_storage()) # 数据共享
    print(y.untyped_storage())

def data_vs_detach():
    """
    张量直接data和detach的区别
    :return:
    """
    # 当你试图使用data对节点进行修改时，妄图对修改后的节点做梯度，会成功，并且会直接影响梯度运算，这不是我们希望的
    # 而使用detach()时，张量的数据会进行复制，不会对原张量进行修改，但是对梯度运算没有影响


if __name__ == "__main__":
    grad_tensor()
    print("------------------")
    other_detach_grad_tensor()