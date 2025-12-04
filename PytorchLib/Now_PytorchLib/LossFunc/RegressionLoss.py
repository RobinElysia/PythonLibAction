import torch.nn as nn
import torch

def get_loss():
    """
    回归任务常用损失
    :return:
    """
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss_L2 = nn.MSELoss()
    print(loss_L2(tensor, tensor))

    # L1 损失
    loss_L1 = nn.L1Loss()
    print(loss_L1(tensor, tensor))

    # smooth L1 损失
    loss_smoothL1 = nn.SmoothL1Loss()
    print(loss_smoothL1(tensor, tensor))


if __name__ == '__main__':
    get_loss()