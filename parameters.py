import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.Tensor()
        # self.y = nn.parameter.Parameter()
        # self.register_buffer('z', torch.Tensor())
net = Net()
