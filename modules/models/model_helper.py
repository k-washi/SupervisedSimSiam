import math
import torch
from torch import nn

class Swish(nn.Module):
    """
    Swish Activation
    https://arxiv.org/abs/1710.05941v1
    """
    def forward(self, x):
        return x * torch.sigmoid(x)



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
