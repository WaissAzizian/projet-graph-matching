import torch
import torch.nn as nn

class.AverageSuffix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # in: N x d x m x m
        # out: N x d x m
        return torch.mean(x, -1)
