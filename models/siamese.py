import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self, module: nn.Module):
        super(Siamese, self).__init__()
        self.gnn = module

    def forward(self, g1, g2):
        emb1, emb2 = self.gnn(g1), self.gnn(g2)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        return out
