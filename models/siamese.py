import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self, module: nn.Module):
        super(Siamese, self).__init__()
        self.gnn = module

    def forward(self, sample):
        if not isinstance(sample, torch.Tensor):
            g1, g2 = sample.batch[:,0], sample.batch[:,1]
            mask = sample.mask
            emb1, emb2 = self.gnn(g1, mask=mask), self.gnn(g2, mask=mask)
        else:
            g1, g2 = sample[:,0], sample[:, 1]
            emb1, emb2 = self.gnn(g1), self.gnn(g2)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        return out
