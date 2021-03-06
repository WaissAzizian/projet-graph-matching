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
            emb1, emb2 = self.gnn(g1, mask), self.gnn(g2, mask)
        else:
            g1, g2 = sample[:,0], sample[:, 1]
            mask = torch.ones(g1.size()[:-1]).to(torch.device(g1.device))
            emb1, emb2 = self.gnn(g1, mask), self.gnn(g2, mask)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        return out, mask
