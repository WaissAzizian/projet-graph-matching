import torch
import torch.nn as nn
import torch.nn.functional as F

#From QAP's original code
def sinkhorn_knopp(A, iterations=1):
    A_size = A.size()
    for it in range(iterations):
        A = A.reshape(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A, dim=-1)
        A = A.reshape(*A_size).permute(0, 2, 1)
        A = A.reshape(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A, dim=-1)
        A = A.reshape(*A_size).permute(0, 2, 1)
    return A

class Siamese(nn.Module):
    def __init__(self, module: nn.Module, sinkhorn_iterations=0):
        super(Siamese, self).__init__()
        self.gnn = module
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, g1, g2):
        emb1, emb2 = self.gnn(g1), self.gnn(g2)
        #here emb.shape=(bs, n_vertices, n_features)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        if self.sinkhorn_iterations > 0:
            out = sinkhorn_knopp(out, iterations=self.sinkhorn_iterations)
        return out
