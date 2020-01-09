import torch
import torch.nn as nn
import torch.nn.functional as F

#From QAP's original code
def sinkhorn_knopp_old(A, iterations=1):
    A_size = A.size()
    for it in range(iterations):
        A = A.reshape(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A, dim=-1)
        A = A.reshape(*A_size).permute(0, 2, 1)
        A = A.reshape(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A, dim=-1)
        A = A.reshape(*A_size).permute(0, 2, 1)
    return A

def sinkhorn_knopp(A, iterations=1, epsilon=1e-3):
    A_size = A.size()
    if iterations > 0:
        A = F.relu(A)
    for it in range(iterations):
        A = A/(A.sum(2).unsqueeze(-1)+epsilon)
        A = A/(A.sum(1).unsqueeze(1)+epsilon)
    return A

def sinkhorn_wasserstein(X, Y, iterations=1, epsilon=1):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    m = X.size(1)
    N = X.size(0)
    u = torch.ones(N, m).to(device)
    v = torch.ones(N, m).to(device)
    a = torch.ones(N, m).to(device)/m
    b = torch.ones(N, m).to(device)/m
    K = torch.exp(-torch.cdist(X, Y)**2/epsilon)
    assert not torch.isnan(K).any()
    for it in range(iterations):
        u = a / (K * v[:, None,:]).sum(2)
        v = b / (K * u[:, :,None]).sum(1)
    P = torch.einsum('bi, bij, bj -> bij', u, K, v)
    assert not torch.isnan(P).any()
    return P

class Siamese(nn.Module):
    def __init__(self, module: nn.Module, sinkhorn_iterations=0, wasserstein_iterations=0):
        super(Siamese, self).__init__()
        self.gnn = module
        self.sinkhorn_iterations = sinkhorn_iterations
        self.wasserstein_iterations = wasserstein_iterations

    def forward(self, g1, g2):
        emb1, emb2 = self.gnn(g1), self.gnn(g2)
        #here emb.shape=(bs, n_vertices, n_features)
        if self.wasserstein_iterations > 0:
            assert self.sinkhorn_iterations == 0
            out = sinkhorn_wasserstein(emb1.contiguous(), emb2.contiguous())
        else:
            out = torch.bmm(emb1, emb2.permute(0, 2, 1))
            out = sinkhorn_knopp(out, iterations=self.sinkhorn_iterations)
        return out
