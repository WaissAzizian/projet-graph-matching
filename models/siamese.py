import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lap
import geomloss

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

def predict_lap(x, y):
    cost_matrices = torch.cdist(x, y, p=2).data.cpu().numpy()
    permutations = np.empty((x.size(0), x.size(1)), dtype=int)
    for index, cost_matrix in enumerate(cost_matrices):
        permutation, _ = lap.lapjv(cost_matrix, return_cost=False)
        permutations[index] = permutation
    return permutations

def accuracy_lap(pred):
    (x, y) = pred
    permutations = predict_lap(x, y)
    m = permutations.shape[1]
    identity = np.arange(m)
    acc = np.mean(permutations == identity[np.newaxis, :])
    return acc

otloss = geomloss.SamplesLoss()
def compute_otloss(pred, _):
    x = pred[0].contiguous()
    y = pred[1].contiguous()
    loss = otloss(x, y)
    if not (loss >= 0).all():
        batch_index = (loss < 0).nonzero()[0][0]
        with open('wrong_data.pkl', 'wb') as f:
            torch.save(x, f)
            torch.save(y, f)
        print("Wrong data:", loss[batch_index], otloss(x[batch_index], y[batch_index]).item())
    assert (loss >= 0).all()
    return loss.mean()

class Siamese(nn.Module):
    def __init__(self, module: nn.Module, sinkhorn_iterations=0, wasserstein_iterations=0, otloss=False):
        super(Siamese, self).__init__()
        self.gnn = module
        self.sinkhorn_iterations = sinkhorn_iterations
        self.otloss = otloss

    def forward(self, g1, g2):
        emb1, emb2 = self.gnn(g1), self.gnn(g2)
        #here emb.shape=(bs, n_vertices, n_features)
        if self.otloss:
            return emb1, emb2
        else:
            out = torch.bmm(emb1, emb2.permute(0, 2, 1))
            out = sinkhorn_knopp(out, iterations=self.sinkhorn_iterations)
        return out
