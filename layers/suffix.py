import torch
import torch.nn as nn

class AverageSuffix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # in: N x d x m x m
        # out: N x d x m
        return torch.mean(x, -1)

class Features_2_to_1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # in: N x d x m x m
        # out: N x (d * basis) x m
        N = x.size(0)
        m = x.size(-1)
        diag_part = torch.diagonal(x, dim1=2, dim2=3)
        max_diag_part = torch.max(diag_part, 2)[0].unsqueeze(-1)
        max_of_rows = torch.max(x, 3)[0]
        max_of_cols = torch.max(x, 2)[0]
        max_all = torch.max(torch.max(x, 2)[0], 2)[0].unsqueeze(-1)

        op1 = diag_part
        op2 = max_diag_part.expand_as(op1)
        op3 = max_of_rows
        op4 = max_of_cols
        op5 = max_all.expand_as(op1)

        return torch.stack([op1, op2, op3, op4, op5]).permute(1, 0, 2, 3).reshape(N, -1, m)

class EquivariantSuffix(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features*5, output_features)
        self.extractor = Features_2_to_1()

    def forward(self, x):
        #in: N x d x m x m
        #out: N x S x m
        #d = input_features
        #s= output_features
        x = self.extractor(x)
        N = x.size(0)
        m = x.size(1)
        x = x.reshape(x.size(0)*x.size(1), x.size(2)*x.size(3))
        x = self.linear(x)
        x = x.view(N, m, -1).permute(0, 2, 1)
        return x

class AverageSuffixClassification(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        # in: N x d x m x m
        # out: N x 2d
        m = x.size(-1)
        sum_diag = torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), -1)
        sum_all = torch.sum(x, (2, 3))
        sum_off = sum_all - sum_diag
        mean_diag = sum_diag/m
        mean_off = sum_off/(m**2 - m)
        #return torch.mean(x, (2, 3))
        return torch.cat((mean_diag, mean_off), -1)

class MaxSuffixClassification(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        # in: N x d x m x m
        # out: N x 2d
        m = x.size(-1)
        max_diag = torch.max(torch.diagonal(x, dim1=-2, dim2=-1), -1)[0]
        indices = torch.arange(m)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        x[:, :, indices, indices] = float('-inf') * torch.ones(m).to(device)
        max_off = torch.max(torch.max(x, -1)[0], -1)[0]
        #return torch.mean(x, (2, 3))
        return torch.cat((max_diag, max_off), -1)
