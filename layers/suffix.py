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
        # out: N x m x d x basis
        diag_part = torch.diagonal(x, dim1=2, dim2=3)
        sum_diag_part = torch.sum(diag_part, 2).unsqueeze(-1)
        sum_of_rows = torch.sum(x, 3)
        sum_of_cols = torch.sum(x, 2)
        sum_all = torch.sum(x, (2,3)).unsqueeze(-1)

        op1 = diag_part
        op2 = sum_diag_part.expand_as(op1)
        op3 = sum_of_rows
        op4 = sum_of_cols
        op5 = sum_all.expand_as(op1)

        m = op1.size(-1)
        op2 = op2/m
        op3 = op3/m
        op4 = op4/m
        op5 = op5/(m**2)

        return torch.stack([op1, op2, op3, op4, op5]).permute(1, 3, 2, 0)

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
        x[:, :, indices, indices] = float('-inf') * torch.ones(m)
        max_off = torch.max(torch.max(x, -1)[0], -1)[0]
        #return torch.mean(x, (2, 3))
        return torch.cat((max_diag, max_off), -1)
