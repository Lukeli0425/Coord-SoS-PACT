import torch
from torch import nn


class TV_Regularizer(nn.Module):
    def __init__(self, weight=1e-6):
        super().__init__()
        self.weight = weight
        
    def forward(self, x):
        grad = torch.abs(x[:, :-1] - x[:, 1:]).mean() + torch.abs(x[:-1, :] - x[1:, :]).mean()
        return self.weight * grad
    
    
class L1_Regularizer(nn.Module):
    def __init__(self, weight=1e-6):
        super().__init__()
        self.weight = weight
        
    def forward(self, x):
        return self.weight * x.abs().mean()
    
    
class L2_Regularizer(nn.Module):
    def __init__(self, weight=1e-6):
        super().__init__()
        self.weight = weight
        
    def forward(self, x):
        return self.weight * (x**2).mean()
    
    