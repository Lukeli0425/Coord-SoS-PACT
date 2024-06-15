import torch
from torch import nn


class TV_Regularizer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        
    def forward(self, x, mask):
        dx = (x[1:, :] - x[:-1, :]) * mask[1:, :]
        dy = (x[:, 1:] - x[:, :-1]) * mask[:, 1:]
        return self.weight * (dx.abs().sum() + dy.abs().sum()).mean()
    
    
class L1_Regularizer(nn.Module):
    def __init__(self, weight, mean):
        super().__init__()
        self.weight = weight
        self.mean = mean
        
    def forward(self, x, mask):
        return self.weight * torch.mean((x-self.mean).abs() * mask)
    
    