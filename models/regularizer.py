import torch
from torch import nn


class Total_Variation(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        
    def forward(self, x, mask):
        dx = (x[1:, :] - x[:-1, :]) * mask[1:, :]
        dy = (x[:, 1:] - x[:, :-1]) * mask[:, 1:]
        return (dx.abs().sum() + dy.abs().sum()).mean() * self.weight 


class Brenner(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        
    def forward(self, x, mask):
        dx = (x[1:, :] - x[:-1, :]) * mask[1:, :]
        dy = (x[:, 1:] - x[:, :-1]) * mask[:, 1:]
        return ((dx**2).sum() + (dx**2).sum()).mean() * self.weight 
    
class L1_Norm(nn.Module):
    def __init__(self, weight, mean=0.0):
        super().__init__()
        self.weight = weight
        self.mean = mean
        
    def forward(self, x):
        return (x-self.mean).abs().mean() * self.weight 
    
    