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
        
    def forward(self, x):
        dx = (x[2:, :] - x[:-2, :])
        dy = (x[:, 2:] - x[:, :-2])
        return ((dx**2).sum() + (dy**2).sum()) * self.weight 

class Tenenbaum(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.sobel = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], dim=0).unsqueeze(1)
        self.sobel.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        return (self.sobel(img) ** 2).sum() * self.weight

class Variance(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        
    def forward(self, x):
        return x.var() * self.weight


class Sharpness(nn.Module):
    def __init__(self, function, weight):
        super().__init__()
        if function == 'Brenner':
            self.function = Brenner(weight=weight)
        elif function == 'Tenenbaum':
            self.function = Tenenbaum(weight=weight)
        elif function == 'Variance':
            self.function = Variance(weight=weight)
        else:
            self.function = None
        
    def forward(self, x):
        if self.function is None:
            return 0.0
        return self.function(x)

class L1_Norm(nn.Module):
    def __init__(self, weight, mean=0.0):
        super().__init__()
        self.weight = weight
        self.mean = mean
        
    def forward(self, x):
        return (x-self.mean).abs().mean() * self.weight 
    

if __name__ == "__main__":
    t = Tenenbaum(1.0)
    img = torch.randn(1,256, 256)
    print(img.var())
    reg = t(img)
    