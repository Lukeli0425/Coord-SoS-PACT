import torch
from torch import nn

from models.SIREN import SIREN
from utils.utils_torch import get_mgrid


class SoS(nn.Module):
    """SoS parameterization module."""
    def __init__(self, mode, mask, v0, mean, std):
        super().__init__()
        self.mode = mode
        self.mask = mask
        self.v0 = v0
        self.mean, self.std = mean, std
        
        if mode == 'None':
            self.SoS = torch.normal(0,1,[(self.mask>0.5).sum(), 1], requires_grad=True).cuda()# * v0
            self.SoS = nn.Parameter(self.SoS, requires_grad=True)
        elif mode == 'SIREN':
            self.mgrid = get_mgrid(self.mask.shape, range=(-1, 1)).cuda()
            self.mgrid = self.mgrid[self.mask.view(-1)>0]
            self.siren = SIREN(in_features=2, out_features=1, hidden_features=128, num_hidden_layers=1, activation_fn='sin')
            self.siren.cuda()
        else:
            raise ValueError('Invalid mode. Choose from [None, SIREN]')
        
    def forward(self):
        SoS = (torch.ones_like(self.mask, requires_grad=True) * self.v0).view(-1,1)
        if self.mode == 'None':
            SoS[self.mask.view(-1)>0] = self.SoS.double() * self.std + self.mean
        elif self.mode == 'SIREN':
            output, _ = self.siren(self.mgrid)
            SoS[self.mask.view(-1)>0] = output.double() * self.std + self.mean
        return SoS.view(self.mask.shape)
        