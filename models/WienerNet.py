import numpy as np
import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift
import torch.nn as nn
import torch.nn.functional as F
from models.ResUNet import ResUNet
from models.PACT import PSF_PACT
from utils.utils_torch import conv_fft_batch, psf_to_otf



class Wiener(nn.Module):
    def __init__(self):
        super(Wiener, self).__init__()
        
    def forward(self, y, h, lam):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = Ht * fft2(y) #.sum(axis=1).unsqueeze(1)
        lhs = HtH + lam
        x = ifftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x


class Wiener_Batched(nn.Module):
    def __init__(self):
        super(Wiener_Batched, self).__init__()
        
    def forward(self, y, h, lam):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(y)).sum(axis=-3).unsqueeze(-3)
        lhs = (HtH + lam).sum(axis=-3).unsqueeze(-3)
        x = ifftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x


class WienerNet(nn.Module):
    def __init__(self, n_delays=8, nc=[32, 64, 128, 256]):
        super(WienerNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.wiener = Wiener_Batched()
        self.lam = nn.Parameter(0.2 * torch.ones([1,1,1,1], device=self.device, requires_grad=True))
        self.denoiser = ResUNet(in_nc=1, out_nc=1, nb=2, nc=nc)
        
    def forward(self, y, psf):

        x = self.wiener(y, psf, self.lam)
        x = self.denoiser(x)
        
        return x 


if __name__ == '__main__':
    for nc in [8, 16, 32]:
        model = WienerNet(nc=[nc, nc*2, nc*4, nc*8])
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %s  (nc=%s)" % (total, nc))
    