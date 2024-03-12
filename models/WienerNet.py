import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

from models.ResUNet import ResUNet


class Wiener(nn.Module):
    def __init__(self, lam=0.1):
        super(Wiener, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        
    def forward(self, y, h):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = Ht * fft2(y) #.sum(axis=1).unsqueeze(1)
        lhs = HtH + self.lam
        x = ifftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x


class Wiener_Batched(nn.Module):
    def __init__(self, lam=0.1):
        super(Wiener_Batched, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        
    def forward(self, y, h):
        H = fft2(fftshift(h))
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(fftshift(y))).sum(axis=-3).unsqueeze(-3)
        lhs = (HtH + self.lam).sum(axis=-3).unsqueeze(-3)
        x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x


class WienerNet(nn.Module):
    def __init__(self,  nc=[16, 32, 64, 128]):
        super(WienerNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.wiener = Wiener_Batched(lam=0.2)
        self.denoiser = ResUNet(in_nc=1, out_nc=1, nb=2, nc=nc)
        
    def forward(self, y, psf):

        x = self.wiener(y, psf)
        x = self.denoiser(x)
        
        return x 


if __name__ == '__main__':
    for nc in [8, 16, 32]:
        model = WienerNet(nc=[nc, nc*2, nc*4, nc*8])
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %s  (nc=%s)" % (total, nc))
    