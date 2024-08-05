import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

from utils.utils_torch import crop_half, get_fourier_coord, pad_double


class Wiener_Batched(nn.Module):
    def __init__(self, lam, order=1, device='cuda:0'):
        super().__init__()
        self.device = device
        
        self.lam = torch.tensor(lam, device=device)
        self.order = torch.tensor(order, device=device)
        self.k, _ = get_fourier_coord(n_points=160, l=6.4e-3)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, y, H):
        y = pad_double(y)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(ifftshift(y, dim=(-2,-1)))).sum(axis=-3).unsqueeze(-3)
        lhs = (HtH + self.lam * (self.k.mean()/self.k) ** self.order).sum(axis=-3).unsqueeze(-3) 
        x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return crop_half(x)


class MultiChannel_Deconv(nn.Module):
    def __init__(self, N_patch=80, l_patch=3.2e-3):
        super().__init__()
        
        self.k, _ = get_fourier_coord(N=2*N_patch, l=2*l_patch)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, y, H):
        y = pad_double(y)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(ifftshift(y, dim=(-2,-1)))).sum(axis=-3).unsqueeze(-3)
        lhs = HtH.sum(axis=-3).unsqueeze(-3) 
        x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return crop_half(x)
    
