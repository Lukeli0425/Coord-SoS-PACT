import numpy as np
import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift
import torch.nn as nn
import torch.nn.functional as F
from models.ResUNet import ResUNet
from models.PACT import PSF_PACT
from utils.utils_torch import conv_fft_batch, psf_to_otf



class ADMM(nn.Module):
    def __init__(self, n_iters):
        super(ADMM, self).__init__()
        self.n_iters = n_iters

        
    def forward(self, y, h, rho, lam):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2

        z = torch.zeros_like(y) # y.clone()
        u = torch.zeros_like(y)
        
        for _ in range(self.n_iters):
            # X-update
            rhs = Ht * fft2(y) + rho * fft2(z - u)
            lhs = HtH + rho
            x = ifftshift(ifft2(rhs/lhs), dim=[-2,-1]).real
            
            # Z-update
            z = torch.maximum(x + u - lam, torch.zeros_like(x)) + torch.minimum(x + u + lam, torch.zeros_like(x))
            
            # Dual-update
            u = u + x - z            

        return x



class ADMM_Batched(nn.Module):
    def __init__(self, n_iters):
        super(ADMM_Batched, self).__init__()
        self.n_iters = n_iters

        
    def forward(self, y, h, lam, rho=0.05):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        B, C, H, W = y.shape    
        
        z, u = torch.zeros([B, 1, H, W], device=y.device), torch.zeros([B, 1, H, W], device=y.device)
        
        for _ in range(self.n_iters):
            # X-update
            rhs = (Ht * fft2(y)).sum(axis=-3).unsqueeze(-3) + rho * fft2(z - u)
            lhs = (HtH + rho).sum(axis=-3).unsqueeze(-3)
            x = ifftshift(ifft2(rhs/lhs), dim=[-2,-1]).real
            
            # Z-update
            z = torch.maximum(x + u - lam, torch.zeros_like(x)) + torch.minimum(x + u + lam, torch.zeros_like(x))
            
            # Dual-update
            u = u + x - z            

        return x
    
    
if __name__ == "__main":
    pass