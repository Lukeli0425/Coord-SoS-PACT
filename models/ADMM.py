import numpy as np
import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift
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
        H = fftn(h, dim=[-2,-1])
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2

        z = torch.zeros_like(y) # y.clone()
        u = torch.zeros_like(y)
        
        for i in range(self.n_iters):
            # X-update
            rhs = Ht * fftn(y, dim=[-2,-1]) + rho * fftn(z - u, dim=[-2,-1])
            lhs = HtH + rho
            x = ifftshift(ifftn(rhs/lhs, dim=[-2,-1]), dim=[-2,-1]).real
            
            # Z-update
            # z = torch.maximum(x + u - lam, torch.zeros_like(x)) - torch.maximum(-x - u - lam, torch.zeros_like(x))
            z = torch.maximum(x + u - lam, torch.zeros_like(x)) + torch.minimum(x + u + lam, torch.zeros_like(x))
            
            # Dual-update
            u = u + x - z            

        return x
    
    
    
if __name__ == "__main":
    pass