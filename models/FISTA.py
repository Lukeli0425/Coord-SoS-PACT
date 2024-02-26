import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift


class FISTA_Batched(nn.Module):
    def __init__(self, n_iters=40, lam=0.1):
        super(FISTA_Batched, self).__init__()
        self.n_iters = n_iters
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        
        
    def forward(self, y, h):
        B, C, H, W = y.shape
        
        x = torch.ones([B,1,H,W], device=y.device) * (torch.max(h) + torch.min(h)) / 2
        v = x.clone()
        t = torch.tensor(1., device=y.device)
        
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        alpha = 0.01 / torch.max(HtH.real) # 2/1
        
        
        for idx in range(self.n_iters):
            # X-update
            x_prev = x
            x = v - alpha * (ifftshift(ifft2(HtH*fft2(v) - Ht*fft2(y)).sum(axis=-3).unsqueeze(-3), dim=(-2,-1)).real + self.lam * torch.sign(v))
            x = torch.maximum(x - self.lam, torch.zeros_like(x)) + torch.minimum(x + self.lam, torch.zeros_like(x)) # reg
            
            # t-update
            t_prev = t
            t = (1 + torch.sqrt(1 + 4*t**2)) / 2
            
            # y-update
            v = x + (t_prev - 1)/t * (x - x_prev)
            # print(x.shape,v.shape)
            
        return v