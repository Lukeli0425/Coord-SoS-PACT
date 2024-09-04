import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

from utils.utils_torch import crop_half, get_fourier_coord, pad_double


class ADMM(nn.Module):
    def __init__(self, n_iters=16, lam=0.1, rho=0.03):
        super(ADMM, self).__init__()
        self.n_iters = n_iters
        self.lam = nn.Parameter(torch.ones(n_iters)*lam, requires_grad=True)
        self.rho = nn.Parameter(torch.ones(n_iters)*rho, requires_grad=True)
        
    def forward(self, y, h):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2

        z = torch.zeros_like(y) # y.clone()
        u = torch.zeros_like(y)
        
        for idx in range(self.n_iters):
            # X-update
            rhs = Ht * fft2(y) + self.rho[idx] * fft2(z - u)
            lhs = HtH + self.rho[idx]
            x = ifftshift(ifft2(rhs/lhs), dim=[-2,-1]).real
            
            # Z-update
            z = torch.maximum(x + u - self.lam[idx], torch.zeros_like(x)) + torch.minimum(x + u + self.lam[idx], torch.zeros_like(x))
            
            # Dual-update
            u = u + x - z            

        return x



class ADMM_Batched(nn.Module):
    def __init__(self, n_iters=20, lam=0.1, rho=0.03, device='cuda:0'):
        super(ADMM_Batched, self).__init__()
        self.device = device
        
        self.n_iters = n_iters
        self.lam = torch.tensor(lam, device=device)
        self.rho = torch.tensor(rho, device=device)
        self.k, self.theta = get_fourier_coord(n_points=160, l=6.4e-3, device=device)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)

        
    def forward(self, y, H):
        y = pad_double(y)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        B, C, H, W = y.shape    
        
        z = torch.zeros([B, 1, H, W], device=self.device)
        u = torch.zeros([B, 1, H, W], device=self.device)
        
        for _ in range(self.n_iters):
            # X-update
            rhs = (Ht * fft2(ifftshift(y, dim=(-2,-1)))).sum(axis=-3).unsqueeze(-3) + self.rho * fft2(z - u)
            lhs = (HtH + self.rho).sum(axis=-3).unsqueeze(-3)
            x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
            
            # Z-update
            z = torch.maximum(x + u - self.lam, torch.zeros_like(x)) + torch.minimum(x + u + self.lam, torch.zeros_like(x))
            
            # Dual-update
            u = u + x - z            

        return crop_half(x)
    
    
if __name__ == "__main__":
    model = ADMM_Batched()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % total)