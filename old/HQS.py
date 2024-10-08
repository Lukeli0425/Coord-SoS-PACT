import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift


class HQS(nn.Module):
    def __init__(self, n_iters=16, lam=0.1, rho=0.03):
        super(HQS, self).__init__()
        self.n_iters = n_iters
        self.lam = nn.Parameter(torch.ones(n_iters)*lam, requires_grad=True)
        self.rho = nn.Parameter(torch.ones(n_iters)*rho, requires_grad=True)
        
    def forward(self, y, h):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2

        z = torch.zeros_like(y) # y.clone()
        
        for idx in range(self.n_iters):
            # X-update
            rhs = Ht * fft2(y) + self.rho[idx] * fft2(z)
            lhs = HtH + self.rho[idx]
            x = ifftshift(ifft2(rhs/lhs), dim=[-2,-1]).real
            
            # Z-update
            z = torch.maximum(x - self.lam[idx], torch.zeros_like(x)) + torch.minimum(x + self.lam[idx], torch.zeros_like(x))
            
        return x



class HQS_Batched(nn.Module):
    def __init__(self, n_iters=16, lam=0.1, rho=0.03):
        super(HQS_Batched, self).__init__()
        self.n_iters = n_iters
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor(rho), requires_grad=True)

        
    def forward(self, y, h):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        B, C, H, W = y.shape    
        
        z = torch.zeros([B, 1, H, W], device=y.device)
        
        for _ in range(self.n_iters):
            # X-update
            rhs = (Ht * fft2(y)).sum(axis=-3).unsqueeze(-3) + self.rho * fft2(z)
            lhs = (HtH + self.rho).sum(axis=-3).unsqueeze(-3)
            x = ifftshift(ifft2(rhs/lhs), dim=[-2,-1]).real
            
            # Z-update
            z = torch.maximum(x - self.lam, torch.zeros_like(x)) + torch.minimum(x + self.lam, torch.zeros_like(x))
            
            # Dual-update       

        return x
    
    
if __name__ == "__main__":
    model = HQS_Batched()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % total)