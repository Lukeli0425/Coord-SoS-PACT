import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftn, fftshift, ifftn, ifftshift

from utils.utils_torch import get_fourier_coord


class FT2D(nn.Module):
    """2D Fourier Transform."""
    def __init__(self):
        super(FT2D, self).__init__()
        
    def forward(self, x):
        return fftn(x, dim=[-2,-1]).abs()


class IFT2D(nn.Module):
    """2D Inverse Fourier Transform."""
    def __init__(self):
        super(IFT2D, self).__init__()
        
    def forward(self, h):
        return ifftshift(ifftn(h, dim=[-2,-1]), dim=[-2,-1]).real


class PSF_PACT(nn.Module):
    def __init__(self, n_delays=8, delay_step=1e-4, n_points=80, l=3.2e-3, device='cpu'):
        super(PSF_PACT, self).__init__() 
        self.device = device
        self.n_points = n_points # Size of PSF image in pixels.
        self.l = l # Length [m] of the PSF image.
        self.n_delays = n_delays
        self.delays = torch.linspace(-(n_delays/2-1), n_delays/2, n_delays, requires_grad=False) * delay_step
        self.delays = self.delays.view(1,n_delays,1,1) # [1,8,1,1]
        
    def forward(self, C0, C1, phi1, C2, phi2, offset):
        self.delays = self.delays.to(self.device)
        k, theta = get_fourier_coord(n_points=self.n_points, l=self.l, device=self.device)
        k, theta = k.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1), theta.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        w = lambda theta: C0 + C1 * torch.cos(theta + phi1) + C2 * torch.cos(2 * theta + phi2) # Wavefront function.
        tf = (torch.exp(-1j*k*(self.delays + offset - w(theta))) + torch.exp(1j*k*(self.delays + offset - w(theta+np.pi)))) / 2
        psf = ifftshift(ifftn(tf, dim=[-2,-1]), dim=[-2,-1]).abs()
        psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        
        return psf
    
    
class PSF_PACT(nn.Module):
    def __init__(self, n_points=80, l=3.2e-3, device='cuda:0'):
        super(PSF_PACT, self).__init__() 
        self.device = device
        self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l, device=device)
        self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        
    def forward(self, w):
        tf = (torch.exp(-1j*self.k2D*(self.delays - w(self.theta2D))) + torch.exp(1j*self.k2D*(self.delays - w(self.theta2D+np.pi)))) / 2
        psf = fftshift(ifft2(tf), dim=[-2,-1]).abs()
        psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        
        return psf