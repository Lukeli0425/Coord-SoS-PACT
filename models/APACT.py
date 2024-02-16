import numpy as np
import torch
import torch.nn as nn
from torch.fft import fftn, fftshift, ifftn, ifftshift

from models.WienerNet import Wiener
from utils.utils_torch import get_fourier_coord


class PSF_PACT(nn.Module):
    def __init__(self, n_delays=8, delay_step=1e-4, n_points=80, l=3.2e-3, device='cpu'):
        super(PSF_PACT, self).__init__() 
        self.device = device
        self.n_points = n_points # Size of PSF image in pixels.
        self.l = l # Length [m] of the PSF image.
        
    def forward(self, C0, C1, phi1, C2, phi2, delays):
        delays = delays.to(self.device)
        k, theta = get_fourier_coord(n_points=self.n_points, l=self.l, device=self.device)
        k, theta = k.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1), theta.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        w = lambda theta: C0 + C1 * torch.cos(theta + phi1) + C2 * torch.cos(2 * theta + phi2) # Wavefront function.
        tf = (torch.exp(-1j*k*(delays - w(theta))) + torch.exp(1j*k*(delays - w(theta+np.pi)))) / 2
        psf = ifftshift(ifftn(tf, dim=[-2,-1]), dim=[-2,-1]).abs()
        psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        
        return psf
    
    
class APACT(nn.Module):
    def __init__(self):
        self.params = None
        self.PSF = PSF_PACT()
        self.deconv = Wiener()
        self.loss = nn.MSELoss()
    
    def generate_params(self):
        pass
    
    def forward(self, y, delays):
        best_loss = torch.inf
        best_params = None
        best_x = None
        for params in enumerate(self.params):
            PSF = self.psf(params)
            x = self.deconv(y, PSF)
            y_hat = ifftshift(ifftn(PSF * fftn(y, dim=[-2,-1]), dim=[-2,-1]), dim=[-2,-1]).abs()
            loss = self.loss(y, y_hat)
            if loss < best_loss:
                best_loss = loss
                best_params = params
                
        return best_x, best_params