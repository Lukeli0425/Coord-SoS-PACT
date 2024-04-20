import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

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


# class PSF_PACT(nn.Module):
#     def __init__(self, n_delays=8, delay_step=1e-4, n_points=80, l=3.2e-3, device='cpu'):
#         super(PSF_PACT, self).__init__() 
#         self.device = device
#         self.n_points = n_points # Size of PSF image in pixels.
#         self.l = l # Length [m] of the PSF image.
#         self.n_delays = n_delays
#         self.delays = torch.linspace(-(n_delays/2-1), n_delays/2, n_delays, requires_grad=False) * delay_step
#         self.delays = self.delays.view(1,n_delays,1,1) # [1,8,1,1]
        
#     def forward(self, C0, C1, phi1, C2, phi2, offset):
#         self.delays = self.delays.to(self.device)
#         k, theta = get_fourier_coord(n_points=self.n_points, l=self.l, device=self.device)
#         k, theta = k.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1), theta.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
#         w = lambda theta: C0 + C1 * torch.cos(theta + phi1) + C2 * torch.cos(2 * theta + phi2) # Wavefront function.
#         tf = (torch.exp(-1j*k*(self.delays + offset - w(theta))) + torch.exp(1j*k*(self.delays + offset - w(theta+np.pi)))) / 2
#         psf = ifftshift(ifftn(tf, dim=[-2,-1]), dim=[-2,-1]).abs()
#         psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        
#         return psf
    
    
class PSF_PACT(nn.Module):
    def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
        super(PSF_PACT, self).__init__() 
        self.device = device
        self.n_delays = n_delays
        self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l, device=device)
        self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        # angle_range = torch.tensor([angle_range[0] % (2*torch.pi), angle_range[1] % (2*torch.pi)])
        self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
        self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        
    def forward(self, delays, w):
        tf = (torch.exp(-1j*self.k2D*(delays - w(self.theta2D))) * self.mask0 + torch.exp(1j*self.k2D*(delays - w(self.theta2D+torch.pi))) * self.mask1) / 2
        psf = fftshift(ifft2(tf), dim=[-2,-1]).abs()
        psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        return psf
    
    
class TF_PACT(nn.Module):
    def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
        super(TF_PACT, self).__init__() 
        self.device = device
        self.n_delays = n_delays
        self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l, device=device)
        self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
        self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        
    def forward(self, delays, w):
        tf = (torch.exp(-1j*self.k2D*(delays - w(self.theta2D))) * self.mask0 + torch.exp(1j*self.k2D*(delays - w(self.theta2D+torch.pi))) * self.mask1) / 2
        return ifftshift(tf, dim=[-2,-1])


class Wavefront_SoS(nn.Module):
    def __init__(self, SoS, R_body, v0, x_vec, y_vec, n_points=90, N_int=500, device='cuda:0'):
        super(Wavefront_SoS, self).__init__()
        self.SoS = torch.tensor(SoS, dtype=torch.float64, device=device)
        self.R_body = torch.tensor(R_body, dtype=torch.float64, device=device)
        self.v0 = torch.tensor(v0, dtype=torch.float64, device=device)
        self.thetas = torch.linspace(0, 2*torch.pi, n_points, dtype=torch.float64, device=device).view(-1,1)
        self.x_vec = torch.tensor(x_vec, dtype=torch.float64, device=device)
        self.y_vec = torch.tensor(y_vec, dtype=torch.float64, device=device)
        self.dx, self.dy = torch.tensor(x_vec[1] - x_vec[0], dtype=torch.float64, device=device), torch.tensor(y_vec[1] - y_vec[0], dtype=torch.float64, device=device)
        self.N_int = N_int
        
        self.device = device
        
    def forward(self, x, y):
        r, phi = torch.sqrt(x**2 + y**2), torch.atan2(x, y)
        
        if r < self.R_body:
            l = torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi)
        else:
            l = 2 * torch.sqrt(torch.maximum(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2, torch.zeros_like(self.thetas))) * (torch.cos(phi-self.thetas) >= 0)
        steps = torch.linspace(0, 1, self.N_int, device=self.device).view(1,-1)
        x_index = ((x - l*steps*torch.sin(self.thetas) - self.x_vec[0]) / self.dx).round().int()
        y_index = ((y - l*steps*torch.cos(self.thetas) - self.y_vec[0]) / self.dy).round().int()
        wf = torch.trapezoid(1-self.v0/self.SoS[-y_index, x_index], l*steps, dim=1)
        return self.thetas.view(-1), wf
    
class Interp1D(nn.Module):
    def __init__(self, mode='linear'):
        super(Interp1D, self).__init__()
        self.mode = mode

    def forward(self, x, y, x_new):
        dx = x[1] - x[0]
        if self.mode == 'round':
            x_new_index = ((x_new - x[0]) / dx).round().int()
            return y[x_new_index]
        elif self.mode == 'linear':
            x_floor = ((x_new - x[0]) / dx).floor().int()
            x_ceil = ((x_new - x[0]) / dx).ceil().int()
            y_new = torch.zeros_like(x_new, dtype=y.dtype, device=y.device)
            y_new[x_ceil!=x_floor] = (((y[x_ceil] - y[x_floor]) * x_new + y[x_floor]*x[x_ceil] - y[x_ceil]*x[x_floor]) / (x[x_ceil] - x[x_floor]))[x_ceil!=x_floor]
            y_new[x_ceil==x_floor] = y[x_ceil[x_ceil==x_floor]]
            return y_new 
            

class TF_PACT(nn.Module):
    def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
        super(TF_PACT, self).__init__() 
        self.device = device
        self.n_delays = n_delays
        self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l, device=device)
        self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
        self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        self.interp1d = Interp1D(mode='linear')
        
    def forward(self, delays, thetas, wfs):
        w = self.interp1d(thetas, wfs, self.theta2D)
        w_pi = self.interp1d(thetas, wfs, torch.remainder(self.theta2D+torch.pi, 2*torch.pi))
        tf = (torch.exp(-1j*self.k2D*(delays - w)) * self.mask0 + torch.exp(1j*self.k2D*(delays - w_pi)) * self.mask1) / 2
        return ifftshift(tf, dim=[-2,-1])