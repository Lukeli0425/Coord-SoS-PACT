import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from utils.utils_torch import *

# class PSF_PACT(nn.Module):
#     def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
#         super().__init__() 
#         self.device = device
#         self.n_delays = n_delays
#         self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l).to('cuda:0')
#         self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
#         self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
#         # angle_range = torch.tensor([angle_range[0] % (2*torch.pi), angle_range[1] % (2*torch.pi)])
#         self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
#         self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        
#     def forward(self, delays, w):
#         tf = (torch.exp(-1j*self.k2D*(delays - w(self.theta2D))) * self.mask0 + torch.exp(1j*self.k2D*(delays - w(self.theta2D+torch.pi))) * self.mask1) / 2
#         psf = fftshift(ifft2(tf), dim=[-2,-1]).abs()
#         psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
#         return psf
    
    
# class TF_PACT(nn.Module):
#     def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
#         super().__init__() 
#         self.device = device
#         self.n_delays = n_delays
#         self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l).to('cuda:0')
#         self.k2D = self.k2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
#         self.theta2D = self.theta2D.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
#         self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
#         self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        
#     def forward(self, delays, w):
#         tf = (torch.exp(-1j*self.k2D*(delays - w(self.theta2D))) * self.mask0 + torch.exp(1j*self.k2D*(delays - w(self.theta2D+torch.pi))) * self.mask1) / 2
#         return ifftshift(tf, dim=[-2,-1])


class Wavefront_SOS(nn.Module):
    def __init__(self, R_body, v0, x_vec, y_vec, n_points=90, N_int=500):
        super().__init__()
        # self.R_body = torch.tensor(R_body, dtype=torch.float64).to('cuda:0')
        self.R_body = nn.Parameter(torch.tensor(R_body, dtype=torch.float64), requires_grad=False)
        self.v0 = torch.tensor(v0, dtype=torch.float64).to('cuda:0')
        self.thetas = torch.linspace(0, 2*torch.pi, n_points, dtype=torch.float64).to('cuda:0').view(-1,1)
        self.x_vec = torch.tensor(x_vec, dtype=torch.float64).to('cuda:0')
        self.y_vec = torch.tensor(y_vec, dtype=torch.float64).to('cuda:0')
        self.dx, self.dy = torch.tensor(x_vec[1] - x_vec[0], dtype=torch.float64).to('cuda:0'), torch.tensor(y_vec[1] - y_vec[0], dtype=torch.float64).to('cuda:0')
        self.N_int = N_int
        
    def forward(self, x, y, SOS):
        r, phi = torch.sqrt(x**2 + y**2), torch.atan2(x, y)
        
        if r < self.R_body:
            l = torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi)
        else:
            # l = 2 * torch.sqrt(torch.maximum(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2, torch.zeros_like(self.thetas))) * (torch.cos(phi-self.thetas) >= 0)
            angle_mask = (torch.cos(phi-self.thetas) >= 0) * (self.R_body >= r*torch.sin(self.thetas-phi).abs())
            l = torch.sqrt((self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) * angle_mask) + r * torch.cos(phi-self.thetas) * angle_mask
        steps = torch.linspace(0, 1.0, self.N_int).to('cuda:0').view(1,-1)
        j_index = ((x - l*steps*torch.sin(self.thetas) - self.x_vec[0]) / self.dx).round().int()
        # j_index = torch.clamp(j_index, -self.x_vec.shape[0], self.x_vec.shape[0]-1)
        # i_index = ((y - l*steps*torch.cos(self.thetas) - self.y_vec[0]) / self.dy).round().int()
        i_index = -((y - l*steps*torch.cos(self.thetas) - self.y_vec[-1]) / self.dy).round().int()
        # i_index = torch.clamp(i_index, -self.y_vec.shape[0]+1, self.y_vec.shape[0])
        # print(x, y)
        # print(l.min(), l.max(), self.thetas[l.argmax()]/2/torch.pi*360)
        # print(i_index.min(), i_index.max(), j_index.min(), j_index.max())
        wf = torch.trapezoid(1-self.v0/SOS[i_index, j_index], l*steps, dim=1)
        return self.thetas.view(-1), wf


class Interp1D(nn.Module):
    """1D Interpolation Module for TF calculation."""
    def __init__(self, mode='linear'):
        super().__init__()
        self.mode = mode

    def forward(self, x, y, x_new):
        dx = x[1] - x[0]
        if self.mode == 'round':
            x_new_index = ((x_new - x[0]) / dx).round().int()
            return y[x_new_index]
        elif self.mode == 'linear':
            x_floor = ((x_new - x[0]) / dx).floor().int()
            x_ceil = ((x_new - x[0]) / dx).ceil().int()
            y_new = torch.zeros_like(x_new, dtype=y.dtype).to('cuda:0')
            # print(x_ceil.max(), x_floor.min())
            # y_new[x_ceil!=x_floor] = (((y[x_ceil] - y[x_floor]) * x_new + y[x_floor]*x[x_ceil] - y[x_ceil]*x[x_floor]) / (x[x_ceil] - x[x_floor]))[x_ceil!=x_floor]
            y_new[x_ceil!=x_floor] = ((y[x_ceil] - y[x_floor]) * x_new + y[x_floor]*x[x_ceil] - y[x_ceil]*x[x_floor])[x_ceil!=x_floor] / (x[x_ceil] - x[x_floor])[x_ceil!=x_floor]
            y_new[x_ceil==x_floor] = y[x_ceil[x_ceil==x_floor]]
            return y_new 
            

class TF_PACT(nn.Module):
    def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi)):
        super().__init__() 
        self.n_delays = n_delays
        self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l)
        self.k2D = self.k2D.to('cuda:0').unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.to('cuda:0').unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
        self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        self.interp1d = Interp1D(mode='linear')
        
    def forward(self, delays, thetas, wfs):
        w = self.interp1d(thetas, wfs, self.theta2D)
        w_pi = self.interp1d(thetas, wfs, torch.remainder(self.theta2D+torch.pi, 2*torch.pi))
        tf = (torch.exp(-1j*self.k2D*(delays - w)) * self.mask0 + torch.exp(1j*self.k2D*(delays - w_pi)) * self.mask1) / 2
        return ifftshift(tf, dim=[-2,-1])
    
    