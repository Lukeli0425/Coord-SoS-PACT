import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from utils.utils_torch import *

# class PSF_PACT(nn.Module):
#     def __init__(self, n_points=80, l=3.2e-3, n_delays=16, angle_range=(0, 2*torch.pi), device='cuda:0'):
#         super().__init__() 
#         self.device = device
#         self.n_delays = n_delays
#         self.k2D, self.theta2D = get_fourier_coord(n_points=n_points, l=l).cuda()
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
    
    
class Fourier2Wavefront(nn.Module):
    def __init__(self, n_points=80):
        super().__init__() 
        self.thetas = torch.linspace(0, 2*torch.pi, n_points, dtype=torch.float64).cuda().view(-1,1)
        
    def forward(self, dc, x2, y2):
        wf = dc + x2 * torch.cos(2*self.thetas) + y2 * torch.sin(2*self.thetas)
        return wf
    
class Wavefront2Fourier(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, wf, thetas):
        dc = wf.mean().view(1)
        x2 = 2 * (wf * torch.cos(2*thetas)).mean().view(1)
        y2 = 2 * (wf * torch.sin(2*thetas)).mean().view(1)
        return torch.cat([dc, x2, y2], dim=0)

# class SOS2Wavefront(nn.Module):
#     def __init__(self, R_body, v0, x_vec, y_vec, n_thetas=180, N_int=500):
#         super().__init__()
#         # self.R_body = torch.tensor(R_body, dtype=torch.float64).cuda()
#         self.R_body = nn.Parameter(torch.tensor(R_body, dtype=torch.float64), requires_grad=False)
#         self.v0 = torch.tensor(v0, dtype=torch.float64).cuda()
#         self.thetas = torch.linspace(0, 2*torch.pi, n_thetas, dtype=torch.float64).cuda().view(-1,1)
#         self.x_vec = torch.tensor(x_vec, dtype=torch.float64).cuda()
#         self.y_vec = torch.tensor(y_vec, dtype=torch.float64).cuda()
#         self.dx, self.dy = torch.tensor(x_vec[1] - x_vec[0], dtype=torch.float64).cuda(), torch.tensor(y_vec[1] - y_vec[0], dtype=torch.float64).cuda()
#         self.N_int = N_int
        
#     def forward(self, x, y, SOS):
#         r, phi = torch.sqrt(x**2 + y**2), torch.atan2(x, y)
#         print(r.shape, phi.shape)
#         if r < self.R_body:
#             l = torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi)
#         else:
#             angle_mask = (torch.cos(phi-self.thetas) >= 0) * (self.R_body >= r*torch.sin(self.thetas-phi).abs())
#             l = torch.sqrt((self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) * angle_mask) + r * torch.cos(phi-self.thetas) * angle_mask
#         print(l.shape)
#         steps = torch.linspace(0, 1.0, self.N_int).cuda().view(1,-1)
#         j_index = ((x - l*steps*torch.sin(self.thetas) - self.x_vec[0]) / self.dx).round().int()
#         i_index = -((y - l*steps*torch.cos(self.thetas) - self.y_vec[-1]) / self.dy).round().int()
#         print(i_index.shape)
#         wf = torch.trapezoid(1-self.v0/SOS[i_index, j_index], l*steps, dim=1)
#         print(wf.shape)
#         return self.thetas.view(-1), wf

class SOS2Wavefront(nn.Module):
    def __init__(self, R_body, v0, x_vec, y_vec, n_thetas=180, N_int=500):
        super().__init__()
        # self.R_body = torch.tensor(R_body, dtype=torch.float64).cuda()
        self.R_body = nn.Parameter(torch.tensor(R_body, dtype=torch.float64), requires_grad=False)
        self.v0 = torch.tensor(v0, dtype=torch.float64).cuda()
        self.thetas = torch.linspace(0, 2*torch.pi, n_thetas, dtype=torch.float64).cuda().view(1,-1,1)
        self.x_vec = torch.tensor(x_vec, dtype=torch.float64).cuda()
        self.y_vec = torch.tensor(y_vec, dtype=torch.float64).cuda()
        self.dx, self.dy = torch.tensor(x_vec[1] - x_vec[0], dtype=torch.float64).cuda(), torch.tensor(y_vec[1] - y_vec[0], dtype=torch.float64).cuda()
        self.N_int = N_int
        
    def forward(self, x, y, SOS):
        r, phi = torch.sqrt(x**2 + y**2).view(-1,1,1), torch.atan2(x, y).view(-1,1,1) # Polar coordinate.
        l = torch.zeros(x.shape[0], self.thetas.shape[1], 1, dtype=torch.float64).cuda() # Angle-dependent distance to tissue boundary.
        l[r.view(-1)<self.R_body] = (torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi))[r.view(-1)<self.R_body]
        angle_mask = (torch.cos(phi-self.thetas)>=0) * (self.R_body>=r*torch.sin(self.thetas-phi).abs())
        l[r.view(-1)>=self.R_body] = (torch.sqrt((self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) * angle_mask) + r * torch.cos(phi-self.thetas) * angle_mask)[r.view(-1)>=self.R_body]
        # if r < self.R_body:
        #     l = torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi)
        # else:
        #     angle_mask = (torch.cos(phi-self.thetas) >= 0) * (self.R_body >= r*torch.sin(self.thetas-phi).abs())
        #     l = torch.sqrt((self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) * angle_mask) + r * torch.cos(phi-self.thetas) * angle_mask
        steps = torch.linspace(0, 1.0, self.N_int).cuda().view(1,1,-1)
        j_index = ((x.view(-1,1,1) - l*steps*torch.sin(self.thetas) - self.x_vec[0]) / self.dx).round().int()
        i_index = -((y.view(-1,1,1) - l*steps*torch.cos(self.thetas) - self.y_vec[-1]) / self.dy).round().int()
        wf = torch.trapezoid(1-self.v0/SOS[i_index, j_index], l*steps, dim=-1)
        return self.thetas.view(1,-1), wf

class Interp1D(nn.Module):
    """1D Wavefront Interpolation Module for TF calculation."""
    def __init__(self, mode='linear'):
        super().__init__()
        self.mode = mode

    def forward(self, x, y, x_new):
        dx = x[0,1] - x[0,0]
        x_new = x_new.repeat(y.shape[0],1,1,1)
        if self.mode == 'round':
            x_new_index = ((x_new - x[0,0]) / dx).round().int()
            return y[x_new_index]
        elif self.mode == 'linear':
            x_floor = ((x_new - x[0,0]) / dx).floor().to(torch.int64)
            x_ceil = ((x_new - x[0,0]) / dx).ceil().to(torch.int64)
            y_new = torch.zeros_like(x_new, dtype=y.dtype).cuda()
            # Batch-wise indexing.
            y = y.unsqueeze(-1).unsqueeze(-1).repeat(1,1,x_new.shape[2],x_new.shape[3])
            x = x.view(1,-1,1,1).repeat(y.shape[0],1,x_new.shape[2],x_new.shape[3])
            y_ceil = torch.gather(y, 1, x_ceil)
            y_floor = torch.gather(y, 1, x_floor)
            xx_ceil = torch.gather(x, 1, x_ceil)
            xx_floor = torch.gather(x, 1, x_floor)
            # Linear interpolation.
            y_new[x_ceil!=x_floor] = ((y_ceil - y_floor) * x_new + y_floor*xx_ceil - y_ceil*xx_floor)[x_ceil!=x_floor] / (xx_ceil - xx_floor)[x_ceil!=x_floor]
            y_new[x_ceil==x_floor] = torch.gather(y, 1, x_ceil[x_ceil==x_floor]) # y[x_ceil[x_ceil==x_floor]]
            return y_new 
            

class Wavefront2TF(nn.Module):
    """Wavefront to Transfer Function Module."""
    def __init__(self, N, l, n_delays, angle_range=(0, 2*torch.pi)):
        super().__init__() 
        self.n_delays = n_delays
        self.k2D, self.theta2D = get_fourier_coord(N=N, l=l)
        self.k2D = self.k2D.cuda().unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.cuda().unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.mask0 = (self.theta2D >= angle_range[0]) * (self.theta2D <= angle_range[1]).float()
        self.mask1 = ((self.theta2D + torch.pi) % (2*torch.pi) >= angle_range[0]) * ((self.theta2D + torch.pi) % (2*torch.pi) <= angle_range[1]).float()
        self.interp1d = Interp1D(mode='linear')
        
    def forward(self, delays, thetas, wfs):
        w = self.interp1d(thetas, wfs, self.theta2D)
        w_pi = self.interp1d(thetas, wfs, torch.remainder(self.theta2D+torch.pi, 2*torch.pi))
        tf = (torch.exp(-1j*self.k2D*(delays - w)) * self.mask0 + torch.exp(1j*self.k2D*(delays - w_pi)) * self.mask1) / 2
        return ifftshift(tf, dim=[-2,-1])
    
    