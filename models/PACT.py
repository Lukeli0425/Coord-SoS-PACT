import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift
from tqdm import tqdm

from models.Regularizer import L1_Regularizer, L2_Regularizer, TV_Regularizer
from models.SIREN import SIREN
from models.Wiener import Wiener_Batched
from utils.reconstruction import gaussian_kernel
from utils.utils_torch import *


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
    
    
# class TF_PACT(nn.Module):
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
    def __init__(self, R_body, v0, x_vec, y_vec, n_points=90, N_int=500, device='cuda:0'):
        super(Wavefront_SoS, self).__init__()
        # self.SoS = torch.tensor(SoS, dtype=torch.float64, device=device)
        self.R_body = torch.tensor(R_body, dtype=torch.float64, device=device)
        self.v0 = torch.tensor(v0, dtype=torch.float64, device=device)
        self.thetas = torch.linspace(0, 2*torch.pi, n_points, dtype=torch.float64, device=device).view(-1,1)
        self.x_vec = torch.tensor(x_vec, dtype=torch.float64, device=device)
        self.y_vec = torch.tensor(y_vec, dtype=torch.float64, device=device)
        self.dx, self.dy = torch.tensor(x_vec[1] - x_vec[0], dtype=torch.float64, device=device), torch.tensor(y_vec[1] - y_vec[0], dtype=torch.float64, device=device)
        self.N_int = N_int
        
        self.device = device
        
    def forward(self, x, y, SoS):
        r, phi = torch.sqrt(x**2 + y**2), torch.atan2(x, y)
        
        if r < self.R_body:
            l = torch.sqrt(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2) + r*torch.cos(self.thetas-phi)
        else:
            l = 2 * torch.sqrt(torch.maximum(self.R_body**2 - (r*torch.sin(self.thetas-phi))**2, torch.zeros_like(self.thetas))) * (torch.cos(phi-self.thetas) >= 0)
        steps = torch.linspace(0, 1, self.N_int, device=self.device).view(1,-1)
        x_index = ((x - l*steps*torch.sin(self.thetas) - self.x_vec[0]) / self.dx).round().int()
        y_index = ((y - l*steps*torch.cos(self.thetas) - self.y_vec[0]) / self.dy).round().int()
        wf = torch.trapezoid(1-self.v0/SoS[-y_index, x_index], l*steps, dim=1)
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
    
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, x, y):
        return torch.mean((x-y) ** 2, axis=(-3,-2,-1))
    
    
class TV_Prior(nn.Module):
    def __init__(self, lam=1e-3):
        super(TV_Prior, self).__init__()
        self.lam = lam
        
    def forward(self, x, mask):
        dx = (x[1:, :] - x[:-1, :]) * mask[1:, :]
        dy = (x[:, 1:] - x[:, :-1]) * mask[:, 1:]
        return self.lam * (torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))).mean()
    
    
class L1_Prior(nn.Module):
    def __init__(self, lam=1e-3, mean=1550):
        super(L1_Prior, self).__init__()
        self.lam = lam
        self.mean = mean
        
    def forward(self, x, mask):
        return self.lam * torch.mean((x-self.mean).abs() * mask)


class Joint_Recon(nn.Module):
    def __init__(self, SoS, x_vec, y_vec, R, v0, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), 
                 lam_tv=1e-3, lam_l1=1e-3, device='cuda:0'):
        super(Joint_Recon, self).__init__()
        self.device = device
        
        self.l = l
        
        fwhm = 1.5e-3 # [m]
        sigma = fwhm / 4e-5 / np.sqrt(2*np.log(2))
        self.gaussian_window = torch.tensor(gaussian_kernel(sigma, n_points), device=device).unsqueeze(0)
        self.k, self.theta = get_fourier_coord(n_points=2*n_points, l=2*l, device=device)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        # self.SoS = nn.Parameter(torch.tensor(SoS, dtype=torch.float64, device=device), requires_grad=True)
        self.SoS = torch.tensor(SoS, dtype=torch.float64, device=device)
        self.mask = torch.zeros_like(self.SoS)
        XX, YY = torch.meshgrid(torch.tensor(x_vec[:,0]), torch.tensor(y_vec[:,0]))
        self.mask[XX**2 + YY**2 <= R**2] = 1
        self.wavefront_sos = Wavefront_SoS(R_body=R, v0=v0, x_vec=x_vec, y_vec=y_vec, n_points=240, N_int=250, device=device)
        self.tf_pact = TF_PACT(n_points=2*n_points, l=2*l, n_delays=n_delays, angle_range=angle_range, device=device)
        self.loss = MSELoss()
        self.tv_prior = TV_Prior(lam=lam_tv)
        self.l2_prior = L1_Prior(lam=lam_l1)
        
        self.siren = SIREN(in_features=2, out_features=1, hidden_features=128, num_hidden_layers=2, activation_fn='sin')
        self.siren.cuda()
        # self.siren.load_state_dict(torch.load('siren.pth'))
        
        self.mgrid = get_mgrid(self.mask.shape, range=(-1, 1)).to(device)
        self.mgrid = self.mgrid[self.mask.view(-1)>0]
        
        
    def forward(self, x, y, y_img, delays):
        output, _ = self.siren(self.mgrid)
        SoS = (torch.ones_like(self.SoS, requires_grad=True) * 1499.363).view(-1,1)
        SoS[self.mask.view(-1)>0] = output.double() * 170.0 + 1550
        SoS = SoS.view(self.mask.shape)

        # self.SoS *= self.mask
        thetas, wfs = self.wavefront_sos(torch.tensor(x), torch.tensor(y), SoS)
        H = self.tf_pact(delays.view(1,-1,1,1), thetas, wfs)
        y_img = y_img * self.gaussian_window

        
        # Deconvolution
        Y = fft2(ifftshift(pad_double(y_img), dim=(-2,-1)))
        Ht, HtH = H.conj(), H.abs() ** 2
        rhs = (Y * Ht).sum(axis=-3).unsqueeze(-3)
        lhs = HtH.sum(axis=-3).unsqueeze(-3)
        X = rhs / lhs
        x = fftshift(ifft2(X), dim=(-2,-1)).real

        loss = self.loss(Y.abs() * self.k, (H * X).abs() * self.k) + self.tv_prior(SoS, self.mask) + self.l2_prior(SoS, self.mask)
            
        return x, SoS, loss
    

def get_mgrid(shape:tuple, range:tuple=(-1, 1)):
    """Generates a flattened grid of (x,y,...) coordinates in a range of `[-1, 1]`.
    
    Args:
        shape (`tuple`): Shape of the datacude to be fitted.
        range (`tuple`, optional): Range of the grid. Defaults to `(-1, 1)`.

    Returns:
        `torch.Tensor`: Generated flattened grid of coordinates.
    """
    tensors = [torch.linspace(range[0], range[1], steps=N) for N in shape]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(shape))
    return mgrid


if __name__ == "__main__":
    jr = Joint_Recon(SoS=1540, x_vec=[-0.02, 0.02], y_vec=[-0.02, 0.02], R=0.01, v0=1540, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), lam_tv=1e-3, lam_l1=1e-3, device='cuda:0')