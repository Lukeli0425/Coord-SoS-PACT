import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from models.pact import TF_PACT, Wavefront_SOS
from models.regularizer import L1_Regularizer, TV_Regularizer
from models.sos import SOS
from utils.reconstruction import gaussian_kernel
from utils.utils_torch import *


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, x, y):
        return torch.mean((x-y) ** 2, axis=(-3,-2,-1))


class NF_APACT(nn.Module):
    def __init__(self, mode, x_vec, y_vec, R, v0, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), 
                 lam_tv=1e-3, lam_l1=0.0, mean=1545, std=130):
        super().__init__()
        self.mode = mode
  
        fwhm = 1.5e-3 # [m]
        sigma = fwhm / 4e-5 / np.sqrt(2*np.log(2))
        self.gaussian_window = torch.tensor(gaussian_kernel(sigma, n_points)).unsqueeze(0).to('cuda:0')
        self.k, _ = get_fourier_coord(n_points=2*n_points, l=2*l)
        self.k = ifftshift(self.k.to('cuda:0'), dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
        XX, YY = torch.meshgrid(torch.tensor(x_vec[:,0]), torch.tensor(y_vec[:,0]), indexing='xy')
        self.mask = torch.zeros_like(XX).to('cuda:0')
        self.mask[XX**2 + YY**2 <= R**2] = 1
        
        self.SOS = SOS(mode=self.mode, mask=self.mask, v0=v0, mean=mean, std=std)
        self.wavefront_SOS = Wavefront_SOS(R_body=R, v0=v0, x_vec=x_vec, y_vec=y_vec, n_points=180, N_int=250)
        self.tf_pact = TF_PACT(n_points=2*n_points, l=2*l, n_delays=n_delays, angle_range=angle_range)
        self.loss = MSELoss()
        self.tv_regularizer = TV_Regularizer(weight=lam_tv)
        # self.l1_regularizer = L1_Regularizer(weight=lam_l1, mean=mean)
        
        
    def forward(self, x, y, y_img, delays):
        # TF calculation.
        SoS = self.SOS()
        thetas, wfs = self.wavefront_SOS(x, y, SoS)
        H = self.tf_pact(delays.view(1,-1,1,1), thetas, wfs)
        
        y_img = y_img * self.gaussian_window
        
        # Deconvolution.
        Y = fft2(ifftshift(pad_double(y_img), dim=(-2,-1)))
        Ht, HtH = H.conj(), H.abs() ** 2
        rhs = (Y * Ht).sum(axis=-3).unsqueeze(-3)
        lhs = HtH.sum(axis=-3).unsqueeze(-3)
        X = rhs / lhs
        x = fftshift(ifft2(X), dim=(-2,-1)).real

        loss = self.loss(Y.abs(), (H * X).abs()) + self.tv_regularizer(SoS, self.mask) #+ self.l1_regularizer(SOS, self.mask)
        # loss = ((Y - H * X).abs() ** 2).mean() + self.tv_regularizer(SoS, self.mask)
            
        return x, SoS, loss


if __name__ == "__main__":
    jr = NF_APACT(SOS=1540, x_vec=[-0.02, 0.02], y_vec=[-0.02, 0.02], R=0.01, v0=1540, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), lam_tv=1e-3, lam_l1=1e-3, device='cuda:0')