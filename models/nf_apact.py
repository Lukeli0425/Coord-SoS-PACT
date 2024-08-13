import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from models.pact import TF_PACT, Wavefront_SOS
from models.regularizer import Brenner, L1_Norm, Total_Variation
from models.sos import SOS
from utils.reconstruction import get_gaussian_window
from utils.utils_torch import *


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean((x-y) ** 2, axis=(-3,-2,-1))


class NF_APACT(nn.Module):
    def __init__(self, mode, n_delays, hidden_layers, hidden_features, pos_encoding, N_freq, lam_tv, lam_ip,
                 x_vec, y_vec, R_body, v0, mean, std, N_patch=80, l_patch=3.2e-3, angle_range=(0, 2*torch.pi)):
        super().__init__()
        self.mode = mode
  
        fwhm = 1.5e-3 # [m]
        sigma = fwhm / 4e-5 / np.sqrt(2*np.log(2))
        self.gaussian_window = torch.tensor(get_gaussian_window(sigma, N_patch)).unsqueeze(0).cuda()
        self.k, _ = get_fourier_coord(N=2*N_patch, l=2*l_patch)
        self.k = ifftshift(self.k.cuda(), dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
        XX, YY = torch.meshgrid(torch.tensor(x_vec[:,0]), torch.tensor(y_vec[:,0]), indexing='xy')
        self.mask = torch.zeros_like(XX).cuda()
        self.mask[XX**2 + YY**2 <= R_body**2] = 1
        
        self.SOS = SOS(mode=self.mode, mask=self.mask, v0=v0, mean=mean, std=std, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq)
        self.wavefront_SOS = Wavefront_SOS(R_body=R_body, v0=v0, x_vec=x_vec, y_vec=y_vec, n_points=180, N_int=250)
        self.tf_pact = TF_PACT(N=2*N_patch, l=2*l_patch, n_delays=n_delays, angle_range=angle_range)
        self.loss = MSELoss()
        self.tv_regularizer = Total_Variation(weight=lam_tv)
        self.regularizer_IP = Brenner(weight=lam_ip)
        
        
    def forward(self, x, y, y_img, delays):
        # TF calculation.
        SoS = self.SOS()
        thetas, wfs = self.wavefront_SOS(x, y, SoS)
        H = self.tf_pact(delays.view(1,-1,1,1), thetas, wfs)
        
        # y_img = y_img * self.gaussian_window
        
        # Deconvolution.
        Y = fft2(ifftshift(pad_double(y_img), dim=(-2,-1)))
        Ht, HtH = H.conj(), H.abs() ** 2
        rhs = (Y * Ht).sum(axis=-3).unsqueeze(-3)
        lhs = HtH.sum(axis=-3).unsqueeze(-3)
        X = rhs / lhs
        x = crop_half(fftshift(ifft2(X), dim=(-2,-1)).real)

        loss = self.loss(Y.abs(), (H * X).abs()) + self.tv_regularizer(SoS, self.mask) + self.regularizer_IP(x)
        # loss = ((Y - H * X).abs() ** 2).mean() + self.tv_regularizer(SoS, self.mask)
            
        return x, SoS, loss


if __name__ == "__main__":
    jr = NF_APACT(SOS=1540, x_vec=[-0.02, 0.02], y_vec=[-0.02, 0.02], R=0.01, v0=1540, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), lam_tv=1e-3, lam_ip=1e-3, device='cuda:0')