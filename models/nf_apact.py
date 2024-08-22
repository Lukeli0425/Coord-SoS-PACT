import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from models.pact import TF_PACT, Wavefront_SOS
from models.regularizer import Sharpness, Total_Variation
from models.sos import SOS_Rep
from utils.reconstruction import get_gaussian_window
from utils.utils_torch import *
from models.deconv import MultiChannel_Deconv


class Data_Fitting_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Y, X, k):
        return torch.mean(k * (Y-X) ** 2, axis=(-3,-2,-1))


class NF_APACT(nn.Module):
    """Neural Fileds for Adaptive Photoacoustic Computed Tomography."""
    def __init__(self, n_delays, hidden_layers, hidden_features, pos_encoding, N_freq, lam_tv, reg, lam,
                 x_vec, y_vec, R_body, v0, mean, std, N_patch=80, l_patch=3.2e-3, fwhm = 1.5e-3, angle_range=(0, 2*torch.pi)):
        super().__init__()

        sigma = fwhm / 4e-5 / np.sqrt(2*np.log(2))
        self.gaussian_window = torch.tensor(get_gaussian_window(sigma, N_patch)).unsqueeze(0).cuda()
        self.k, _ = get_fourier_coord(N=2*N_patch, l=2*l_patch)
        self.k = ifftshift(self.k.cuda(), dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        self.k /= self.k.mean()
        
        XX, YY = torch.meshgrid(torch.tensor(x_vec[:,0]), torch.tensor(y_vec[:,0]), indexing='xy')
        self.SOS_mask = torch.zeros_like(XX).cuda()
        self.SOS_mask[XX**2 + YY**2 <= R_body**2] = 1
        self.SOS_results = None
        
        self.SOS = SOS_Rep(mode='SIREN', mask=self.SOS_mask, v0=v0, mean=mean, std=std, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq)
        self.wavefront_SOS = Wavefront_SOS(R_body=R_body, v0=v0, x_vec=x_vec, y_vec=y_vec, n_thetas=180, N_int=250)
        self.tf_pact = TF_PACT(N=2*N_patch, l=2*l_patch, n_delays=n_delays, angle_range=angle_range)
        self.deconv = MultiChannel_Deconv()
        self.data_fitting = Data_Fitting_Loss()
        self.tv_regularizer = Total_Variation(weight=lam_tv)
        self.sharpness_regularizer = Sharpness(function=reg, weight=lam)
    
    def save_SOS(self):
        """Save the SOS after optimization."""
        self.SOS_results = self.SOS()
    
    def forward(self, x, y, patch_stack, delays, task='train'):
        # Compute SOS. 
        SOS = self.SOS() if task =='train' else self.SOS_results # Use the saved SOS during deconvolution.
        
        # Compute TF stack.  
        thetas, wfs = self.wavefront_SOS(x, y, SOS)
        H = self.tf_pact(delays.view(1,-1,1,1), thetas, wfs)
        
        # Apply Gaussian window to image patch.
        patch_stack = patch_stack * self.gaussian_window
        
        # Deconvolve patch stacks using Pseudo-inverse.
        x, X, Y = self.deconv(patch_stack, H)

        # Compute loss.
        loss = self.data_fitting(Y.abs(), (H * X).abs(), self.k) + self.tv_regularizer(SOS, self.SOS_mask) #- self.sharpness_regularizer(x)
            
        return x, SOS, loss


if __name__ == "__main__":
    jr = NF_APACT(SOS=1540, x_vec=[-0.02, 0.02], y_vec=[-0.02, 0.02], R=0.01, v0=1540, n_points=80, l=3.2e-3, n_delays=32, angle_range=(0, 2*torch.pi), lam_tv=1e-3, lam=1e-3, device='cuda:0')