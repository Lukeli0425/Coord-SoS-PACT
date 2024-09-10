import math

import torch
import torch.nn as nn
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from models.deconv import MultiChannelDeconv
from models.pact import SOS2Wavefront, Wavefront2TF
from models.regularizer import Sharpness, TotalVariation
from models.siren import SIREN
from utils.reconstruction import get_gaussian_window
from utils.utils_torch import *
from utils.utils_torch import get_mgrid


class SOSRep(nn.Module):
    """SOS parameterization module."""
    def __init__(self, rep, mask, v0, mean, std, hidden_layers=None, hidden_features=None, pos_encoding=None, N_freq=None):
        super().__init__()
        self.rep = rep
        self.mask = mask
        self.v0 = v0
        self.mean, self.std = mean, std
        
        if rep == 'None':
            self.SOS = torch.normal(0,1,[(self.mask>0.5).sum(), 1], requires_grad=True).cuda()# * v0
            self.SOS = nn.Parameter(self.SOS, requires_grad=True)
        elif rep == 'SIREN':
            self.mgrid = get_mgrid(self.mask.shape, range=(-1, 1)).cuda()
            self.mgrid = self.mgrid[self.mask.view(-1)>0]
            self.siren = SIREN(in_features=2, out_features=1, hidden_features=hidden_features, hidden_layers=hidden_layers, activation_fn='sin', pos_encoding=pos_encoding, N_freq=N_freq)
            self.siren.cuda()
        else:
            raise NotImplementedError("Invalid representation. Choose from ['None', 'SIREN']")
        
    def forward(self):
        SOS = (torch.ones_like(self.mask, requires_grad=True) * self.v0).view(-1,1)
        if self.rep == 'None':
            SOS[self.mask.view(-1)>0] = self.SOS.double() * self.std + self.mean
        elif self.rep == 'SIREN':
            output, _ = self.siren(self.mgrid)
            SOS[self.mask.view(-1)>0] = output.double() * self.std + self.mean
        return SOS.view(self.mask.shape)
        
        
class DataFittingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Y, X, k):
        return torch.mean(k * (Y-X) ** 2, axis=(-3,-2,-1))


class NFAPACT(nn.Module):
    """Neural Fields for Adaptive Photoacoustic Computed Tomography."""
    def __init__(self, n_delays, hidden_layers, hidden_features, pos_encoding, N_freq, lam_tv, reg, lam,
                 x_vec, y_vec, R_body, v0, mean, std, N_patch=80, l_patch=3.2e-3, fwhm = 1.5e-3, angle_range=(0, 2*torch.pi)):
        super().__init__()

        sigma = fwhm / 4e-5 / math.sqrt(2*math.log(2))
        self.gaussian_window = torch.from_numpy(get_gaussian_window(sigma, N_patch)).unsqueeze(0).cuda()
        self.k, _ = get_fourier_coord(N=2*N_patch, l=2*l_patch)
        self.k = ifftshift(self.k.cuda(), dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        self.k /= self.k.mean()
        
        XX, YY = torch.meshgrid(torch.tensor(x_vec[:,0]), torch.tensor(y_vec[:,0]), indexing='xy')
        self.sos_mask = torch.zeros_like(XX).cuda()
        self.sos_mask[XX**2 + YY**2 <= R_body**2] = 1
        self.sos_deconv = None
        
        self.SOS = SOSRep(rep='SIREN', mask=self.sos_mask, v0=v0, mean=mean, std=std, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq)
        self.sos2wavefront = SOS2Wavefront(R_body=R_body, v0=v0, x_vec=x_vec, y_vec=y_vec, n_thetas=256, N_int=256)
        self.wavefront2tf = Wavefront2TF(N=2*N_patch, l=2*l_patch, n_delays=n_delays, angle_range=angle_range)
        self.deconv = MultiChannelDeconv()
        self.data_fitting = DataFittingLoss()
        self.tv_regularizer = TotalVariation(weight=lam_tv)
        # self.sharpness_regularizer = Sharpness(function=reg, weight=lam)
    
    def save_sos(self):
        """Save the SOS after optimization."""
        self.sos_deconv = self.SOS()
        
    def load_sos(self, SOS):
        """Load the SOS for deconvolution."""
        self.sos_deconv = SOS
        
    def forward(self, x, y, patch_stack, delays, task='train'):
        # Compute SOS. 
        SOS = self.SOS() if task =='train' else self.sos_deconv # Use the saved SOS during deconvolution.
        
        # Compute TF stack.  
        thetas, wfs = self.sos2wavefront(x, y, SOS)
        H = self.wavefront2tf(delays.view(1,-1,1,1), thetas, wfs)

        # Apply Gaussian window to image patch.
        patch_stack = patch_stack * self.gaussian_window
        
        # Deconvolve patch stacks using Pseudo-inverse.
        x, X, Y = self.deconv(patch_stack, H)

        # Compute loss.
        loss = self.data_fitting(Y.abs(), (H * X).abs(), self.k) + self.tv_regularizer(SOS, self.sos_mask) #- self.sharpness_regularizer(x)

        return x, SOS, loss.sum()

