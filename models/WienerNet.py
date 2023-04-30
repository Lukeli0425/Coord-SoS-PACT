import math
import numpy as np
import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.ResUNet import ResUNet
from models.PACT import PSF_PACT
from utils.utils_torch import conv_fft_batch, psf_to_otf, get_fourier_coord



class Wiener(nn.Module):
    def __init__(self) -> None:
        super(Wiener).__init__()
        
    def forward(self, y, h, lam):
        _, H, Ht, HtH = psf_to_otf(h)
        rhs = x0.sum(axis=1).unsqueeze(1)
        lhs = HtH.sum(axis=1).unsqueeze(1) + lam
        x = ifftshift(ifftn(rhs/lhs, dim=[-2,-1]), dim=[-2,-1]).real
        
        return x


class WienerNet(nn.Module):
    def __init__(self, n_delays=8) -> None:
        super(WienerNet).__init__()
        self.net = ResUNet(in_nc=n_delays, out_nc=1, nc=64, nb=2, nc=[16, 32, 64, 128])
        
    def forward(self, y):
        
        return
        