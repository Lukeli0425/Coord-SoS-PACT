import numpy as np
import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift
import torch.nn as nn
import torch.nn.functional as F
from models.ResUNet import ResUNet
from models.PACT import PSF_PACT
from utils.utils_torch import conv_fft_batch, psf_to_otf



class Wiener(nn.Module):
    def __init__(self):
        super(Wiener, self).__init__()
        
    def forward(self, y, h, lam):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = Ht * fft2(y) #.sum(axis=1).unsqueeze(1)
        lhs = HtH + lam
        x = ifftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x


class WienerNet(nn.Module):
    def __init__(self, n_delays=8, nc=[32, 64, 128, 256]):
        super(WienerNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # self.kernel_estimator = ResUNet(in_nc=n_delays, out_nc=n_delays, nb=2, nc=[32, 64, 128, 256])
        
        # self.cnn = nn.Sequential(
        #     Down(8,8),
        #     Down(8,8),
        #     Down(8,16),
        #     Down(16,16)
        # )
        # self.mlp = nn.Sequential(
        #     nn.Linear(16*8*8, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 5),
        #     nn.Softplus()
        # )
        # self.psf = PSF_PACT(n_points=128, n_delays=n_delays, device=self.device)
        # self.C0 = torch.tensor(7.8e-4, requires_grad=True)
        # self.psf_filter = torch.ones([1,n_delays,128,128], device=self.device, requires_grad=True)
        # self.psf_bias = torch.ones([1,n_delays,128,128], device=self.device, requires_grad=True)
        
        self.wiener = Wiener()
        self.lam = nn.Parameter(0.25 * torch.ones([8,1,1], requires_grad=True, device=self.device))
        self.denoiser = ResUNet(in_nc=n_delays, out_nc=1, nb=2, nc=nc)
        
    def forward(self, y, psf):
        # print(self.lam)
        # psf = self.kernel_estimator(y)
        # N = y.shape[0] # Batch size.
        
        # # PSF parameter estimation.
        # H = fftn(y, dim=[-2,-1])
        # HtH = torch.abs(H) ** 2
        # x = self.cnn(HtH.float())
        # x = x.view(N, 1, 16*8*8)
        # params = self.mlp(x) + 1e-6
        # params = params.repeat(1,8,1).unsqueeze(-1)
        
        # PSF reconstruction.
        # psf = self.psf(C0=params[:,:,0:1,:], 
        #                C1=params[:,:,1:2,:], 
        #                phi1=params[:,:,2:3,:], 
        #                C2=params[:,:,3:4,:], 
        #                phi2=params[:,:,4:,:]) * self.psf_filter + self.psf_bias * 1e-5
        
        # psf = self.psf(C0=self.C0, 
        #                C1=0, 
        #                phi1=0, 
        #                C2=0, 
        #                phi2=0) * self.psf_filter + self.psf_bias * 1e-5
        # psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.

        x = self.wiener(y, psf, self.lam)
        x = self.denoiser(x)
        
        return x 
        
if __name__ == '__main__':
    for nc in [8, 16, 32]:
        model = WienerNet(nc=[nc, nc*2, nc*4, nc*8])
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %s  (nc=%s)" % (total, nc))
    