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


  

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SubNet(nn.Module):
    def __init__(self, n_iters=8):
        super(SubNet, self).__init__()
        self.n_iters = n_iters
        self.cnn = nn.Sequential(
            Down(8,8),
            Down(8,16),
            Down(16,32),
            Down(32,32)
        )
        self.mlp = nn.Sequential(
            nn.Linear(32*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2*self.n_iters),
            nn.Softplus()
        )
        
    def forward(self, y):
        B, _, h, w  = y.size()
        H = fftn(y, dim=[-2,-1])
        HtH = torch.abs(H) ** 2
        x = self.cnn(HtH.float())
        x = x.view(B, 1, 32*8*8)
        output = self.mlp(x) + 1e-6

        rho1_iters = output[:,:,0:self.n_iters].view(B, 1, 1, self.n_iters)
        rho2_iters = output[:,:,self.n_iters:2*self.n_iters].view(B, 1, 1, self.n_iters).repeat(1,8,1,1)
        
        return rho1_iters, rho2_iters


class X_Update(nn.Module):
    def __init__(self):
        super(X_Update, self).__init__()
        
    def forward(self, x0, HtH, z, u1, rho1):
        rhs = x0.sum(axis=1).unsqueeze(1) + rho1 * (z - u1) 
        lhs = HtH.sum(axis=1).unsqueeze(1) + rho1
        x = ifftn(rhs/lhs, dim=[-2,-1])

        return x.real
    
    
class Z_Update_ResUNet(nn.Module):
    """Updating Z with ResUNet as denoiser."""
    def __init__(self):
        super(Z_Update_ResUNet, self).__init__() 
        self.net = ResUNet(in_nc=1, out_nc=1, nc=[16, 32, 64, 128])

    def forward(self, z):
        z_out = self.net(z.float())
        return z_out


class H_Update(nn.Module):
    def __init__(self):
        super(H_Update, self).__init__()
        
    def forward(self, h0, XtX, g, u2, rho2):
        rhs = h0 + rho2 * (g - u2) 
        lhs = XtX + rho2
        x = ifftn(rhs/lhs, dim=[-2,-1])
        return x.real



class G_Update_CNN(nn.Module):
    """Updating G with CNN and forward model."""
    def __init__(self, n_delays=8):
        super(G_Update_CNN, self).__init__() 
        self.cnn = nn.Sequential(
            Down(8,8),
            Down(8,8),
            Down(8,16),
            Down(16,16)
        )
        self.mlp = nn.Sequential(
            nn.Linear(16*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
            nn.Softplus()
        )
        self.psf = PSF_PACT(n_points=128, n_delays=n_delays)

    def forward(self, h0, device):
        N = h0.shape[0] # Batch size.
        
        # PSF parameter estimation.
        H = fftn(h0, dim=[-2,-1]).to(device)
        HtH = torch.abs(H) ** 2
        x = self.cnn(HtH.float())
        x = x.view(N, 1, 16*8*8)
        params = self.mlp(x) + 1e-6
        params = params.repeat(1,8,1).unsqueeze(-1)
        
        # PSF reconstruction.
        g_out = self.psf(C0=params[:,:,0:1,:], C1=params[:,:,1:2,:], phi1=params[:,:,2:3,:], C2=params[:,:,3:4,:], phi2=params[:,:,4:,:], device=device)

        return g_out


class Unrolled_ADMM(nn.Module):
    def __init__(self, n_iters=8, n_delays=8):
        super(Unrolled_ADMM, self).__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n = n_iters # Number of iterations.
        self.n_delays = n_delays # Number of delays.
        self.X = X_Update() # FFT based quadratic solution.
        self.Z = Z_Update_ResUNet() # Denoiser.
        self.H = H_Update()
        self.G = G_Update_CNN(n_delays=self.n_delays) # Model-based PSF denoiser.
        # self.SubNet = SubNet(self.n)
        self.rho1_iters = torch.ones(size=[self.n,], requires_grad=True)
        self.rho2_iters = torch.ones(size=[self.n,], requires_grad=True)

    def init(self, y):
        B = y.shape[0] # Batch size.
        x = y[:,-1:,:,:]
        psf_pact = PSF_PACT(n_delays=self.n_delays)
        h = psf_pact(C0=2e-5 * torch.ones([B,self.n_delays,1,1], device=self.device), 
                     C1=2e-5 * torch.ones([B,self.n_delays,1,1], device=self.device), 
                     phi1=torch.zeros([B,self.n_delays,1,1], device=self.device), 
                     C2=2e-5 * torch.ones([B,self.n_delays,1,1], device=self.device), 
                     phi2=torch.zeros([B,self.n_delays,1,1], device=self.device),
                     device=self.device)
        return x, h
        
    def forward(self, y):
        
        B, _, H, W = y.size()
        
        x, h = self.init(y) # Initialization.
        rho1_iters, rho2_iters = self.SubNet(y) 	# Hyperparameters.
        
        # Other ADMM variables.
        z = Variable(x.data.clone()).to(self.device)
        g = Variable(h.data.clone()).to(self.device)
        u1 = torch.zeros(x.size()).to(self.device)
        u2 = torch.zeros(h.size()).to(self.device)
		
        # ADMM iterations
        for n in range(self.n):
            _, H, Ht, HtH = psf_to_otf(h, y.size(), self.device)
            
            # rho1 = rho1_iters[:,:,:,n].view(B,1,1,1)
            # rho2 = rho2_iters[:,:,:,n].view(B,8,1,1)
            rho1, rho2 = self.rho1_iters[n], self.rho2_iters[n]
            
            # X, Z, H, G updates.
            x = self.X(x0=conv_fft_batch(Ht, y), HtH=HtH, z=z, u1=u1, rho1=rho1)
            z = self.Z(x + u1)
            
            _, X, Xt, XtX = psf_to_otf(x, x.size(), self.device)
            h = self.H(h0=conv_fft_batch(Xt, y), XtX=XtX, g=g, u2=u2, rho2=rho2)
            g = self.G(h + u2, self.device)

            # Lagrangian dual variable updates.
            u1 = u1 + x - z            
            u2 = u2 + h - g

        return x, h



if __name__ == '__main__':
    model = Unrolled_ADMM(n_iters=4)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % (total))