import math
import numpy as np
import torch
import torch.fft as tfft
from torch.fft import fftn, ifftn, fftshift
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.ResUNet as ResUNet
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
    def __init__(self, n):
        super(SubNet, self).__init__()
        self.n = n
        self.conv_layers = nn.Sequential(
            Down(1,4),
            Down(8,8),
            Down(8,16),
            Down(16,16))
        self.mlp = nn.Sequential(
            nn.Linear(16*8*8+1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2*self.n),
            nn.Softplus())
        
    def forward(self, kernel, alpha):
        N, _, h, w  = kernel.size()
        h1, h2 = int(np.floor(0.5*(128-h))), int(np.ceil(0.5*(128-h)))
        w1, w2 = int(np.floor(0.5*(128-w))), int(np.ceil(0.5*(128-w)))
        k_pad = F.pad(kernel, (w1,w2,h1,h2), "constant", 0)
        H = tfft.fftn(k_pad, dim=[2,3])
        HtH = torch.abs(H)**2
        x = self.conv_layers(HtH.float())
        x = torch.cat((x.view(N,1,16*8*8),  alpha.float().view(N,1,1)), axis=2).float()
        output = self.mlp(x) + 1e-6

        rho1_iters = output[:,:,0:self.n].view(N, 1, 1, self.n).repeat(1,8,1,1)
        rho2_iters = output[:,:,self.n:2*self.n].view(N, 1, 1, self.n).repeat(1,8,1,1)
        
        return rho1_iters, rho2_iters


class X_Update(nn.Module):
    def __init__(self):
        super(X_Update, self).__init__()
        
    def forward(self, x0, HtH, z, u1, rho1):
        rhs = x0.sum(axis=1) + rho1 * (z - u1) 
        lhs = HtH.sum(axis=1) + rho1
        x = tfft.ifftn(rhs/lhs, dim=[2,3])
        return x.real
    
    
class Z_Update_ResUNet(nn.Module):
    """Updating Z with ResUNet as denoiser."""
    def __init__(self):
        super(Z_Update_ResUNet, self).__init__() 
        self.net = ResUNet(in_nc=1, out_nc=1)

    def forward(self, z):
        z_out = self.net(z.float())
        return z_out


class H_Update(nn.Module):
    def __init__(self):
        super(H_Update, self).__init__()
        
    def forward(self, h0, XtX, g, u2, rho2):
        rhs = h0 + rho2 * (g - u2) 
        lhs = XtX + rho2
        x = tfft.ifftn(rhs/lhs, dim=[2,3])
        return x.real


class PSF_PACT(nn.Module):
    """Updating G with CNN and PSF model."""
    def __init__(self, n_delays=8, delay_step=2e-4, n_points=128, l=3.2e-3):
        super(PSF_PACT, self).__init__() 
        self.n_points = n_points # Size of PSF image in pixels.
        self.l = l # Length [m] of the PSF image.
        self.delays = torch.linspace(-(n_delays/2-1), n_delays/2, n_delays) * delay_step
        self.delays = self.delays.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1,8,1,1]
        
    def forward(self, C0, C1, phi1, C2, phi2):
        k, theta = get_fourier_coord(n_points=self.n_points, l=self.l)
        k, theta = k.unsqueeze(0).unsqueeze(0).repeat(1,8,1,1), theta.unsqueeze(0).unsqueeze(0).repeat(1,8,1,1)
        w = lambda theta: C0 + C1 * torch.cos(theta + phi1) + C2 * torch.cos(2 * theta + phi2) # Wavefront function.
        tf = (torch.exp(-1j*k*(self.delays - w(theta))) + torch.exp(1j*k*(self.delays - w(theta+np.pi)))) / 2
        psf = fftshift(ifftn(tf, dim=[-2,-1])).abs()
        psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
        
        return psf


class G_Update_CNN(nn.Module):
    """Updating G with CNN and forward model."""
    def __init__(self):
        super(G_Update_CNN, self).__init__() 
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
            nn.Linear(64, 5)
        )
        self.psf = PSF_PACT(n_points=128)

    def forward(self, h0):
        N = h0.shape[0] # Batch size.
        
        # PSF parameter estimation.
        H = fftn(h0, dim=[-2,-1])
        HtH = torch.abs(H) ** 2
        x = self.cnn(HtH.float())
        x = x.view(N, 1, 32*8*8)
        params = self.mlp(x) + 1e-6
        params = params.repeat(1,8,1).unsqueeze(-1)
        
        # PSF reconstruction.
        g_out = self.psf(C0=params[:,:,0:1,:], C1=params[:,:,1:2,:], phi1=params[:,:,2:3,:], C2=params[:,:,3:4,:], phi2=params[:,:,4:,:])

        return g_out


class Unrolled_ADMM(nn.Module):
    def __init__(self, n_iters=8):
        super(Unrolled_ADMM, self).__init__()
        self.n = n_iters # Number of iterations.
        self.X = X_Update() # FFT based quadratic solution.
        self.Z = Z_Update_ResUNet() # Denoiser.
        self.H = H_Update()
        self.G = G_Update_CNN()
        self.SubNet = SubNet(self.n)

    def init(self, y):
        x = None
        h = None
        return x, h
        
    def forward(self, y):
        device = torch.device("cuda:0" if y.is_cuda else "cpu")
        N, _, _, _ = y.size()
        
        x, h = self.init(y) # Initialization.
        rho1_iters, rho2_iters = self.SubNet(y) 	# Hyperparameters.
        
        # Other ADMM variables.
        z = Variable(x.data.clone()).to(device)
        g = Variable(h.data.clone()).to(device)
        u1 = torch.zeros(y.size()).to(device)
        u2 = torch.zeros(h.size()).to(device)
		
        # ADMM iterations
        for n in range(self.n):
            _, H, Ht, HtH = psf_to_otf(h, y.size())
            
            rho1 = rho1_iters[:,:,:,n].view(N,1,1,1)
            rho2 = rho2_iters[:,:,:,n].view(N,1,1,1)
            
            # X, Z, H, G updates.
            x = self.X(x0=conv_fft_batch(Ht, y), HtH=HtH, z=z, u1=u1,rho1=rho1)
            z = self.Z(x + u1)
            
            _, X, Xt, XtX = psf_to_otf(h, y.size())
            h = self.H(h0=conv_fft_batch(Xt, y), XtX=XtX, g=g, u2=u2, rho2=rho2)
            g = self.G(h + u2)
            
            # Lagrangian updates.
            u1 = u1 + x - z            
            u2 = u2 + h - g
   
        return x, h



if __name__ == '__main__':
    model = Unrolled_ADMM()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % (total))