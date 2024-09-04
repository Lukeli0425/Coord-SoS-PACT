import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.fft import fftn, ifftn, ifftshift

from models.ResUNet import ResUNet
from utils.utils_torch import psf_to_otf

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv, self).__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


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
        
    def forward(self, Ht, y, HtH, z, u, rho):
        rhs = Ht * fftn(y, dim=[-2,-1]) + rho * fftn(z - u, dim=[-2,-1])
        lhs = HtH + rho
        x = ifftshift(ifftn(rhs/lhs, dim=[-2,-1]), dim=[-2,-1]).real

        return x
    
    
class Z_Update_ResUNet(nn.Module):
    """Updating Z with ResUNet as denoiser."""
    def __init__(self, nc=[16, 32, 64, 128]):
        super(Z_Update_ResUNet, self).__init__() 
        self.net = ResUNet(in_nc=8, out_nc=1, nc=nc)

    def forward(self, z):
        z_out = self.net(z.float())
        return z_out



class Unrolled_ADMM(nn.Module):
    def __init__(self, n_iters=8, n_delays=8, nc=[16, 32, 64, 128]):
        super(Unrolled_ADMM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n = n_iters # Number of iterations.
        self.n_delays = n_delays # Number of delays.
        self.X = X_Update() # FFT based quadratic solution.
        self.Z = Z_Update_ResUNet(nc=nc) # Denoiser.
        # self.psf_pact = PSF_PACT(n_delays=self.n_delays, device=self.device)
        # self.C0 = torch.tensor(7.8e-4, requires_grad=True)
        # self.psf_filter = torch.ones([1,n_delays,128,128], requires_grad=True, device=self.device)
        # self.psf_bias = torch.ones([1,n_delays,128,128], requires_grad=True, device=self.device)
        
        # self.SubNet = SubNet(self.n)
        self.rho_iters = torch.ones(size=(self.n,), requires_grad=True, device=self.device)

    def init_l2(self, y, H,):
        Ht, HtH = torch.conj(H), torch.abs(H)**2
        rhs = tfft.fftn(conv_fft_batch(Ht, y/alpha), dim=[2,3])
        lhs = HtH + (1/alpha)
        x0 = torch.real(tfft.ifftn(rhs/lhs, dim=[2,3]))
        x0 = torch.clamp(x0,0,1)
        return x0

    def forward(self, y, h):
        # y = y.cuda(self.device, non_blocking=True)
        B, _, H, W = y.size()
        # Other ADMM variables.
        z = Variable(x.data.clone()).to(self.device)
        u = torch.zeros(x.size()).to(self.device)
        # x = y
        # h = self.psf_pact(C0=self.C0, C1=0, phi1=0, C2=0, phi2=0) * self.psf_filter + self.psf_bias * 1e-5
        _, H, Ht, HtH = psf_to_otf(h)
        # rho1_iters, rho2_iters = self.SubNet(y)     # Hyperparameters.
        
        # z = Variable(x.data.clone()).to(self.device)
        
        # ADMM iterations
        for n in range(self.n):            
            # rho1 = rho1_iters[:,:,:,n].view(B,1,1,1)
            # rho2 = rho2_iters[:,:,:,n].view(B,8,1,1)
            rho = self.rho_iters[n]
            
            # X, Z updates.
            x = self.X(Ht=Ht, y=y, HtH=HtH, z=z, u=u, rho=rho)
            z = self.Z(x + u)

            # Lagrangian dual variable updates.
            u = u + x - z            

        return z



if __name__ == '__main__':
    model = Unrolled_ADMM(n_iters=4)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %s" % (total))