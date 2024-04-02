import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

from models.ResUNet import ResUNet
from utils.utils_torch import crop_half, get_fourier_coord, pad_double


class Wiener(nn.Module):
    def __init__(self, lam=0.1):
        super(Wiener, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)
        
    def forward(self, y, h):
        H = fft2(h)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = Ht * fft2(y) #.sum(axis=1).unsqueeze(1)
        lhs = HtH + self.lam
        x = ifftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return x

def pad(img, device='cuda:0'):
    B, C, H, W = img.shape
    img_pad = torch.zeros(B, C, H*2, W*2, device=device)
    img_pad[:,:,H//2:3*H//2, W//2:3*W//2] = img
    return img_pad

def crop(img):
    B, C, H, W = img.shape
    return img[:,:,H//4:3*H//4, W//4:3*W//4]


# class PACT_Deconv(nn.Module):
#     def __init__(self, tf, lam, order=1, device='cuda:0'):
#         super(PACT_Deconv, self).__init__()
#         self.device = device
#         self.lam = torch.tensor(lam, device=device)
#         self.k, self.theta = get_fourier_coord(n_points=160, l=6.4e-3, device=device)
#         self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
        
    

class Wiener_Batched(nn.Module):
    def __init__(self, lam, device='cuda:0'):
        super(Wiener_Batched, self).__init__()
        self.device = device
        self.lam = torch.tensor(lam, device=device)
        self.k, self.theta = get_fourier_coord(n_points=160, l=6.4e-3, device=device)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, y, h):
        y, h = pad_double(y), pad_double(h)
        H = fft2(ifftshift(h))
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(ifftshift(y))).sum(axis=-3).unsqueeze(-3)
        lhs = (HtH + self.lam * ((self.k.mean()/self.k))**1).sum(axis=-3).unsqueeze(-3) 
        x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return crop_half(x)


class Wiener_Batched(nn.Module):
    def __init__(self, lam, order=1, device='cuda:0'):
        super(Wiener_Batched, self).__init__()
        self.device = device
        self.lam = torch.tensor(lam, device=device)
        self.order = torch.tensor(order, device=device)
        self.k, self.theta = get_fourier_coord(n_points=160, l=6.4e-3, device=device)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, y, H):
        y = pad_double(y)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        
        rhs = (Ht * fft2(ifftshift(y, dim=(-2,-1)))).sum(axis=-3).unsqueeze(-3)
        lhs = (HtH + self.lam * (self.k.mean()/self.k) ** self.order).sum(axis=-3).unsqueeze(-3) 
        x = fftshift(ifft2(rhs/lhs), dim=(-2,-1)).real
        
        return crop_half(x)


class WienerNet(nn.Module):
    def __init__(self,  nc=[16, 32, 64, 128]):
        super(WienerNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.wiener = Wiener_Batched(lam=0.2)
        self.denoiser = ResUNet(in_nc=1, out_nc=1, nb=2, nc=nc)
        
    def forward(self, y, psf):

        x = self.wiener(y, psf)
        x = self.denoiser(x)
        
        return x 


if __name__ == '__main__':
    for nc in [8, 16, 32]:
        model = WienerNet(nc=[nc, nc*2, nc*4, nc*8])
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %s  (nc=%s)" % (total, nc))
    