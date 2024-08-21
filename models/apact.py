import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift
from tqdm import tqdm

from models.pact import Fourier_Series, Wavefront_Fourier, Wavefront_SOS
from models.regularizer import Total_Squared_Variation, Total_Variation
from models.sos import SOS_Rep
from utils.reconstruction import get_gaussian_window
from utils.utils_torch import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, x, y):
        return torch.mean((x-y) ** 2, axis=(-3,-2,-1))


class TF_APACT(nn.Module):
    def __init__(self, delays, N, l):
        super(TF_APACT, self).__init__() 
        self.n_delays = delays.shape[0]
        self.delays = delays.view(self.n_delays,1,1).cuda()
        self.k2D, self.theta2D = get_fourier_coord(N=N, l=l)
        self.k2D = self.k2D.cuda().unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        self.theta2D = self.theta2D.cuda().unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        
    def forward(self, dc, x, y):
        w = dc + x * torch.cos(2 * self.theta2D) + y * torch.sin(2 * self.theta2D) # Wavefront function.
        tf = (torch.exp(-1j*self.k2D*(self.delays - w)) + torch.exp(1j*self.k2D*(self.delays - w))) / 2
        return ifftshift(tf, dim=(-2,-1))
 
    
class APACT(nn.Module):
    def __init__(self, delays, R_body, v0, Nx, Ny, dx, dy, x_vec, y_vec, lam_tv, mean, std, dc_range, amp, step, N_patch=80, fwhm=1.5e-3,
                 generate_TF=True, 
                 data_path='./TFs'):
        super(APACT, self).__init__() 
        self.logger = logging.getLogger('APACT')
        
        self.data_path = data_path
        self.TF_dir = os.path.join(data_path, 'TFs')
        os.makedirs(self.TF_dir, exist_ok=True)
        
        # Wavefront sampling parameters
        self.N = None
        self.dc_range = dc_range
        self.amp = amp
        self.step = step
        self.delays = delays
        self.k, _ = get_fourier_coord(N=160, l=6.4e-3)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        sigma = fwhm / 4e-5 / np.sqrt(2*np.log(2))
        self.gaussian_window = torch.tensor(get_gaussian_window(sigma, N_patch)).unsqueeze(0).cuda()
        
        # Forawrd models and dconvolution models
        self.tf = TF_APACT(delays=self.delays, N=160, l=6.4e-3)
        self.loss = MSELoss()
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        # SOS reconstruction params.
        self.v0 = v0
        self.Nx, self.Ny, self.dx, self.dy = Nx, Ny, dx, dy
        self.x_vec, self.y_vec = torch.tensor(x_vec).cuda(), torch.tensor(y_vec).cuda()
        XX, YY = torch.meshgrid(self.x_vec[:,0], self.y_vec[:,0], indexing='xy')
        XX, YY = XX.cuda(), YY.cuda()
        self.SOS_mask = torch.zeros_like(XX).cuda()
        self.SOS_mask[XX**2 + YY**2 <= R_body**2] = 1
        # self.XX, self.YY = XX[self.SOS_mask>0].view(1,-1), YY[self.SOS_mask>0].view(1,-1)
        # self.theta = nn.Parameter(0.001*torch.randn(size=(self.SOS_mask.sum().int(), 1), dtype=torch.float64).cuda(), requires_grad=True)
        # self.SOS = torch.ones_like(self.SOS_mask).cuda() * self.v0
        self.tv_reg = Total_Squared_Variation(weight=lam_tv)
        self.wf_SOS = Wavefront_SOS(R_body=R_body, v0=v0, x_vec=x_vec, y_vec=y_vec, n_points=360, N_int=500)
        self.fourier_series = Fourier_Series()
        self.SOS = SOS_Rep(mode='None', mask=self.SOS_mask, v0=v0, mean=mean, std=std,
                           hidden_features=64, hidden_layers=1, pos_encoding=False)
        
        self.TFs, self.params, self.best_params = None, None, []
        if generate_TF:
            self.generate_TFs()
            self.load_params()
        else:
            try:
                self.load_params()
            except:
                self.logger.info(' Failed loading wavefront parameters.')
                self.generate_TFs()
                self.load_params()
            
            
    def load_params(self):
        """Load the precalculated wavefront parameters from `self.data_path`."""
        self.params = torch.load(os.path.join(self.data_path, 'params.pth')).cuda()
        self.N = self.params.shape[0]
        self.logger.info(' Successfully loaded transfer functions.')

    
    def generate_TFs(self):
        self.logger.info(' Generating transfer functions...')
        dcs = np.arange(self.dc_range[0], self.dc_range[1], self.step)
        xs = np.arange(-self.amp, self.amp, self.step)
        ys = np.arange(-self.amp, self.amp, self.step)
        params = []
        idx = 0
        for dc in dcs:
            for x in xs:
                for y in ys:
                    params.append((dc, x, y))
                    torch.save(self.tf(dc, x, y), os.path.join(self.TF_dir, f'TF_{idx}.pth'))
                    idx += 1
        torch.save(torch.tensor(params), os.path.join(self.data_path, 'params.pth'))
    
    def forward(self, patch_stack):
        best_loss, best_x = torch.inf, None
        best_tf, best_params = None, None
        
        # Apply Gaussian window to image patch.
        patch_stack = patch_stack * self.gaussian_window
        Y = fft2(ifftshift(pad_double(patch_stack), dim=(-2,-1)))
        
        for idx in range(self.N):
            H = torch.load(os.path.join(self.TF_dir, f'TF_{idx}.pth')).cuda()
            Ht, HtH = H.conj(), H.abs() ** 2
            rhs = (Y * Ht).sum(axis=-3).unsqueeze(-3)
            lhs = HtH.sum(axis=-3).unsqueeze(-3)
            X = rhs / lhs
            loss = self.loss(Y.abs() * self.k, (H * X).abs() * self.k)
            if loss < best_loss:
                best_loss = loss.item()
                best_x = crop_half(fftshift(ifft2(X), dim=(-2,-1))).real
                best_tf = H
                best_params = self.params[idx]
        self.best_params.append(best_params)
        
        return best_x, best_tf, best_params, best_loss
    
    def save_wavefront_params(self):
        self.best_params = torch.concatenate(self.best_params, dim=0)
        torch.save(self.best_params, os.path.join(self.data_path, 'best_params.pth'))
        self.logger.info(' Wavefront params saved to "%s".', os.path.join(self.data_path, 'best_params.pth'))
    
    def prepare_SOS_reconstruction(self, patch_centers, wf_params):
        self.A = torch.zeros(3*len(patch_centers), self.SOS_mask.sum().int()).cuda().double()
        # b = torch.load(os.path.join(self.data_path, 'best_params.pth')).view(-1,1).cuda()
        b = wf_params.view(-1,1).cuda()
        b /= self.dx*self.dy
        
        self.x_p, self.y_p = patch_centers[:,0].view(-1,1), patch_centers[:,1].view(-1,1)
        x, y = self.XX-self.x_p+self.dx/2, self.YY-self.y_p+self.dy/2
        self.A[0::3,:] = 1 / torch.sqrt(x**2+y**2) / (2*torch.pi)
        self.A[1::3,:] = (x**2-y**2) / (torch.sqrt(x**2+y**2)**3) / torch.pi
        self.A[2::3,:] = (2*x*y) / (torch.sqrt(x**2+y**2)**3) / torch.pi
        self.ATb = self.A.T @ b
        W = torch.exp(-((self.x_p-self.x_p.T)**2+(self.y_p-self.y_p.T)**2)/2/self.dx/3)
        W = W / W.sum(dim=1)
        self.C = torch.zeros((3*W.shape[0], 3*W.shape[1]), dtype=torch.float64).cuda()
        self.C[0::3,0::3] = W
        self.C[1::3,1::3] = W
        self.C[2::3,2::3] = W
        self.ATC = self.A.T @ self.C
        # print(W.shape, self.C.shape, W.sum())
        
    def reconstruct_SOS(self):
        SOS = torch.ones_like(self.SOS_mask, dtype=torch.float64).cuda() * self.v0
        SOS[self.SOS_mask > 0] = self.v0 / (1-self.theta.view(-1))
        loss = self.loss_fn(self.ATC @ (self.A@self.theta), self.ATb)  # + self.tv_reg(SOS, self.SOS_mask) 
        return loss, SOS, self.theta
    
    def optimize_SOS(self, x, y, fourier_params):
        SOS = self.SOS()
        thetas, wf_SOS = self.wf_SOS(x, y, SOS)
        fourier_params1 = self.fourier_series(wf_SOS, thetas)
        # print(fourier_params.shape, fourier_params.device, fourier_params1.shape, fourier_params1.device)
        # print(fourier_params.shape, fourier_params1.shape)
        loss = self.loss_fn(fourier_params, fourier_params1) + self.tv_reg(SOS, self.SOS_mask)
        return loss, SOS

if __name__ == '__main__':
    delays = np.arange(-8e-4, 8e-4, 1e-5)
    apact = APACT(delays)
    # apact.generate_TFs()
    apact.load_TFs()
    print(apact.TFs.shape)
    print(apact.params.shape)
    print(apact.TFs[0].shape)
    