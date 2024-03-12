import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifft2, ifftn, ifftshift

# from models.WienerNet import Wiener_Batched
from utils.utils_torch import get_fourier_coord

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, x, y):
        return torch.mean((x-y) ** 2, axis=(-3,-2,-1))


class TF_APACT(nn.Module):
    def __init__(self, delays, n_points=80, l=3.2e-3, device='cpu'):
        super(TF_APACT, self).__init__() 
        self.device = device
        self.n_delays = delays.shape[0]
        self.delays = torch.tensor(delays).view(self.n_delays,1,1).to(self.device)
        self.n_points = n_points # Size of PSF image in pixels.
        self.l = l # Length [m] of the PSF image.
        
    def forward(self, dc, x, y):
        k, theta = get_fourier_coord(n_points=self.n_points, l=self.l, device=self.device)
        k, theta = k.unsqueeze(0).repeat(self.n_delays,1,1), theta.unsqueeze(0).repeat(self.n_delays,1,1)
        w = lambda theta: dc + x * torch.cos(2 * theta) + y * torch.sin(2 * theta) # Wavefront function.
        tf = (torch.exp(-1j*k*(self.delays - w(theta))) + torch.exp(1j*k*(self.delays - w(theta+np.pi)))) / 2
        
        return ifftshift(tf, dim=(-2,-1))


class Deconv_APACT(nn.Module):
    def __init__(self):
        super(Deconv_APACT, self).__init__() 
        
    def forward(self, y, tf):
        
        
        return
 
    
class APACT(nn.Module):
    def __init__(self, delays, generate_TF=True, dc_range=(-2e-4, 1.6e-4), amp=3.2e-4, step=4e-5, data_path='./TFs', device='cuda:0'):
        super(APACT, self).__init__() 
        self.logger = logging.getLogger('APACT')
        self.device = device
        
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        # Wavefront sampling parameters
        self.N = None
        self.dc_range = dc_range
        self.amp = amp
        self.step = step
        self.delays = delays
        self.k, self.theta = get_fourier_coord(device=self.device)
        self.k = ifftshift(self.k, dim=(-2,-1)).unsqueeze(0).unsqueeze(0)
        
        # Forawrd models and dconvolution models
        self.tf = TF_APACT(delays=self.delays)
        # self.deconv = Deconv_APACT()
        self.loss = MSELoss()
        
        self.TFs, self.params = None, None
        if generate_TF:
            self.generate_TFs()
            self.load_params()
        else:
            try:
                self.load_params()
            except:
                self.logger.info('Failed loading wavefront parameters.')
                self.generate_TFs()
                self.load_params()
            
            
    def load_params(self):
        """Load the precalculated wavefront parameters from `self.data_path`."""
        # self.TFs = torch.from_numpy(np.load(os.path.join(self.data_path, 'TFs.npy'))).to(self.device)
        self.params = torch.from_numpy(np.load(os.path.join(self.data_path, 'params.npy'))).to(self.device)
        self.N = self.params.shape[0]
        self.logger.info('Successfully loaded transfer functions.')

    
    def generate_TFs(self):
        self.logger.info('Generating transfer functions...')
        dcs = np.arange(self.dc_range[0], self.dc_range[1], self.step)
        xs = np.arange(-self.amp, self.amp, self.step)
        ys = np.arange(-self.amp, self.amp, self.step)
        tfs, params = [], []
        idx = 0
        for dc in dcs:
            for x in xs:
                for y in ys:
                    # tfs.append(self.tf(dc, x, y))
                    params.append((dc, x, y))
                    np.save(os.path.join(self.data_path, f'TF_{idx}.npy'), self.tf(dc, x, y))
                    idx += 1
        # np.save(os.path.join(self.data_path, 'TFs.npy'), np.array(tfs))
        np.save(os.path.join(self.data_path, 'params.npy'), np.array(params))
    
    def forward(self, y):
        best_loss, best_x = torch.inf, None
        best_tf, best_params = None, None
        
        Y = fft2(ifftshift(y, dim=(-2,-1)))
        
        for idx in range(self.N):
            H = torch.from_numpy(np.load(os.path.join(self.data_path, f'TF_{idx}.npy'))).unsqueeze(0).to(self.device)
            Ht, HtH = H.conj(), H.abs() ** 2
            rhs = (Y * Ht).sum(axis=-3).unsqueeze(-3)
            lhs = HtH.sum(axis=-3).unsqueeze(-3)
            X = rhs / lhs

            loss = self.loss(Y.abs() * self.k, (H * X).abs() * self.k)

            if loss < best_loss:
                best_loss = loss
                best_x = fftshift(ifft2(X), dim=(-2,-1)).real
                best_tf = H
                best_params = self.params[idx]
        
        
        # H, Ht = self.TFs, self.TFs.conj()
        # rhs = (Ht * Y).sum(axis=-3).unsqueeze(-3)
        # lhs = (Ht * H).sum(axis=-3).unsqueeze(-3) + 0.002
        # X = rhs / lhs
        # Y_hat = H * X
        
        # # y_hat = ifftshift(ifft2(H * X), dim=[-2,-1]).real
        # loss = self.loss(Y.abs().repeat(Y_hat.shape[0],1,1,1) * self.k, Y_hat.abs() * self.k)
        # best_idx, best_loss = torch.argmin(loss), loss.min()   
        # # best_x = ifftshift(ifft2(X[best_idx]), dim=[-2,-1]).real
        # best_x = ifft2(X[best_idx]).real
        # # best_x = x[best_idx]
                
        return best_x, best_tf, best_params, best_loss
    

if __name__ == '__main__':
    delays = np.arange(-8e-4, 8e-4, 1e-5)
    apact = APACT(delays)
    # apact.generate_TFs()
    apact.load_TFs()
    print(apact.TFs.shape)
    print(apact.params.shape)
    print(apact.TFs[0].shape)
    