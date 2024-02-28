import os
import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftn, fftshift, ifftn, ifftshift
import logging

from models.WienerNet import Wiener_Batched
from utils.utils_torch import get_fourier_coord


class TF_APACT(nn.Module):
    def __init__(self, delays, n_points=80, l=3.2e-3, device='cpu'):
        super(TF_APACT, self).__init__() 
        self.device = device
        self.n_delays = delays.shape[0]
        self.delays = torch.tensor(delays).view(1,self.n_delays,1,1).to(self.device)
        self.n_points = n_points # Size of PSF image in pixels.
        self.l = l # Length [m] of the PSF image.
        
    def forward(self, dc, x, y):
        k, theta = get_fourier_coord(n_points=self.n_points, l=self.l, device=self.device)
        k, theta = k.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1), theta.unsqueeze(0).unsqueeze(0).repeat(1,self.n_delays,1,1)
        w = lambda theta: dc + x * torch.cos(2 * theta) + y * torch.sin(2 * theta) # Wavefront function.
        tf = (torch.exp(-1j*k*(self.delays - w(theta))) + torch.exp(1j*k*(self.delays - w(theta+np.pi)))) / 2
        
        return tf
    
    
class APACT(nn.Module):
    def __init__(self, delays, lam=0.03, data_path='./TFs', device='cuda:0'):
        super(APACT, self).__init__() 
        self.logger = logging.getLogger('APACT')
        self.device = device
        
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        # Wavefront sampling parameters
        self.dc_range = (-2e-4, 1.6e-4)
        self.amp = 3.2e-4
        self.step = 4e-5
        self.delays = delays
        
        # Forawrd models and dconvolution models
        self.tf = TF_APACT(delays=self.delays)
        self.deconv = Wiener_Batched(lam=lam)
        self.loss = nn.MSELoss()
        
        self.TFs, self.params = None, None
        try:
            self.load_TFs()
        except:
            self.logger.info('Failed loading transfer functions.')
            self.generate_TFs()
            self.load_TFs()
            
            
    def load_TFs(self):
        """Load the precaculated transfer functions from the data_path."""
        self.TFs = torch.from_numpy(np.load(os.path.join(self.data_path, 'TFs.npy'))).to(self.device)
        self.params = torch.from_numpy(np.load(os.path.join(self.data_path, 'params.npy'))).to(self.device)
        self.logger.info('Successfully loaded transfer functions.')

    
    def generate_TFs(self):
        self.logger.info('Generating transfer functions...')
        dcs = np.arange(self.dc_range[0], self.dc_range[1], self.step)
        xs = np.arange(-self.amp, self.amp, self.step)
        ys = np.arange(-self.amp, self.amp, self.step)
        tfs, params = [], []
        for dc in dcs:
            for x in xs:
                for y in ys:
                    tfs.append(self.tf(dc, x, y))
                    params.append((dc, x, y))
                    
        np.save(os.path.join(self.data_path, 'TFs.npy'), np.array(tfs))
        np.save(os.path.join(self.data_path, 'params.npy'), np.array(params))
    
    def forward(self, y):
        best_loss, best_x = torch.inf, None
        best_tf, best_params = None, None
        
        for tf, params in zip(self.TFs, self.params):
            psf = ifftshift(ifft2(tf), dim=[-2,-1]).abs()
            psf /= psf.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1) # Normalization.
            x = self.deconv(y, psf)
            y_hat = ifftshift(ifft2(psf * fft2(y)), dim=[-2,-1]).abs()
            loss = self.loss(y, y_hat)
            if loss < best_loss:
                best_loss = loss
                best_x = x
                best_tf = tf
                best_params = params
                
        return best_x, best_tf, best_params, best_loss
    

if __name__ == '__main__':
    delays = np.arange(-8e-4, 8e-4, 1e-5)
    apact = APACT(delays)
    # apact.generate_TFs()
    apact.load_TFs()
    print(apact.TFs.shape)
    print(apact.params.shape)
    print(apact.TFs[0].shape)
    