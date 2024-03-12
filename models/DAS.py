import torch
from torch import nn

class DAS(nn.Module):
    def __init__(self, R_ring, T_sample, v0, x_vec, y_vec, d_delay=0, ring_error=0):
        super(DAS, self).__init__()
        self.R_ring = R_ring
        self.T_sample = T_sample
        self.v0 = v0
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.d_delay = d_delay
        self.ring_error = ring_error

    def forward(self, sinogram):
        N_transducer, H, W = sinogram.shape[0], self.x_vec.shape[0], self.y_vec.shape[0]
        
        angle_transducer = 2 * torch.pi / N_transducer * (torch.arange(N_transducer) + 1)
        x_transducer = self.R_ring * torch.sin(angle_transducer - torch.pi)
        y_transducer = self.R_ring * torch.cos(angle_transducer - torch.pi)
        distance_to_transducer = torch.sqrt((x_transducer - self.x_vec)**2 + (y_transducer - self.y_vec)**2) - self.d_delay + self.ring_error
        
        related_data = torch.zeros((N_transducer, H, W))
        
        return
        