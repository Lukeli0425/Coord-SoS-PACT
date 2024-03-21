import torch
from torch import nn


class DAS(nn.Module):
    """Delay-And-Sum image reconstruction module Photoacoustic Computed Tomography with ring array ."""
    def __init__(self, R_ring, N_transducer, T_sample, x_vec, y_vec, mode='zero', clip=False):
        """Initialize parameters of the Delay-And-Sum module.

        Args:
            R_ring (`float`): Raduis of the ring transducer array.
            N_transducer (`int`): Number of uniformly distributed transducers in the ring array.
            T_sample (`float`): Sample time interval [s].
            x_vec (`numpy.ndarray`): X coordinates of the image grid.
            y_vec (`numpy.ndarray`): Y coordinates of the image grid.
            mode (str, optional): _description_. Defaults to `zero`.
            clip (bool, optional): Whether to clip the time index into the time range of the sinogram. Defaults to `False` to accelerate the reconstruction.
        """
        super(DAS, self).__init__()
        
        self.R_ring = torch.tensor(R_ring)
        self.N_transducer = N_transducer
        self.T_sample = torch.tensor(T_sample)
        self.x_vec = torch.tensor(x_vec).view(1, -1, 1)
        self.y_vec = torch.tensor(y_vec).view(1, 1, -1)
        self.H, self.W = self.x_vec.shape[1], self.y_vec.shape[2]
        self.mode = mode
        self.clip = clip
        
        angle_transducer = 2 * torch.pi / self.N_transducer * (torch.arange(self.N_transducer) + 1).view(-1, 1, 1)
        self.x_transducer = self.R_ring * torch.cos(angle_transducer - torch.pi)
        self.y_transducer = self.R_ring * torch.sin(angle_transducer - torch.pi)
        self.distance_to_transducer = torch.sqrt((self.x_transducer - self.x_vec)**2 + (self.y_transducer - self.y_vec)**2)
        self.id_transducer = torch.arange(self.N_transducer).view(self.N_transducer,1,1).repeat(1,self.H,self.W)

    def forward(self, sinogram, v0, d_delay=torch.zeros(1), ring_error=torch.zeros(1)):
        if sinogram.shape[0] != self.N_transducer:
            raise ValueError('Invalid number of transducer in the sinogram.')
        
        if self.mode == 'zero':
            sinogram[:, 0] = 0
            sinogram[:, -1] = 0
        
        id_time = torch.round((self.distance_to_transducer + ring_error - d_delay) / (v0 * self.T_sample)).int()
        if self.clip:
            id_time = id_time.maximum(torch.zeros_like(id_time)).minimum(torch.ones_like(id_time) * (sinogram.shape[1]-1))
       
        return sinogram[self.id_transducer, id_time].mean(0)