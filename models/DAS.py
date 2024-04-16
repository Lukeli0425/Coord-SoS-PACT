import torch
from torch import nn


class DAS(nn.Module):
    """Delay-And-Sum image reconstruction module for Photoacoustic Computed Tomography with ring array ."""
    def __init__(self, R_ring, N_transducer, T_sample, x_vec, y_vec, angle_range=(0, 2*torch.pi), mode='zero', clip=False, device='cuda:0'):
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
        self.device = device
        self.R_ring = torch.tensor(R_ring, device=self.device)
        self.N_transducer = N_transducer
        self.T_sample = torch.tensor(T_sample, device=self.device)
        self.H, self.W = x_vec.shape[0], y_vec.shape[0]
        self.x_vec = torch.tensor(x_vec, device=self.device).view(1, -1, 1)
        self.y_vec = torch.tensor(y_vec, device=self.device).view(1, 1, -1)
        self.mode = mode
        self.clip = clip
        
        # angle_transducer = 2 * torch.pi / self.N_transducer * (torch.arange(self.N_transducer, device=self.device) + 1).view(-1, 1, 1)
        angle_transducer = (torch.linspace(angle_range[0], angle_range[1], self.N_transducer, device=self.device) + (angle_range[1] - angle_range[0])/self.N_transducer).view(-1, 1, 1)
        self.x_transducer = self.R_ring * torch.cos(angle_transducer - torch.pi)
        self.y_transducer = self.R_ring * torch.sin(angle_transducer - torch.pi)
        self.distance_to_transducer = torch.sqrt((self.x_transducer - self.x_vec)**2 + (self.y_transducer - self.y_vec)**2)
        self.id_transducer = torch.arange(self.N_transducer).view(self.N_transducer,1,1).repeat(1,self.H,self.W)

    def forward(self, sinogram, v0, d_delay, ring_error):
        if sinogram.shape[0] != self.N_transducer:
            raise ValueError('Invalid number of transducer in the sinogram.')
        
        if self.mode == 'zero':
            sinogram[:, 0] = 0
            sinogram[:, -1] = 0
        
        id_time = torch.round((self.distance_to_transducer + ring_error - d_delay) / (v0 * self.T_sample)).int()
        if self.clip:
            id_time = id_time.maximum(torch.zeros_like(id_time)).minimum(torch.ones_like(id_time) * (sinogram.shape[1]-1))
       
        return sinogram[self.id_transducer, id_time].mean(0)
    
    

class DAS_dual(nn.Module):
    """Delay-And-Sum image reconstruction module using a dual SoS distribution for Photoacoustic Computed Tomography with ring array ."""
    def __init__(self, R_ring, N_transducer, T_sample, x_vec, y_vec, angle_range=(0, 2*torch.pi), R_body=0.0, center=(0.0, 0.0), mode='zero', clip=False, device='cuda:0'):
        """Initialize parameters of the Dual SoS Delay-And-Sum module.

        Args:
            R_ring (`float`): Raduis of the ring transducer array.
            N_transducer (`int`): Number of uniformly distributed transducers in the ring array.
            T_sample (`float`): Sample time interval [s].
            x_vec (`numpy.ndarray`): X coordinates of the image grid.
            y_vec (`numpy.ndarray`): Y coordinates of the image grid.
            mode (`str`, optional): _description_. Defaults to `zero`.
            clip (`bool`, optional): Whether to clip the time index into the time range of the sinogram. Defaults to `False` to accelerate the reconstruction.
            R_body (`float`, optional): Radius of the circular body. Defaults to `0.0`.
            center (`tuple`, optional): Center coordinates [m] of the circular body. Defaults to `(0.0, 0.0)`.
        """
        super(DAS_dual, self).__init__()
        self.device = device
        self.R_ring = torch.tensor(R_ring, device=self.device)
        self.N_transducer = N_transducer
        self.T_sample = torch.tensor(T_sample, device=self.device)
        self.H, self.W = x_vec.shape[0], y_vec.shape[0]
        x_vec = torch.tensor(x_vec, device=self.device).view(1, -1, 1).repeat(1,1,self.W)
        y_vec = torch.tensor(y_vec, device=self.device).view(1, 1, -1).repeat(1,self.H,1)
        self.mode = mode
        self.clip = clip
 
        # angle_transducer = 2 * torch.pi / self.N_transducer * (torch.arange(self.N_transducer, device=self.device) + 1).view(-1, 1, 1)
        angle_transducer = (torch.linspace(angle_range[0], angle_range[1], self.N_transducer, device=self.device) + (angle_range[1] - angle_range[0])/self.N_transducer).view(-1, 1, 1)
        x_transducer = self.R_ring * torch.cos(angle_transducer - torch.pi)
        y_transducer = self.R_ring * torch.sin(angle_transducer - torch.pi)
        self.distance_to_transducer = torch.sqrt((x_transducer - x_vec)**2 + (y_transducer - y_vec)**2)
        angle_points = torch.arctan2(y_vec - center[0], x_vec - center[1]) + torch.pi
        angle_to_transducer = torch.arctan2(y_transducer - y_vec, x_transducer - x_vec)
        r = torch.sqrt((x_vec - center[0])**2 + (y_vec - center[1])**2).repeat(self.N_transducer, 1, 1)
        self.distance_in_body = torch.zeros_like(self.distance_to_transducer)
        self.distance_in_body[r < R_body] = (torch.sqrt(R_body**2 - (r*torch.sin(angle_points - angle_to_transducer))**2) + r * torch.cos(angle_points - angle_to_transducer))[r<R_body]
        self.distance_in_body[r >= R_body] = (2 * torch.sqrt(torch.maximum(R_body**2 - (r*torch.sin(angle_points - angle_to_transducer))**2, torch.zeros_like(angle_to_transducer))) * (torch.cos(angle_points - angle_to_transducer) >= 0))[r >= R_body]
        self.id_transducer = torch.arange(self.N_transducer).view(self.N_transducer,1,1).repeat(1,self.H,self.W)

    def forward(self, sinogram, v0, v1, d_delay, ring_error):
        if sinogram.shape[0] != self.N_transducer:
            raise ValueError('Invalid number of transducer in the sinogram.')
        
        if self.mode == 'zero':
            sinogram[:, 0] = 0
            sinogram[:, -1] = 0
        
        id_time = torch.round(((self.distance_to_transducer - self.distance_in_body + ring_error - d_delay) / v0 + self.distance_in_body / v1 ) / self.T_sample).int()
        if self.clip:
            id_time = id_time.maximum(torch.zeros_like(id_time)).minimum(torch.ones_like(id_time) * (sinogram.shape[1]-1))
       
        return sinogram[self.id_transducer, id_time].mean(0)