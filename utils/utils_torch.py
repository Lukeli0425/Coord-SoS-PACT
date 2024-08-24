import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftn, fftshift, ifftn, ifftshift


def pad_double(img:torch.Tensor) -> torch.Tensor:
    H, W = img.shape[-2:]
    return F.pad(img, (W//2, W//2, H//2, H//2))


def crop_half(img:torch.Tensor) -> torch.Tensor:
    H, W = img.shape[-2:]
    return img[...,H//4:3*H//4, W//4:3*W//4]
    
    
def conv_fft_batch(H, x):
	"""Batched version 2D convolution using FFT."""
	Y_fft = fftn(x, dim=[-2,-1]) * H
	y = ifftshift(ifftn(Y_fft, dim=[-2,-1]), dim=[-2,-1]).real
	return y


def psf_to_otf(psf):
	
	# psf = torch.zeros(size)

	# center = (ker.shape[2] + 1) // 2
	# psf[:, :, :center, :center] = ker[:, :, center:, center:]
	# psf[:, :, :center, -center:] = ker[:, :, center:, :center]
	# psf[:, :, -center:, :center] = ker[:, :, :center, center:]
	# psf[:, :, -center:, -center:] = ker[:, :, :center, :center]
	# psf = ker
	H = fftn(psf, dim=[-2,-1])
	Ht, HtH = torch.conj(H), torch.abs(H) ** 2

	return psf, H, Ht, HtH


def get_fourier_coord(N:int=80, l:float=3.2e-3, device:str='cuda:0') -> tuple:
	fx1D = torch.linspace(-np.pi/l, np.pi/l, N, requires_grad=False, device=device)
	fy1D = torch.linspace(-np.pi/l, np.pi/l, N, requires_grad=False, device=device)
	[fx2D, fy2D] = torch.meshgrid(fx1D, fy1D, indexing='xy')
	k2D = torch.sqrt(fx2D**2 + fy2D**2) * N
	theta2D = torch.arctan2(fy2D, fx2D) + np.pi/2 # Add `np.pi/2` to match the polar definition of the theta.
	return k2D, theta2D % (2*torch.pi)


def get_mgrid(shape:tuple, range:tuple=(-1, 1)) -> torch.Tensor:
    """Generates a flattened grid of (x,y,...) coordinates in a range of `[-1, 1]`.
    
    Args:
        shape (`tuple`): Shape of the datacude to be fitted.
        range (`tuple`, optional): Range of the grid. Defaults to `(-1, 1)`.

    Returns:
        `torch.Tensor`: Generated flattened grid of coordinates.
    """
    tensors = [torch.linspace(range[0], range[1], steps=N) for N in shape]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='xy'), dim=-1)
    mgrid = mgrid.reshape(-1, len(shape))
    return mgrid


def get_total_params(model: torch.nn.Module) -> int:
    """Calculate the total number of parameters in a model.
	
	Args:
		model (`torch.nn.Module`): PyTorch model.

	Returns:
		`int`: Total number of parameters in the model.
	"""
    return sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])