import numpy as np
import torch
import torch.nn as nn
from torch.fft import fftn, fftshift, ifftn, ifftshift


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


def get_fourier_coord(n_points=64, l=3.2e-3, device='cuda:0'):
	fx1D = torch.linspace(-np.pi/(3.2e-3), np.pi/(3.2e-3), n_points)
	fy1D = torch.linspace(-np.pi/(3.2e-3), np.pi/(3.2e-3), n_points)
	[fx2D, fy2D] = torch.meshgrid(fx1D, fy1D, indexing='xy')
	k2D = torch.sqrt(fx2D**2 + fy2D**2) * n_points
	theta2D = torch.arctan2(fy2D, fx2D) + np.pi/2 # Add `np.pi/2` to match the polar definition of the theta.
 
	return k2D.to(device), theta2D.to(device)