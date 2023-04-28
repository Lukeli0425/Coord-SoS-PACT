import numpy as np
import torch
import torch.fft
import torch.nn as nn


def fftn(x):
	x_fft = torch.fft.fftn(x, dim=[2,3])
	return x_fft


def ifftn(x):
	return torch.fft.ifftn(x, dim=[2,3])


def conv_fft_batch(H, x):
	"""Batched version of FFT convolution using PyTorch."""
	Y_fft = fftn(x) * H
	y = ifftn(Y_fft).real
	return y


def psf_to_otf(ker, size, device):
	
	psf = torch.zeros(size)
	# circularly shift

	center = (ker.shape[2] + 1) // 2
	psf[:, :, :center, :center] = ker[:, :, center:, center:]
	psf[:, :, :center, -center:] = ker[:, :, center:, :center]
	psf[:, :, -center:, :center] = ker[:, :, :center, center:]
	psf[:, :, -center:, -center:] = ker[:, :, :center, :center]

	# otf = torch.rfft(psf, 3, onesided=False)
	H = torch.fft.fftn(psf, dim=[-2,-1]).to(device)
	Ht, HtH = torch.conj(H), torch.abs(H)**2

	return psf, H, Ht, HtH


def get_fourier_coord(n_points=128, l=3.2e-3, device='cuda:0'):
	fx1D = torch.linspace(-np.pi/l, np.pi/l, n_points, requires_grad=False)
	fy1D = torch.linspace(-np.pi/l, np.pi/l, n_points, requires_grad=False)
	[fx2D, fy2D] = torch.meshgrid(fx1D, fy1D)
	k2D = torch.sqrt(fx2D**2 + fy2D**2) * n_points
	theta2D = torch.arctan2(fy2D, fx2D)
 
	return k2D.to(device), theta2D.to(device)