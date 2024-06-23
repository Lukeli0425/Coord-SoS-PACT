import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftn, ifftshift


def standardize(img):
    """Standardize the image to have zero mean and unit standard deviation.

    Args:
        img (`numpy.ndarray`): Input image.

    Returns:
        `numpy.ndarray`: Standardized image.
    """
    return (img - img.mean()) / img.std()

def normalize(img):
    """Normalize the image to `[0, 1]`.

    Args:
        img (`numpy.ndarray`): Input image.

    Returns:
        `numpy.ndarray`: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())


def PSF(theta, k, w, delay):
    tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+torch.pi)))) / 2
    psf = ifftshift(ifft2(fftshift(tf, dim=[-2,-1])), dim=[-2,-1]).abs()
    psf /= psf.sum(axis=(-2,-1)) # Normalization.
    return psf


def TF(theta, k, w, delay):
    tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+np.pi)))) / 2
    return tf


def condition_number(psf):
    """Calculate the condition number of a PSF.

    Args:
        tf (`numpy.ndarray`): PSF image.

    Returns:
        `float`: Condition number.
    """
    # H = fft2(psf)
    H = psf
    return H.abs().max() / H.abs().min()    