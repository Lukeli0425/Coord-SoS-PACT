from netrc import NetrcParseError
import numba
import numpy as np
from numpy.fft import fft, fft2, fftshift, ifft, ifft2, ifftshift
import math


def deconvolve_sinogram(sinogram, EIR):
    """Correct EIR phase of the sinogram.

    Args:
        sinogram (`numpy.array`): _description_
        EIR (`numpy.array`): EIR of the transducer.

    Returns:
        `numpy.array`: The corrected sinogrm.
    """
    delta = np.zeros_like(EIR)
    delta[0, np.argmax(EIR)] = 1
    delta_ft = fft(delta, axis=1)
    EIR_ft = fft(EIR, axis=1)
    Sinogram_ft = fft(sinogram, axis=1)
    Sinogram_ft *= np.exp(1j * (np.angle(delta_ft) - np.angle(EIR_ft)))
    
    return np.real((ifft(Sinogram_ft, axis=1)))


def get_delays(R, v0, v1, n_delays, mode='uniform'):
    if mode == 'uniform':
        return np.linspace((1-v0/v1) * R * 0, (1-v0/v1) * R * 1.2, n_delays)
    elif mode == 'quadric':
        return (1-v0/v1) * R * np.sqrt(np.linspace(0,1,n_delays))
    else:
        raise NotImplementedError()


@numba.jit(nopython=True) 
def delay_and_sum(R_ring, T_sample, v0, sinogram, x_vec, y_vec, d_delay=0, ring_error=0):
    """Generate a 2D delay-and-sum recontructed PACT image of ring transducer array. This function is accelerated by `numba.jit` on a GPU.

    Args:
        R_ring (`float`): The R_ring [m] of the ring transducer array.
        T_sample (`float`): Sample time interval [s] of the signals.
        v0 (`float`): The sound speed [m/s] used in delay-and-sum recontruction.
        sinogram (`numpy.ndarray`): A 2D array and each column of it is the signal recievde by one transducer. The nummber of transducers should be the number of columns. The transducers should be evenly distributed on a circle in counterclockwise arrangement and the first column correspond to the transducer in the dirrection `2pi/N` in the first quartile. The first sample should be at time 0 when the photoacoustic effect happens.
        x_vec (`numpy.ndarray`): The vector [m] defining the x coordinates of the grid points on which the recontruction is done. The values in the vector should be unifromly-spaced in ascending order. The origin of the cartesian coordinate system is the center of the ring array.
        y_vec (`numpy.ndarray`): The vector [m] defining the y coordinates of the grid points on which the recontruction is done. The values in the vector should be unifromly-spaced in ascending order. The origin of the cartesian coordinate system is the center of the ring array.
        d_delay (`float`): The delay distance [m] of the signals used in DAS. The default value is 0.
        ring_error (`numpy.ndarray`): The radial displacement error of the transducers. The default value is 0.

    Returns:
        `numpy.ndarray`: A 2D array of size `(len(y_vec), len(x_vec))`. `Image[t, s]` is the recontructed photoacoustic amplitude at the grid point `(x_vec[s], y_vec[t])`.
    """
    H, W = len(x_vec), len(y_vec)
    N_transducer = sinogram.shape[0]
    Image = np.zeros((len(x_vec), len(y_vec)))
    delta_angle = 2 * np.pi / N_transducer
    angle_transducer = delta_angle * (np.arange(N_transducer,) + 1)
    x_transducer = R_ring * np.sin(angle_transducer - np.pi)
    y_transducer = R_ring * np.cos(angle_transducer - np.pi)
    
    related_data = np.zeros((N_transducer,))
    
    for s in range(H):
        for t in range(W):
            distance_to_transducer = np.sqrt((x_transducer - x_vec[s])**2 + (y_transducer - y_vec[t])**2) - d_delay + ring_error
            for k in range(N_transducer):
                id = math.floor(distance_to_transducer[k] / (v0 * T_sample))
                if id > 0 and id <= sinogram.shape[1]:
                    related_data[k] = sinogram[k, id]
                else:
                    related_data[k] = 0.0
            Image[t, s] = related_data.mean()
    return Image

def gaussian_kernel(sigma, size):
    function = lambda x, y: np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*(sigma**2)))
    kernel = np.fromfunction(function, (size, size), dtype=float)
    return kernel / np.sum(kernel)