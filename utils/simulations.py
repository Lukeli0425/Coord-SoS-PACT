import os
from tempfile import gettempdir

import numba
import numpy as np
from sympy import im
import torch
from numpy.fft import fft, ifft
from numpy.random import choice, rand
from torch.fft import fftshift, ifftn

from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from kwave.options import SimulationOptions
from kwave.utils import *

# def read_images(path, files):
#     return [np.load(os.path.join(path, file)) for file in files]


# def subtract_background(image):
#     """Conduct background subtraction to blood vessel section images.

#     Args:
#         image (`numpy.ndarray`): Input blood vessel section image.

#     Returns:
#         `numpy.ndarray`: Background subtracted blood vessel section image.
#     """
#     return image - image[0,0]


# def join_images(images, n_row=2, n_col=2):
    
#     if len(images) != n_row * n_col:
#         raise ValueError('Number of images does not match the number of rows and columns.')

#     images_row = []
#     for i in range(n_row):
#         images_row.append(np.concatenate(images[n_row*i:n_row*i+n_col], axis=1))
#     image_joint = np.concatenate(images_row , axis=0)
    
#     return image_joint


# def split_images(images, img_size=(64, 64), step=(32, 32)):
    
#     n_x, n_y = (images[0].shape[0]-img_size[0])//step[0]+1, (images[0].shape[1]-img_size[1])//step[1]+1
#     images_out = [[] for i in range(n_x*n_y)]
#     for image in images:
#         for idx in range(2*n_x-1):
#             for idy in range(2*n_y-1):
#                 images_out[n_x*idx+idy].append(image[idx*step[0]:idx*step[0]+img_size[0], idy*step[1]:idy*step[1]+img_size[1]])
                
#     images_out = [np.array(image) for image in images_out] # Convert each stack to numpy.ndarray.
    
#     return images_out
    
    
def center(img):
    """Calculate the center of an image.

    Args:
        img (`numpy.ndarray`): 2D image.

    Returns:
        `tuple`: Indices of the ceenter pixel.
    """
    img = np.abs(img)
    Nx, Ny = img.shape
    x_sum, y_sum = 0, 0
    for i in range(Nx):
        for j in range(Ny):
            x_sum += i * img[i,j]
            y_sum += j * img[i,j]
    x = int(x_sum / img.sum())
    y = int(y_sum / img.sum())
    return (x, y)


def zero_pad(image, Nx, Ny):
    """Pads image with zeros.

    Args:
        image (`numpy.ndarray`): Input image.
        N (`int`): Image size atter padding.

    Returns:
        `numpy.ndarray`: Padded image.
    """
    
    image_pad = np.zeros((Nx, Ny))
    
    pad_start_x, pad_end_x = (Nx-image.shape[0])//2, (Nx+image.shape[0])//2
    pad_start_y, pad_end_y = (Ny-image.shape[1])//2, (Ny+image.shape[1])//2
    image_pad[pad_start_x:pad_end_x, pad_start_y:pad_end_y] = image
    
    return image_pad, (pad_start_x, pad_end_x), (pad_start_y, pad_end_y)


def random_rotate(img):
    angle = choice([0, 90, 180, 270])
    flip_x, flip_y = choice([True, False]), choice([True, False])
    if flip_x:
        img = np.flip(img, axis=-2)
    if flip_y:
        img = np.flip(img, axis=-1)

    img = np.rot90(img, k=angle//90, axes=(-2,-1))

    return img
    


def get_water_SoS(t):
    """Calculate the speed of sound of water at temperature `t` in Celsius."""
    a = [1.402385e3, 5.038813, -5.799136e-2, 3.287156e-4, -1.398845e-6, 2.787860e-9]
    SoS = 0
    for i in range(len(a)):
        SoS += a[i] * t**i
    return SoS


def reorder_binary_sensor_data(sensor_data, sensor, kgrid, PML_size):
    """Reorder the binary sensor data collected by a ring array in angular order.

    Args:
        sensor_data (`numpy.ndarray`): Input sensor data with shape `[N_transducer, N_T]`.
        sensor (`kSensor`): K-wave sensor object.
        kgrid (`kWaveGrid`): K-wave grid object.
        PML_size (int): Size of PML.

    Returns:
        `numpy.ndarray`: Reordered sensor data with shape `(N_transducer, N_T)`.
    """
    x_sensor = kgrid.x[sensor.mask[PML_size:-PML_size,PML_size:-PML_size] == 1]
    y_sensor = kgrid.y[sensor.mask[PML_size:-PML_size,PML_size:-PML_size] == 1]
    
    angle = np.arctan2(-x_sensor, -y_sensor)
    angle[angle < 0] = 2 * np.pi + angle[angle < 0]
    reorder_index = np.argsort(angle)
    
    return sensor_data[reorder_index]


def transducer_response(sensor_data):
    """Apply transducer response to sinogram.

    Args:
        sensor_data (`numpy.ndarray`): Sinogram with shape `[N_transducer, N_T]`..
        T_sample ('float): Sample time interval of the signals [s].

    Returns:
        `numpy.ndarray`: Output sinogram.
    """

    sensor_data = np.append(sensor_data, np.zeros((sensor_data.shape[0],1)), axis=1)
    return -2 * (sensor_data[:,1:] - sensor_data[:,:-1])


def deconvolve_sinogram(sinogram, EIR, t0):
    delta = np.zeros_like(EIR)
    delta[0, t0] = 1
    
    delta_ft = fft(delta, axis=1)
    EIR_ft = fft(EIR, axis=1)
    Sinogram_ft = fft(sinogram, axis=1)
    
    Sinogram_ft *= np.exp(1j * (np.angle(delta_ft) - np.angle(EIR_ft)))
    
    sinogram_deconv = np.real(ifft(Sinogram_ft, axis=1))
    return sinogram_deconv


def get_medium(kgrid, Nx=2552, Ny=2552, 
               v0=1500.0, v1=1600.0, v2=1650.0, 
               R=0.01, R1=0.06, offset=0.0, rou=1000):
    """
    Generate K-wave medium object with varying SoS distribution.

    Args:
        Nx (int, optional): _description_. Defaults to `2552`.
        Ny (int, optional): _description_. Defaults to `2552`.
        SoS_background (float, optional): SoS of the background medium. [m/s]. Defaults to `1500.0`.
        R (float, optional): Radius of the large circle in SoS distribution. [m]. Defaults to `0.01`.
        R1 (float, optional): Radius of the small circle in SoS distribution. [m]. Defaults to `0.06`.
        offset (tuple, optional): Offset of circle in SoS distribution. [m]. Defaults to `(0.0, 0.0)`.
        rou (int, optional): Density. [g/cm^3] Defaults to `1000`.

    Returns:
        `kWaveMedium`: The medium object with varying SoS distribution.
    """

    XX, YY = np.meshgrid(kgrid.x_vec.copy(), kgrid.y_vec.copy())
    SoS = np.ones((Ny, Nx)) * v0
    SoS[XX**2 + YY**2 < R**2] = v1
    SoS[(XX - offset[0])**2 + (YY - offset[1])**2 < R1**2] = v2

    medium = kWaveMedium(sound_speed=SoS, sound_speed_ref=v0, density=rou)
    
    return medium


def forward_2D(p0, kgrid, medium, sensor, T_sample, PML_size=8):
    """2D forward simluation with K-wave.

    Args:
        p0 (`numpy.ndarray`): Initial pressure distribution.
        kgrid (`kWaveGrid`): K-wave grid object.
        medium (`kWaveMedium`): K-wave medium object.
        sensor (`kSensor`): K-wave sensor object.
        PML_size (int, optional): Size of PML. Defaults to 8.

    Returns:
        `numpy.ndarray`: Photoacoustic data collected by tranceducers with shape `(N_transducer, N_T)`.
    """
    
    pathname = gettempdir()

    source = kSource()
    source.p0 = p0

    # Smooth the initial pressure distribution and restore the magnitude.
    # source.p0 = smooth(source.p0, False)

    # Create the time array.
    kgrid.makeTime(medium.sound_speed)
    kgrid.setTime(4000, T_sample)

    # Set the input arguements: force the PML to be outside the computational grid switch off p0 smoothing within kspaceFirstOrder2D.
    input_args = {
        'PMLInside': False,
        'PMLSize': PML_size,
        'Smooth': False,
        'SaveToDisk': os.path.join(pathname, 'input.h5'),
        'SaveToDiskExit': False, 
    }

    # Run the simulation.
    sensor_data = kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })
    
    return sensor_data


@numba.jit(nopython=True) 
def delay_and_sum(R_ring, T_sample, V_sound, Sinogram, ImageX, ImageY, d_delay=0):
    """Generate a 2D Delay And Sum reconstructed PACT image of ring transducer array. This function is accelerated by `numba.jit` on a GPU.

    Args:
        R_ring (`float`): The R_ring [m] of the ring transducer array.
        T_sample (`float`): Sample time interval of the signals [s].
        V_sound (`float`): The sound speed [m/s] used for Delay And Sum reconstruction.
        Sinogram (`numpy.ndarray`): A 2D array and each column of it is the signal recievde by one transducer. The nummber of transducers should be the number of columns. The transducers should be evenly distributed on a circle in counterclockwise arrangement and the first column correspond to the transducer in the dirrection `2pi/N` in the first quartile. The first sample should be at time 0 when the photoacoustic effect happens.
        ImageX (`numpy.ndarray`): The vector [m] defining the x coordinates of the grid points on which the reconstruction is done. The values in the vector should be unifromly-spaced in ascending order. The origin of the cartesian coordinate system is the center of the ring array.
        ImageY (`numpy.ndarray`): The vector [m] defining the y coordinates of the grid points on which the reconstruction is done. The values in the vector should be unifromly-spaced in ascending order. The origin of the cartesian coordinate system is the center of the ring array.
        d_delay (`float`): The delay distance [m] of the signals used in DAS. The default value is 0.

    Returns:
        `numpy.ndarray`: A 2D array of size `(len(ImageY), len(ImageX))`. `Image[t, s]` is the reconstructed photoacoustic amplitude at the grid point `(ImageX[s], ImageY[t])`.
    """

    N_transducer = Sinogram.shape[0]
    Image = np.zeros((len(ImageX), len(ImageY)))
    delta_angle = 2*np.pi / N_transducer
    angle_transducer = delta_angle * (np.arange(N_transducer,) + 1)

    x_transducer = R_ring * np.sin(angle_transducer - np.pi)
    y_transducer = R_ring * np.cos(angle_transducer - np.pi)
    
    related_data = np.zeros((N_transducer,))
    
    for s in range(len(ImageX)):
        for t in range(len(ImageY)):
            distance_to_transducer = np.sqrt((x_transducer - ImageX[s])**2 + (y_transducer - ImageY[t])**2) - d_delay
            for k in range(N_transducer):
                id = floor(distance_to_transducer[k]/(V_sound * T_sample))
                if id > Sinogram.shape[1] or id < 0:
                    related_data[k] = 0
                else:
                    related_data[k] = Sinogram[k, id]
            Image[t, s] = related_data.mean()

    return Image


def wavefront_fourier(C0, C1, phi1, C2, phi2):
    return lambda theta: C0 + C1 * torch.cos(theta - phi1) + C2 * torch.cos(2 * (theta - phi2))


def wavefront_real(R, r, phi, v0, v1):
    if r < R:
        return lambda theta: (1-v0/v1) * (torch.sqrt(R**2 - (r*torch.sin(theta-phi))**2) + r * torch.cos(theta-phi))
    else:
        return lambda theta: (1-v0/v1) * 2 * torch.sqrt(torch.maximum(R**2 - (r*torch.sin(theta-phi))**2, torch.zeros_like(theta))) * (torch.cos(phi-theta) >= 0)


def PSF(theta, k, w, delay):
    tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+np.pi)))) / 2
    psf = fftshift(ifftn(tf, dim=[-2,-1]), dim=[-2,-1]).abs()
    psf /= psf.sum(axis=(-2,-1)) # Normalization.
    return psf

# def PSF(theta, k, w, delay):
#     tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+np.pi)))) / 2
#     # psf = fftshift(ifftn(tf, dim=[-2,-1]), dim=[-2,-1]).abs()
#     # psf = tf/psf.sum(axis=(-2,-1)) # Normalization.
#     return tf


def get_delays(R, v0, v1, n_delays, mode='linear'):
    if mode == 'linear':
        return np.linspace(0, (1-v0/v1) * R, n_delays)
    elif mode == 'quadric':
        return (1-v0/v1) * R * np.sqrt(np.linspace(0,1,n_delays))
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    # obs_pad = np.zeros([1,256,256])
    # gt_imgs = split_images(obs_pad, img_size=(128, 128))
    # print(len(gt_imgs))
    # gt_imgs = [np.squeeze(gt_img, axis=0) for gt_img in gt_imgs]
    # print(gt_imgs[0].shape)
    
    img = np.zeros([560, 560])
    img = random_rotate(img)