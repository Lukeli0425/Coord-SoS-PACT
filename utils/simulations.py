import os 
import numpy as np
import numba



def read_images(path, files):
    return [np.load(os.path.join(path, file)) for file in files]

def join_images(images, n_row=2, n_col=2):
    
    if len(images) != n_row * n_col:
        raise ValueError('Number of images does not match the number of rows and columns.')

    images_row = []
    for i in range(n_row):
        images_row.append(np.concatenate(images[n_row*i:n_row*i+n_col], axis=1))
    image_joint = np.concatenate(images_row , axis=0)
    
    return image_joint

def split_images(image, img_size=(128, 128)):
    images = []
    n_x, n_y = image.shape[0]//img_size[0], image.shape[1]//img_size[1]
    for idx in range(n_x):
        for idy in range(n_y):
            images.append(image[idx*img_size[0]:(idx+1)*img_size[0], idy*img_size[1]:(idy+1)*img_size[1]])
    return images
    

def reorder_binary_sensor_data(sensor_data, sensor, kgrid, PML_size):

    x_sensor = kgrid.x[sensor.mask[PML_size:-PML_size,PML_size:-PML_size] == 1]
    y_sensor = kgrid.y[sensor.mask[PML_size:-PML_size,PML_size:-PML_size] == 1]
    
    angle = np.arctan2(-x_sensor, -y_sensor)
    angle[angle < 0] = 2 * np.pi + angle[angle < 0]
    reorder_index = np.argsort(angle)
    
    return sensor_data[reorder_index]


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
        `numpy.ndarray`: A 2D array of size `(len(ImageY), len(ImageX))`. Image(t, s) is the reconstructed photoacoustic amplitude at the grid point (ImageX(s), ImageY(t)).
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


def subtract_background(image):
    """Conduct background subtraction to blood vessel section images.

    Args:
        image (`numpy.ndarray`): Input blood vessel section image.

    Returns:
        `numpy.ndarray`: Background subtracted blood vessel section image.
    """
    return image - image[0,0]


def zero_pad(image, pad_x, pad_y):
    """Pads image with zeros.

    Args:
        image (`numpy.ndarray`): Input image.
        pad_x (`int`): Pad length along x direction.
        pad_y (`int`): Pad length along y direction.

    Returns:
        `numpy.ndarray`: Padded image.
    """
    
    image_pad = np.zeros((pad_x, pad_y))
    
    pad_start, pad_end = int(pad_x/2-image.shape[0]/2), int(pad_x/2+image.shape[0]/2)
    image_pad[pad_start:pad_end, pad_start:pad_end] = image
    
    return image_pad
    