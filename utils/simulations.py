import os 
import numpy as np
import numba





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
    
    image_pad = np.zeros((image.shape[0] + 2*pad_x, image.shape[1] + 2*pad_y))
    image_pad[pad_x:-pad_x, pad_y:-pad_y] = image
    
    return image_pad
    