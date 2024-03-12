import numpy as np


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
