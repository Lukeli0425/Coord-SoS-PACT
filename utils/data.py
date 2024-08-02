import logging
import os

import h5py
import hdf5storage
import numpy as np
import yaml


def load_config(file):
    """Load config files in `.yaml` format.

    Args:
        file (`str`): Path to file.

    Returns:
        `dict`: Dictionary of data.
    """
    logger = logging.getLogger('DataIO')
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug(' Successfully loaded data from "%s".', file)
    return config
    

def load_mat(file):
    """Load data in `.mat` files.

    Args:
        file (`str`): Path to file.

    Returns:
        `tuple`: Tuple of data cubes converted to numpy arrays.
    """
    logger = logging.getLogger('DataIO')
    
    dict = h5py.File(file)
    keys = list(dict.keys())

    if len(keys) == 1:
        data = np.array(dict[keys[0]])
    else:
        data = ()
        for key in keys:
            data += (np.array(dict[key]), )
    logger.debug(' Successfully loaded data from "%s".', file)
    return data
    

def save_mat(file, data, key='data'):
    """Save data to `.mat` file.

    Args:
        file (`str`): Path to file.
        data (`numpy.ndarray`): The dictionary of data to be saved.
        key (`str`, optional): The key to be used in the dictionary. Defaults to 'data'.
    """
    logger = logging.getLogger('DataIO')
    if os.path.exists(file):
        os.remove(file)
    hdf5storage.savemat(file, {key: data})
    logger.debug(' Successfully saved data to "%s".', file)
    