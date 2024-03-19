import os
import h5py
import hdf5storage
import numpy as np


def load_mat(file):
    """Load data in `.mat` files.

    Args:
        file (`str`): Path to file.

    Returns:
        `tuple`: Tuple of data cubes converted to numpy arrays.
    """
    dict = h5py.File(file)
    keys = list(dict.keys())

    if len(keys) == 1:
        return np.array(dict[keys[0]])
    else:
        data = ()
        for key in keys:
            data += (np.array(dict[key]), )
        return data
    

def save_mat(file_name, data, key='data'):
    """Save data to `.mat` file.

    Args:
        file_name (`str`): Path to file.
        data (`numpy.ndarray`): The dictionary of data to be saved.
        key (`str`, optional): The key to be used in the dictionary. Defaults to 'data'.
    """
    if os.path.exists(file_name):
        os.remove(file_name)
    hdf5storage.savemat(file_name, {key: data})
    
