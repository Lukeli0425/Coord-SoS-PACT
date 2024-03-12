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
        print(keys)
        data = ()
        for key in keys:
            data += (np.array(dict[key]), )
        return data
    

def save_mat(file_name, data):
    """Save data to `.mat` file.

    Args:
        file_name (`str`): Path to file.
        data (`dict`): The dictionary of data to be saved.
    """
    if os.path.exists(file_name):
        os.remove(file_name)
    hdf5storage.savemat(file_name, data)