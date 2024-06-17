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
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Successfully loaded data from "{file}".')
    return config
    

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
        data = np.array(dict[keys[0]])
    else:
        data = ()
        for key in keys:
            data += (np.array(dict[key]), )
    print(f'Successfully loaded data from "{file}".')
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
    print(f'Successfully saved data to "{file_name}".')
    