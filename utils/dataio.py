import json
import logging
import os

import h5py
import hdf5storage
import numpy as np
import yaml

from utils.reconstruction import deconvolve_sinogram


def load_config(file:str) -> dict:
    """Load config files in `.yaml` format.

    Args:
        file (str): Path to file.

    Returns:
        dict: Dictionary of data.
    """
    logger = logging.getLogger('DataIO')
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug(' Successfully loaded data from "%s".', file)
    return config
    

def load_mat(file:str) -> np.ndarray:
    """Load data in `.mat` files.

    Args:
        file (str): Path to file.

    Returns:
        tuple: Tuple of data cubes converted to numpy arrays.
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
    

def save_mat(file:str, data:np.ndarray, key:str='data') -> None:
    """Save data to `.mat` file.

    Args:
        file (str): Path to file.
        data (np.ndarray): The dictionary of data to be saved.
        key (str, optional): The key to be used in the dictionary. Defaults to `'data'`.
    """
    logger = logging.getLogger('DataIO')
    if os.path.exists(file):
        os.remove(file)
    hdf5storage.savemat(file, {key: data})
    logger.debug(' Successfully saved data to "%s".', file)


def prepare_data(data_dir, pa_signal_file, EIR_file, ring_error_file) -> None:
    """Prepare PA signals, EIR, and ring error for reconstruction.

    Args:
        data_dir (str): Path to data directory.
        pa_signal_file (str): PA signals file name.
        EIR_file (str): EIR file name.
        ring_error_file (str): Ring error file name.

    Returns:
        tuple: A tuple of PA signals, EIR, and ring error.
    """
    pa_signal = load_mat(os.path.join(data_dir, pa_signal_file))
    EIR = load_mat(os.path.join(data_dir, EIR_file)) if EIR_file else None
    pa_signal = deconvolve_sinogram(pa_signal, EIR) if EIR is not None else pa_signal.astype(np.float32) # Deconvolve EIR.
    if ring_error_file:
        ring_error, _ = load_mat(os.path.join(data_dir, ring_error_file))
        ring_error = np.interp(np.arange(0, 512, 1), np.arange(0, 512, 2), ring_error[:,0]) # Upsample ring error.
    else:
        ring_error = np.zeros(1)
    return pa_signal, EIR, ring_error.reshape(-1,1,1)


def save_log(results_path:str, log:dict) -> None:
    """Save log dictionary to `.json` file.

    Args:
        results_path (str): Path to results directory.
        log (dict): Log dictionary.
    """
    logger = logging.getLogger('DataIO')
    log_file = os.path.join(results_path, 'log.json')
    with open(log_file, 'w') as f:
        json.dump(log, f)
    logger.debug(' Successfully saved log to "%s".', log_file)
    
    
def load_log(log_file:str) -> dict:
    """Load log dictionary from `.json` file.

    Args:
        log_file (str): Path to log file.

    Returns:
        dict: Log dictionary.
    """
    logger = logging.getLogger('DataIO')
    with open(log_file, 'r') as f:
        log = json.load(f)
    logger.debug(' Successfully loaded log file from "%s".', log_file)
    return log
