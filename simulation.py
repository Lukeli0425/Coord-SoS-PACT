import argparse
import logging
import os
import sys
from tempfile import gettempdir
from time import time

sys.path.append('../')

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from kwave.utils import *
from utils.dataio import *
from utils.simulations import *

DATA_PATH = 'data/'


def k_wave_simulation(sample_id:int, params:dict):
    logger = logging.getLogger('K-Wave')
    logger.info(' Running K-Wave simulation on %s ...', params['description'])
    
    # Simulation parameters.
    Nx, Ny = params['Nx_pad'], params['Ny_pad']
    dx, dy = params['dx'], params['dy']# Grid point spacing in the y direction [m].
    v0 = get_water_sos(params['T'])             # Background SOS [m/s].
    rou = params['rou']                                # Density [kg/m^3].
    PML_size = params['PML_size']                                # Size of the PML in grid points.
    R_ring = params['R_ring']                               # Radius of the ring array [m].      
    T_sample = params['T_sample']                           # Sample time step [s].
    N_transducer = params['N_transducer']                          # Number of transducers in ring array.
    center_pos = params['center_pos']
    arc_angle = params['arc_angle'] * np.pi
    upsample_factor = params['upsample_factor']
    pathname = gettempdir()
    
    # Load and pad IP.
    IP_img = load_mat(os.path.join(DATA_PATH, 'IP.mat'))
    IP_pad = np.zeros((Nx, Ny))
    pad_start, pad_end = (Nx-IP_img.shape[0]) // 2, (Ny+IP_img.shape[1]) // 2
    IP_pad[pad_start:pad_end, pad_start:pad_end] = IP_img
    
    # Load and pad SOS.
    SOS = load_mat(os.path.join(DATA_PATH, params['SOS']))
    SOS_pad = np.ones((Nx, Ny)) * v0
    SOS_pad[pad_start:pad_end, pad_start:pad_end] = SOS

    t_start = time()
    
    # Preparations.
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    medium = kWaveMedium(sound_speed=SOS_pad, sound_speed_ref=v0, density=rou)
    source = kSource()
    source.p0 = IP_pad

    # Smooth the initial pressure distribution and restore the magnitude.
    source.p0 = smooth(source.p0, True)

    cart_sensor_mask = makeCartCircle(R_ring, N_transducer, center_pos, arc_angle)
    sensor = kSensor(cart_sensor_mask) # Assign to sensor structure.

    # Create the time array.
    kgrid.makeTime(medium.sound_speed)
    kgrid.setTime(2000*upsample_factor, T_sample/upsample_factor) 

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
    logger.info(f" Sinogram shape: %s", sensor_data.shape)
    
    sensor_data = sensor_data[:, ::upsample_factor]
    sensor_data = reorder_binary_sensor_data(sensor_data, sensor, kgrid, PML_size)
    sinogram = transducer_response(sensor_data)
    
    t_end = time()
    logger.info(' Simulation completed in %.2f s.', t_end-t_start)
    
    # Save results and log.
    save_mat(os.path.join(DATA_PATH, params['sinogram']), sinogram.swapaxes(0,1), 'sinogram')
    log = {'time':t_end-t_start, **params}
    save_log(DATA_PATH, log)
    logger.info(' Results saved to "%s".', DATA_PATH)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1', '2'])
    parser.add_argument('--sample_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parser.parse_args()
    
    # Select GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load parameters.
    config = load_config('config.yaml')
    bps, sps, tps = config['basic_params'], config['simulation'], config[f'numerical {args.sample_id}']

    k_wave_simulation(sample_id=args.sample_id, params=bps|sps|tps)