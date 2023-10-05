import argparse
import json
import logging
import os
from tempfile import gettempdir

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
from numpy.random import rand
from torch import nn
from tqdm import tqdm

from kwave.ktransducer import *
from kwave.utils import *
from utils.dataset import mkdir
from utils.simulations import (delay_and_sum, forward_2D, get_medium,
                               reorder_binary_sensor_data, transducer_response,
                               zero_pad)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def generate_data(dataset_path, n_train=150, 
                  image_size=(2040, 2040), PML_size=4, 
                  SoS_background = 1500.0, R_ring=0.05, N_transducer=512,
                  SoS_das=1512.0, n_delays=8, delay_step=1e-4,
                  n_start=0):
    
    logger = logging.getLogger('DataGenerator')
    
    mkdir(dataset_path)
    sec_path = os.path.join(dataset_path, 'sections')
    mkdir(sec_path)
    vis_path = os.path.join(dataset_path, 'visualization')
    mkdir(vis_path)
    for folder in ['train', 'test']:
        mkdir(os.path.join(dataset_path, folder))
        for subfolder in ['gt', 'obs', 'SoS']:
            mkdir(os.path.join(dataset_path, folder, subfolder))
    
    
    logger.info(' K-wave 2D simulation...')
    Nx, Ny = image_size
    dx, dy = 5e-5, 5e-5
    T_sample = 1/10e6
    d_delays = np.linspace(-(n_delays/2-1), n_delays/2, n_delays) * delay_step
    
    pathname = os.path.join(gettempdir(), f'{n_start}')
    mkdir(pathname)
    
    sec_files = os.listdir(sec_path)
    if 'vis' in sec_files:
        sec_files.remove('vis')

    for idx in tqdm(range(0, len(sec_files))):
        # Simulation parameters.
        R = 0.006 + 7.5e-4 * (rand() -0.5) # U(0.009,0.011)
        R1 = 0.003 + 3e-4 * (rand() -0.5) # U(0.005, 0.007)
        offset = (1e-3 * rand(), 1e-3 * rand())
        rou = 1000
    
        # Pad initial pressure distributions.
        IP_img = np.load(os.path.join(sec_path, f'{idx}.npy'))
        IP_pad, (pad_start_x, pad_end_x), (pad_start_y, pad_end_y) = zero_pad(IP_img, Nx, Ny)
        
        # K-wave 2D forward simulation.
        kgrid = kWaveGrid([Nx, Ny], [dx, dy])
        kgrid.dt = T_sample
        
        medium_uniform = get_medium(kgrid=kgrid, 
                                    Nx=Nx, Ny=Ny, 
                                    SoS_background=SoS_background,
                                    R=0.0, R1=0.0, offset=(0.0, 0.0), rou=rou)
        
        cart_sensor_mask = makeCartCircle(radius=R_ring, num_points=N_transducer,
                                          center_pos=[0,0], arc_angle=2*np.pi)
        sensor = kSensor(cart_sensor_mask) # Assign to sensor structure.
        
        # Uniform SoS distribution (ground truth).
        sensor_data = forward_2D(p0=IP_pad, 
                                 kgrid=kgrid, 
                                 medium=medium_uniform,
                                 sensor=sensor,
                                 PML_size=PML_size,
                                 n_start=n_start)
        sensor_data = reorder_binary_sensor_data(sensor_data=sensor_data, 
                                                 sensor=sensor, 
                                                 kgrid=kgrid, 
                                                 PML_size=PML_size)
        sensor_data = transducer_response(sensor_data, T_sample) # Add transducer response.
        
        # Delay and Sum Reconstruction.
        gt_img = delay_and_sum(R_ring,
                               kgrid.dt,
                               medium_uniform.sound_speed_ref,
                               sensor_data,
                               kgrid.x_vec[pad_start_x:pad_end_x],
                               kgrid.y_vec[pad_start_y:pad_end_y],
                               d_delay=0)

        
        # Heterogeneous SoS distribution (observation).
        medium = get_medium(kgrid=kgrid, 
                            Nx=Nx, Ny=Ny, 
                            SoS_background=SoS_background,
                            R=R, R1=R1, offset=offset, rou=rou)
        
        cart_sensor_mask = makeCartCircle(radius=R_ring, num_points=N_transducer,
                                          center_pos=[0,0], arc_angle=2*np.pi)
        sensor = kSensor(cart_sensor_mask) # Assign to sensor structure.
        
        sensor_data = forward_2D(p0=IP_pad, 
                                 kgrid=kgrid, 
                                 medium=medium,
                                 sensor=sensor,
                                 PML_size=PML_size,
                                 n_start=n_start)
        sensor_data = reorder_binary_sensor_data(sensor_data=sensor_data, 
                                                 sensor=sensor, 
                                                 kgrid=kgrid, 
                                                 PML_size=PML_size)
        sensor_data = transducer_response(sensor_data, T_sample) # Add transducer response.
        
        # Delay and Sum Reconstruction.
        recons = []
        for d_delay in d_delays:
            recon = delay_and_sum(R_ring,
                                  kgrid.dt,
                                  SoS_das,
                                  sensor_data,
                                  kgrid.x_vec[pad_start_x:pad_end_x],
                                  kgrid.y_vec[pad_start_y:pad_end_y],
                                  d_delay=d_delay)
            recons.append(recon)
        obs_imgs = np.array(recons) # Stack in to 3D array of shape [8, 256, 256].


        # Crop and save ground truth and observation images.
        # np.save(IP_img, obs_imgs)
        folder = 'train' if idx < n_train else 'test'
        for i in range(7):
            for j in range(7):
                gt_file = os.path.join(dataset_path, folder, 'gt', f"gt_{idx*49+7*i+j}.npy")
                np.save(gt_file, gt_img[32*i:32*i+64, 32*j:32*j+64])
                obs_file = os.path.join(dataset_path, folder, 'obs', f"obs_{idx*49+7*i+j}.npy")
                np.save(obs_file, obs_imgs[:, 32*i:32*i+64, 32*j:32*j+64])
                
        gt_file = os.path.join(dataset_path, folder, 'gt', f"gt_large_{idx}.npy")
        np.save(gt_file, gt_img)
        obs_file = os.path.join(dataset_path, folder, 'obs', f"obs_large_{idx}.npy")
        np.save(obs_file, obs_imgs)
        SoS_file = os.path.join(dataset_path, folder, 'SoS', f"SoS_{idx}.npy")
        np.save(SoS_file, medium.sound_speed)
    
        # Visualization.
        if idx < 5:
            # Overview.
            plt.figure(figsize=(12,12))
            plt.subplot(3,3,1)
            plt.imshow(gt_img)
            plt.title('Ground Truth')
            for k in range(8):
                plt.subplot(3,3,k+2)
                plt.imshow(obs_imgs[k,:,:])
                plt.title(f'Observation({k})')
            plt.savefig(os.path.join(vis_path, f'vis_{idx}.jpg'), bbox_inches='tight')
            plt.close()
            
            # Small patch.
            for i in range(7):
                for j in range(7):
                    plt.figure(figsize=(12,12))
                    plt.subplot(3,3,1)
                    plt.imshow(gt_img[32*i:32*i+64, 32*j:32*j+64])
                    plt.title('Ground Truth')
                    for k in range(8):
                        plt.subplot(3,3,k+2)
                        plt.imshow(obs_imgs[k, 32*i:32*i+64, 32*j:32*j+64])
                        plt.title(f'Observation({k})')
                    plt.savefig(os.path.join(vis_path, f'vis_{idx}_patch_{i}_{j}.jpg'), bbox_inches='tight')
                    plt.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=130)
    parser.add_argument('--n_start', type=int, default=0)
    parser.add_argument('--n_delays', type=int, default=8)
    opt = parser.parse_args()
    
    generate_data(dataset_path='/mnt/WD6TB/tianaoli/dataset/Brain/', 
                  n_train=opt.n_train,
                  n_delays=opt.n_delays,
                  n_start=opt.n_start)
    
