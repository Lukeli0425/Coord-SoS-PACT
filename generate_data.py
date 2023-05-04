import os
import argparse
import logging
import numpy as np
from numpy.random import rand
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt
from kwave.utils import *
from kwave.ktransducer import *
from utils.simulations import read_images, join_images, zero_pad, split_images
from utils.simulations import get_medium, reorder_binary_sensor_data, forward_2D, delay_and_sum
from utils.dataset import mkdir
from tempfile import gettempdir

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def generate_sections(vessel_data_path, sec_path, n_sec=5, min_std=0.11):
    
    # Create directories.
    mkdir(os.path.join(sec_path, 'vis'))
    
    
    idx_sec = 0
    for idx in tqdm(range(3890)):
        data = scio.loadmat(os.path.join(vessel_data_path, f'{idx+1}.mat'))
        data = np.array(data['skin'])
        for i_sec in range(n_sec):
            sec = data[:,:,i_sec*6+1]
            sec -= sec[0, 0] # Background subtraction.
            
            if sec.std() > min_std:
                np.save(os.path.join(sec_path, f'sec_{idx_sec}'), sec)
                idx_sec += 1
                
                # Visualization.
                if idx_sec < 30:
                    plt.imshow(sec)
                    plt.savefig(os.path.join(sec_path, 'vis', f'sec_{idx_sec}.jpg'))
    


def generate_data(dataset_path, vessel_data_path, n_train=10000, sectioning=False,
                  image_size=(2024, 2024), PML_size=12, 
                  sos_background = 1500.0, R_ring=0.05, N_transducer=512,
                  n_delays=8, delay_step=2e-4,
                  n_start=0):
    
    logger = logging.getLogger('DataGenerator')
    
    mkdir(dataset_path)
    sec_path = os.path.join(dataset_path, 'sections')
    mkdir(sec_path)
    vis_path = os.path.join(dataset_path, 'visualization')
    mkdir(vis_path)
    for folder in ['train', 'test']:
        mkdir(os.path.join(dataset_path, folder))
        for subfolder in ['gt', 'obs', 'sos']:
            mkdir(os.path.join(dataset_path, folder, subfolder))
        
    if sectioning:
        logger.info('Creating blood vessel sections...')
        generate_sections(logger=logger,
                          vessel_data_path=vessel_data_path,
                          sec_path=sec_path, 
                          n_sec=6)
    
    
    logger.info('K-wave 2D simulation...')
    Nx, Ny = image_size
    dx, dy = 5e-5, 5e-5
    T_sample = 1/40e6
    d_delays = np.linspace(-(n_delays/2-1), n_delays/2, n_delays) * delay_step
    
    pathname = os.path.join(gettempdir(), f'{n_start}')
    mkdir(pathname)
    
    sec_files = os.listdir(sec_path)
    sec_files.remove('vis')
    # for idx in tqdm(range(len(sec_files)//4)):
    for idx in tqdm(range(2500, 3000)):
        # Simulation parameters.
        R = 0.01 + 0.004 * (rand() -0.5) # U(0.008,0.012)
        R1 = 0.006 + 0.001 * (rand() -0.5) # U(0.005, 0.007)
        offset = 0.001 * rand()
        rou = 1000
    
        # Join and pad ground truth images.
        gt_imgs = read_images(sec_path, sec_files[4*idx:4*idx+4])
        gt_joint = join_images(gt_imgs)
        gt_pad, (pad_start_x, pad_end_x), (pad_start_y, pad_end_y) = zero_pad(gt_joint, Nx, Ny)
        
        # K-wave 2D forward simulation.
        kgrid = kWaveGrid([Nx, Ny], [dx, dy])
        kgrid.dt = T_sample
        
        medium = get_medium(kgrid=kgrid, 
                            Nx=Nx, Ny=Ny, 
                            sos_background=sos_background,
                            R=R, R1=R1, offset=offset, rou=rou)
        
        cart_sensor_mask = makeCartCircle(radius=R_ring, num_points=N_transducer,
                                          center_pos=[0,0], arc_angle=2*np.pi)
        sensor = kSensor(cart_sensor_mask) # Assign to sensor structure.
        
        sensor_data = forward_2D(p0=gt_pad, 
                                 kgrid=kgrid, 
                                 medium=medium,
                                 sensor=sensor,
                                 PML_size=PML_size,
                                 n_start=n_start)
        sensor_data = reorder_binary_sensor_data(sensor_data=sensor_data, 
                                                 sensor=sensor, 
                                                 kgrid=kgrid, 
                                                 PML_size=PML_size)
        
        # Delay and Sum Reconstruction.
        obs_pad = []
        for d_delay in d_delays:
            recon = delay_and_sum(R_ring,
                                  kgrid.dt,
                                  medium.sound_speed_ref,
                                  sensor_data,
                                  kgrid.x_vec[pad_start_x:pad_end_x],
                                  kgrid.y_vec[pad_start_y:pad_end_y],
                                  d_delay=d_delay)
            obs_pad.append(recon)
        
        # Split observation images.
        obs_imgs = split_images(obs_pad, img_size=(128, 128))
        print(obs_imgs[0].shape)

        # Save ground truth and observation images.
        for i, (gt_img, obs_img) in enumerate(zip(gt_imgs, obs_imgs)):
            folder = 'train' if 4*idx+i < n_train else 'test'
            gt_file = os.path.join(dataset_path, folder, 'gt', f"gt_{4*idx+i}.npy")
            np.save(gt_file, gt_img)
            obs_file = os.path.join(dataset_path, folder, 'obs', f"obs_{4*idx+i}.npy")
            np.save(obs_file, obs_img)
        
        # Visualization.
        if idx < 5:
            for j, (gt_img, obs_img) in enumerate(zip(gt_imgs, obs_imgs)):
                plt.figure(figsize=(12,12))
                plt.subplot(3,3,1)
                plt.imshow(gt_img)
                plt.title('Ground Truth')
                for k in range(8):
                    plt.subplot(3,3,k+2)
                    plt.imshow(obs_img[k,:,:])
                    plt.title(f'Observation({k})')
                plt.savefig(os.path.join(vis_path, f'vis_{4*idx+j}.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_start', type=int, default=0)
    parser.add_argument('--sectioning', action="store_true")
    parser.add_argument('--n_delays', type=int, default=8)
    opt = parser.parse_args()
    
    generate_data(dataset_path='/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/', 
                  vessel_data_path='/mnt/WD6TB/tianaoli/SkinVessel/',
                  n_train=opt.n_train, sectioning=opt.sectioning,
                  n_delays=opt.n_delays,
                  n_start=opt.n_start)
    
