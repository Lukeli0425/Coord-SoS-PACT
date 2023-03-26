import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt
from utils.simulations import reorder_binary_sensor_data, read_images, join_images, split_images, forward_2D, delay_and_sum

def generate_sections(logger, vessel_data_path, sec_path, n_sec=5, min_std=0.11):
    
    # Create directories.
    if not os.path.exists(os.path.join(sec_path, 'vis')):
        os.mkdir(os.path.join(sec_path, 'vis'))
    
    
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
    


def generate_data(dataset_path, vessel_data_path, n_train=10000, load_info=True, sectioning=True,
                  image_size=(2024, 2024), PML_size=12, 
                  sos_background = 1500.0, R_ring=0.05,
                  n_delays=8, delay_step=2e-7):
    logger = logging.getLogger('DataGenerator')
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    sec_path = os.path.join(dataset_path, 'sections')
    if not os.path.exists(sec_path):
        os.mkdir(sec_path)
        
    if sectioning:
        logger.info('Creating sections...')
        generate_sections(logger=logger,
                          vessel_data_path=vessel_data_path,
                          sec_path=sec_path, 
                          n_sec=6)
    
    # Simulation parameters.
    d_delays  = np.linspace(-(n_delays/2-1), n_delays/2, n_delays) * delay_step
   
    sec_files = os.listdir(sec_path)
    for idx in tqdm(range(len(sec_files)//4)):
        # Join and pad ground truth images.
        gt_imgs = read_images(sec_path, sec_files[idx:idx+4])
        gt_joint = join_images(gt_imgs)
        
        # K-wave 2D forward simulation.
        sensor_data = forward_2D(gt_joint)
        
        # Delay and Sum Reconstruction.
        obs_joint = []
        for d_delay in d_delays:
            recon = delay_and_sum(R_ring, 
                                    kgrid.dt, 
                                    medium.sound_speed_ref, 
                                    sensor_data,
                                    kgrid.x_vec[pad_start:pad_end], 
                                    kgrid.y_vec[pad_start:pad_end],
                                    d_delay=d_delay)
            obs_joint.append(recon)
        
        # Split observation images.
        images_obs = split_images(obs_joint, img_size=(128, 128))

        # Save ground truth and observation images.
        
        
        break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--sectioning', action="store_true")
    parser.add_argument('--n_delays', type=int, default=8)
    opt = parser.parse_args()
    
    generate_data(dataset_path='/mnt/WD6TB/tianaoli/dataset/skinVessel_PACT/', 
                  vessel_data_path='/mnt/WD6TB/tianaoli/skinVessel/',
                  n_train=opt.n_train, sectioning=opt.sectioning,
                  n_delays=opt.n_delays)
    
    
