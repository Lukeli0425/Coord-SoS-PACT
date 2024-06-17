import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import choice, rand
from tqdm import tqdm

from kwave.ktransducer import *
from kwave.utils import *
from utils.simulations import (PSF, center, deconvolve_sinogram, delay_and_sum,
                               forward_2D, get_delays, get_medium,
                               get_water_SoS, random_rotate,
                               reorder_binary_sensor_data, transducer_response,
                               wavefront_real, zero_pad)
from utils.utils_torch import get_fourier_coord

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class PACT_Data_Generator():
    def __init__(self, dataset_path, load_info=True):
        self.logger = logging.getLogger('DataGenerator')
        
        # Define and create directories.
        self.dataset_path = dataset_path
        self.sino_path = os.path.join(dataset_path, 'sinogram')
        self.gold_path = os.path.join(dataset_path, 'gold')
        self.obs_path = os.path.join(dataset_path, 'obs')
        self.psf_path = os.path.join(dataset_path, 'psf')
        self.SoS_path = os.path.join(dataset_path, 'SoS')
        self.fullimg_path = os.path.join(dataset_path, 'full_image')
        self.IP_path = os.path.join(self.fullimg_path, 'IP')
        self.obs_full_path = os.path.join(self.fullimg_path, 'obs')
        self.vis_path = os.path.join(dataset_path, 'visualization')
        
        for dir in [self.sino_path, self.gold_path, self.obs_path, self.psf_path, self.SoS_path, 
                    self.IP_path, self.obs_full_path, self.vis_path]:
            Path(dir).mkdir(parents=True, exist_ok=True)
            
        self.info_file = os.path.join(self.dataset_path, 'info.json')


    def load_info(self):
        """Load information from json file.

        Raises:
            Exception: Failed loading information.
        """
        try:
            with open(self.info_file, 'r') as f:
                self.info = json.load(f)
            self.logger.info(' Successfully loaded dataset information from %s.', self.info_file)
        except:
            raise Exception(' Failed loading dataset information from %s.', self.info_file)
            # self.logger.critical(' Failed loading information from %s.', self.info_file)


    def gerenate_params(self, n_total=274, R_ring=0.05, N_transducer=512, n_delays=8, image_size=(2544, 2544), PML_size=8):
        """Generate simulation parameters and save to json file.

        Args:
            n_total (int, optional): _description_. Defaults to 274.
            R_ring (float, optional): _description_. Defaults to 0.05.
            N_transducer (int, optional): _description_. Defaults to 512.
            n_delays (int, optional): _description_. Defaults to 8.
            image_size (tuple, optional): _description_. Defaults to (2544, 2544).
            PML_size (int, optional): _description_. Defaults to 8.
        """
        self.logger.info(' Generating simulation parameters...')
        
        dx, dy = 4e-5, 4e-5
        l = 3.2e-3 # Patch size [m].
        T_sample = 1/40e6 # Sampling period [s].
        
        sequence = np.arange(0, n_total) # Generate random sequence for dataset.
        sequence = np.delete(sequence, [167, 166, 20, 168, 21, 239])
        np.random.shuffle(sequence)
        params = []
        for k, idx in enumerate(sequence):
            idx = int(idx)
            T = 24. + 10. * rand() # Water temperature [C].
            v0 = get_water_SoS(T) # Background SoS [m/s].
            v1 = 1560.0 + 100. * rand() # SoS in tissue [m/s].
            v2 = v1 + (50. + 50. * rand()) * choice([1,-1])  
            R = 9.6e-3 + 1e-3 * rand() * choice([1,-1])
            R1 = 2.0e-3 + 1.0e-3 * rand()
            offset = ((0.9+0.4*rand())*1e-3 * choice([1,-1]), (0.9+0.4*rand())*1e-3 * choice([1,-1]))
            rou = 990. + 40. * rand() # Density [kg/m^3].

            params.append({'idx':idx, 'water temperature':T, 'v0':v0, 'v1':v1, 'v2':v2, 'R':R, 'R1':R1, 'offset':offset, 'density':rou})

        self.info = {'n_total':n_total, 'params':params, 'R_ring':R_ring, 'N_transducer':N_transducer, 'n_delays':n_delays,
                     'T_sample':T_sample, 'dx':dx, 'dy':dy, 'image_size':image_size, 'patch_size':l, 'PML_size':PML_size}
        
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)
        self.logger.info(' Dataset information saved to %s.', self.info_file)


    def generate_data(self, n_start=0):
        """Generate data using K-wave simulation.

        Args:
            n_start (int, optional): _description_. Defaults to 0.
        """

        # Load mice data.
        data_path = '/mnt/WD6TB/tianaoli/Mice/'
        data = h5py.File(os.path.join(data_path, 'mice_full_recon.mat'))
        mice_full_recon = np.array(data['full_recon_all'])
        
        # Load K-wave EIR.
        EIR_data = h5py.File('tutorials/data/EIR_KWAVE.mat')
        EIR = np.array(EIR_data['ht'])
        EIR = np.append(EIR, 0)
        EIR = -2 * (EIR[1:] - EIR[:-1])
        EIR = np.expand_dims(EIR, axis=0)
        
        # Load information.
        try:
            self.load_info()
        except:
            self.gerenate_params()
        
        n_total = self.info['n_total']
        N_transducer = self.info['N_transducer']
        R_ring = self.info['R_ring']
        n_delays = self.info['n_delays']
        PML_size = self.info['PML_size']
        Nx, Ny = self.info['image_size']
        dx, dy = self.info['dx'], self.info['dy']
        l = self.info['patch_size']
        T_sample = self.info['T_sample']
        params = self.info['params']
        
        # for k in tqdm(range(0, 269)):
        for k in range(n_start, n_start+1):
            self.logger.info(' Generating data... [%s/%s] ', k+1, n_total)
            # Simulation parameters.
            idx = params[k]['idx']
            T = params[k]['water temperature'] # Water temperature [C].
            v0, v1, v2 = params[k]['v0'], params[k]['v1'], params[k]['v2']
            R, R1, offset = params[k]['R'], params[k]['R1'], params[k]['offset']
            rou = params[k]['density'] # Density [kg/m^3].
            delays = get_delays(R, v0, v1, n_delays, 'linear')

            # Resizing, centering and cropping.
            img = mice_full_recon[idx, :, :]
            img = cv2.resize(img, (640, 640)) # Resize to (384, 384).
            x_c, y_c = center(img)
            IP_img = img[x_c-280:x_c+280, y_c-280:y_c+280]
            
            # Randomly rotate.
            IP_img = random_rotate(IP_img)
            np.save(os.path.join(self.IP_path, f"IP_{k}.npy"), IP_img) # Save initial pressure.
            
            # Pad initial pressure distributions.
            IP_pad, (pad_start_x, pad_end_x), (pad_start_y, pad_end_y) = zero_pad(IP_img, Nx, Ny)
            
            # K-wave 2D forward simulation.
            kgrid = kWaveGrid([Nx, Ny], [dx, dy])
            kgrid.dt = T_sample / 2
            
            medium = get_medium(kgrid=kgrid, Nx=Nx, Ny=Ny, 
                                v0=v0, v1=v1, v2=v2,
                                R=R, R1=R1, offset=offset, rou=rou)
            
            cart_sensor_mask = makeCartCircle(radius=R_ring, num_points=N_transducer,
                                            center_pos=[0,0], arc_angle=2*np.pi)
            sensor = kSensor(cart_sensor_mask) # Assign to sensor structure.
            
            sensor_data = forward_2D(p0=IP_pad, 
                                    kgrid=kgrid, 
                                    medium=medium,
                                    sensor=sensor,
                                    T_sample=T_sample/2,
                                    PML_size=PML_size)
            sensor_data = sensor_data[:, ::2]
            sensor_data = reorder_binary_sensor_data(sensor_data=sensor_data, 
                                                    sensor=sensor, 
                                                    kgrid=kgrid, 
                                                    PML_size=PML_size)
            sinogram = transducer_response(sensor_data) # Add transducer response.
            
            np.save(os.path.join(self.sino_path, f"sinogram_{k}.npy"), sinogram) # Save sinogram.
            np.save(os.path.join(self.SoS_path, f"SoS_{k}.npy"), medium.sound_speed) # Save SoS distribution.
            
            sinogram_deconv = deconvolve_sinogram(sinogram, EIR, np.argmax(EIR))
        
            # Delay and Sum Reconstruction.
            recons = []
            for d_delay in delays:
                recon = delay_and_sum(R_ring,
                                    T_sample,
                                    v0,
                                    sinogram_deconv,
                                    kgrid.x_vec[pad_start_x:pad_end_x],
                                    kgrid.y_vec[pad_start_y:pad_end_y],
                                    d_delay=d_delay)
                recons.append(recon)
            obs_imgs = np.array(recons) # Stack in to 3D array of shape [8, 256, 256].
            
            np.save(os.path.join(self.obs_full_path, f"fullimg_{k}.npy"), obs_imgs) # Save full observation image.
            
            
            for i in range(13):
                for j in range(13):
                    # Crop and save ground truth and observation images.
                    gt_file = os.path.join(self.gold_path, f"gt_{k*169+13*i+j}.npy")
                    np.save(gt_file, IP_img[40*i:40*i+80, 40*j:40*j+80])
                    obs_file = os.path.join(self.obs_path, f"obs_{k*169+13*i+j}.npy")
                    np.save(obs_file, obs_imgs[:, 40*i:40*i+80, 40*j:40*j+80])

                    # Calculate PSF based on location of patch.
                    R += 0.1e-3 * rand() * choice([-1, 1])
                    x, y = (j-6)*l / 2, (6-i)*l / 2
                    r, phi = np.sqrt(x**2 + y**2), np.arctan2(x, y)
                    w_real = wavefront_real(R, r, phi, v0, v1)
                    k2D, theta2D = get_fourier_coord(n_points=80, l=l, device='cpu')
                    psfs = []
                    for _, delay in enumerate(delays):
                        psfs.append(PSF(theta2D, k2D, w_real, delay))
                    psf = torch.stack(psfs, dim=0)
                    torch.save(psf, os.path.join(self.psf_path, f'psf_{k*169+13*i+j}.pth'))
                    
                    # Visualization.
                    if k < 4 and i == 5 and j == 3:
                        plt.figure(figsize=(16, 5))
                        for id, delay in enumerate(delays):
                            plt.subplot(2,4,id+1)
                            plt.imshow(psfs[id])
                            plt.xticks([])
                            plt.yticks([])
                            plt.title('Delay={:.2f}mm'.format(delays[id]*1e3), fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.vis_path, f'psf_vis_{k}_({i}_{j}).jpg'))
                        plt.close()
                    
            
            # Visualization.
            
            # Overview.
            plt.figure(figsize=(12,12))
            plt.subplot(3,3,1)
            plt.imshow(IP_img)
            plt.title('Ground Truth')
            for m in range(8):
                plt.subplot(3,3,m+2)
                plt.imshow(obs_imgs[m,:,:])
                plt.title(f'Observation({m})')
            plt.savefig(os.path.join(self.vis_path, f'vis_{k}.jpg'), bbox_inches='tight')
            plt.close()
                
            if k < 4:
                # Small patch.
                for i in range(13):
                    for j in range(13):
                        plt.figure(figsize=(12,12))
                        plt.subplot(3,3,1)
                        plt.imshow(IP_img[40*i:40*i+80, 40*j:40*j+80])
                        plt.title('Ground Truth')
                        for m in range(8):
                            plt.subplot(3,3,m+2)
                            plt.imshow(obs_imgs[m, 40*i:40*i+80, 40*j:40*j+80])
                            plt.title(f'Observation({m})')
                        plt.savefig(os.path.join(self.vis_path, f'vis_{k}_patch_{i}_{j}.jpg'), bbox_inches='tight')
                        plt.close()
                    
            


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_start', type=int, default=0)
    parser.add_argument('--n_delays', type=int, default=8)
    opt = parser.parse_args()
    
    data_generator = PACT_Data_Generator(dataset_path='/mnt/WD6TB/tianaoli/dataset/Mice_new1/')
    data_generator.generate_data(n_start=opt.n_start)