import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt


def generate_sections(vessel_data_path, dataset_path, n_sec=5, min_std=0.11):
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    sec_path = os.path.join(dataset_path, 'sections')
    if not os.path.exists(sec_path):
        os.mkdir(sec_path)

    if not os.path.exists(os.path.join(sec_path, 'vis')):
        os.mkdir(os.path.join(sec_path, 'vis'))
    
    
    idx_sec = 0
    # for idx in tqdm(range(3890)):
    for idx in range(3890):
        data = scio.loadmat(os.path.join(vessel_data_path, f'{idx+1}.mat'))
        data = np.array(data['skin'])
        for i_sec in range(n_sec):
            sec = data[:,:,i_sec*6+1]
            sec -= sec[0, 0] # Background subtraction.
            
            if sec.std() > min_std:
                np.save(os.path.join(sec_path, f'sec_{idx_sec}'), sec)
                idx_sec += 1
                
                if idx_sec < 30:
                    plt.imshow(sec)
                    plt.savefig(os.path.join(sec_path, 'vis', f'sec_{idx_sec}.jpg'))
                    print(f'{idx_sec}: {sec.std()}')

    print(idx_sec)
    


def generate_data(data_path, n_train=10000, load_info=True):
    vessel_files = os.listdir(data_path)
    
    for idx in tqdm(range(3890)):
        data = scio.loadmat(os.path.join(data_path, f'{idx+1}.mat'))
        # print(data)
        # skin = np.array(['skin'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=10000)
    opt = parser.parse_args()
    
    # generate_data(data_path='/Users/luke/Downloads/skinVessel/', n_train=opt.n_train)
    
    generate_sections(vessel_data_path='/mnt/WD6TB/tianaoli/skinVessel/', 
                      dataset_path='/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/', 
                      n_sec=6)
