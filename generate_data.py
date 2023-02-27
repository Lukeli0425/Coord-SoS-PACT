import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as scio


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
    
    generate_data(data_path='/Users/luke/Downloads/skinVessel/', n_train=opt.n_train)
