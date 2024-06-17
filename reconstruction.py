import argparse
import os

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from kwave.ktransducer import kWaveGrid
from models.DAS import DAS
from models.Joint_Recon import Joint_Recon
from utils.data import *
from utils.dataset import get_jr_dataloader
from utils.reconstruction import *
from utils.simulations import get_water_SoS
from utils.utils_torch import get_total_params

DATA_DIR = 'data/'
RESULT_DIR = 'results/'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def deconv():
    pass


def joint_reconstruction(n_iters, lr, weight,
                         basic_params, task_params):
    # Load data.
    sinogram = load_mat(os.path.join(DATA_DIR, task_params['sinogram']))
    if task_params['EIR']:
        EIR = load_mat(os.path.join(DATA_DIR, task_params['EIR']))
        sinogram = deconvolve_sinogram(sinogram, EIR)
    else:
        EIR = None
    if task_params['ring_error']:
        ring_error, _ = load_mat(os.path.join(DATA_DIR, 'RING_ERROR_NEW.mat'))
        ring_error = np.interp(np.arange(0, 512, 1), np.arange(0, 512, 2), ring_error[:,0]) # Upsample ring error.
        # ring_error = torch.tensor(0.0)
    else:
        ring_error = ring_error = torch.tensor(0.0)
    
    # Preparations.
    Nx, Ny = basic_params['Nx'], basic_params['Ny']
    dx, dy = basic_params['dx'], basic_params['dy']
    l = basic_params['l']
    R = task_params['R']
    v0 = get_water_SoS(task_params['T'])   # Background SoS [m/s].
    
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec, kgrid.y_vec
    
    # Delay-and-Sum.
    das = DAS(R_ring=0.05, N_transducer=512, T_sample=1/40e6, x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    
    n_delays = 32
    delays = np.linspace(-10e-4, 6e-4, n_delays)
    
    img_stack = []
    with torch.no_grad():
        for d_delay in tqdm(delays, desc='DAS'):
            recon = das(sinogram=torch.tensor(sinogram).cuda(), 
                        v0=torch.tensor(v0).cuda(),
                        d_delay=torch.tensor(d_delay).cuda(),
                        ring_error=torch.tensor(ring_error.reshape(-1,1,1)).cuda())
            img_stack.append(recon)
    img_stack = torch.stack(img_stack, dim=0)
    data_loader = get_jr_dataloader(img_stack, l=l)
    
    # Joint Reconstruction.
    joint_recon = Joint_Recon(mode='SIREN', x_vec=x_vec, y_vec=y_vec, R=R, v0=v0, n_points=80, l=3.2e-3, n_delays=n_delays, angle_range=(0, 2*torch.pi), 
                              lam_tv=weight)
    joint_recon.cuda()
    print("Number of parameters: %s" % (get_total_params(joint_recon)))

    optimizer = Adam(params=joint_recon.parameters(), lr=lr)
    
    loss_list = []
    for epoch in range(n_iters):
        joint_recon.train()
        train_loss = 0
        for x, y, img in data_loader:
            x, y, img = x.cuda(), y.cuda(), img.cuda()
            rec_jr, SoS_jr, loss = joint_recon(x, y, img, torch.tensor(delays).cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        loss_list.append(train_loss)
        print("Joint Reconstruction:  [{}/{}]  loss={:0.4g} ".format(epoch+1, n_iters, train_loss/len(data_loader)))
        
    # Save results.
    SoS_jr.squeeze(0).squeeze(0).detach().cpu().numpy()
    save_mat(os.path.join(RESULT_DIR, 'SoS_jr.mat'), SoS_jr.swapaxes(0,1), 'SoS')

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='numerical_phantom', choices=['numerical_phantom', 'leaf_phantom', 'mouse_liver'])
    parser.add_argument('--method', type=str, default='Joint_Recon', choices=['Joint_Recon', 'APACT', 'Deconv', 'Dual_DAS', 'DAS'])
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--eval_intv', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--reg', type=str, default='TV', choices=[None, 'TV'])
    parser.add_argument('--weight', type=float, default=2e-9)
    args = parser.parse_args()
    
    # Load configuration file.
    config = load_config('config.yaml')
    basic_params = config['basic_params']
    task_params = config[args.task]
    
    
    
    # Joint reconstruction.
    if args.method == 'Joint_Recon':
        joint_reconstruction(n_iters=args.n_iters, lr=args.lr, weight=args.weight,
                             basic_params=basic_params, task_params=task_params)
    elif args.method == 'APACT':
        pass
    elif args.method == 'Deconv':
        deconv()
    elif args.method == 'Dual_DAS':
        pass
    elif args.method == 'DAS':
        pass
    else:
        raise ValueError("Invalid method.")
    
    # Deconvolution.
    deconv()
    
    # Evaluation.
    pass