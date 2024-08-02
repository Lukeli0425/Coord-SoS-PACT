import argparse
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import LogNorm, Normalize
from torch.optim import Adam
from tqdm import tqdm

from kwave.ktransducer import kWaveGrid
from models.apact import APACT
from models.das import DAS, Dual_SOS_DAS
from models.deconv import MultiChannel_Deconv
from models.nf_apact import NF_APACT
from models.pact import TF_PACT, Wavefront_SOS
from utils.data import *
from utils.dataset import get_jr_dataloader
from utils.reconstruction import *
from utils.simulations import get_water_SOS
from utils.utils_torch import get_total_params
from utils.visualization import *

plt.set_loglevel("warning")

DATA_DIR = 'data/'
RESULT_DIR = 'results_new/'


def load_data(sinogram_file, EIR_file, ring_error_file):
    sinogram = load_mat(os.path.join(DATA_DIR, sinogram_file))
    if EIR_file:
        EIR = load_mat(os.path.join(DATA_DIR, EIR_file))
    else:
        EIR = None
    if ring_error_file:
        ring_error, _ = load_mat(os.path.join(DATA_DIR, ring_error_file))
        ring_error = np.interp(np.arange(0, 512, 1), np.arange(0, 512, 2), ring_error[:,0]) # Upsample ring error.
    else:
        ring_error = np.zeros(1)
    return torch.tensor(sinogram).cuda(), EIR, torch.tensor(ring_error.reshape(-1,1,1)).cuda()


def das(v_das, basic_params, task_params):
    logger = logging.getLogger('DAS')
    logger.info(" Reconstructing %s with Delay-and-Sum (v_das=%.1fm/s).", task_params['description'], v_das)
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'DAS_{:.1f}/'.format(v_das))
    os.makedirs(results_path, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = load_data(sinogram_file=task_params['sinogram'], EIR_file=task_params['EIR'], ring_error_file=task_params['ring_error'])
    
    # Deconvolve EIR.
    if EIR is not None:
        sinogram = deconvolve_sinogram(sinogram, EIR)
    
    # Preparations.
    Nx, Ny = basic_params['Nx'], basic_params['Ny']
    dx, dy = basic_params['dx'], basic_params['dy']
    x_c, y_c = task_params['x_c'], task_params['y_c']
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    
    das = DAS(R_ring=basic_params['R_ring'], N_transducer=basic_params['N_transducer'], T_sample=basic_params['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()

    sinogram = sinogram[:,task_params['t0']:]
    ring_error = torch.tensor(ring_error).cuda()
    with torch.no_grad():
        t_start = time()
        IP_rec = das(sinogram=sinogram, v0=v_das, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()

    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_path, 'rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title("DAS Reconstruction", fontsize=16)
    plt.text(7, 23, "$v_{das}$"+" = {:.1f} m/s".format(v_das), color='white', fontsize=15)
    plt.text(426, 23, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'vis.png'))
    
    logger.info(' Results saved in "%s".', results_path)

def dual_sos_das(v_sb, basic_params, task_params):
    logger = logging.getLogger('Dual-SOS DAS')
    logger.info(" Reconstructing %s with Dual-SOS DAS (v_sb=%.1fm/s).", task_params['description'], v_sb)
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'Dual_SOS_DAS_{:.1f}/'.format(v_sb))
    os.makedirs(results_path, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = load_data(sinogram_file=task_params['sinogram'], EIR_file=task_params['EIR'], ring_error_file=task_params['ring_error'])
    
    # Deconvolve EIR.
    if EIR is not None:
        sinogram = deconvolve_sinogram(sinogram, EIR)
    
    # Preparations.
    Nx, Ny = basic_params['Nx'], basic_params['Ny']
    dx, dy = basic_params['dx'], basic_params['dy']
    x_c, y_c = task_params['x_c'], task_params['y_c']
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    v0 = get_water_SOS(task_params['T'])
    
    das_dual = Dual_SOS_DAS(R_ring=basic_params['R_ring'], N_transducer=basic_params['N_transducer'], T_sample=basic_params['T_sample'], 
                            x_vec=x_vec, y_vec=y_vec, R_body=task_params['R_body'], center=(x_c*dx, y_c*dy), mode='zero')
    das_dual.cuda()
    das_dual.eval()

    sinogram = torch.tensor(sinogram[:,task_params['t0']:]).cuda()
    ring_error = torch.tensor(ring_error).cuda()
    with torch.no_grad():
        t_start = time()
        IP_rec = das_dual(sinogram=sinogram, v0=v0, v1=v_sb, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()
        
    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_path, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title("Dual SOS DAS Reconstruction", fontsize=16)
    plt.text(7, 23, "$v_{sb}$"+" = {:.1f} m/s".format(v_sb), color='white', fontsize=15)
    plt.text(426, 23, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'vis.png'))
    
    logger.info(' Results saved in "%s".', results_path)


def deconv(basic_params, task_params):
    logger = logging.getLogger('Deconv')
    logger.info(' Results saved to "%s".', results_path)

def apact(basic_params, task_params):
    logger = logging.getLogger('APACT')
    logger.info(" Reconstructing %s with APACT.", task_params['task'])

def nf_apact(n_delays, n_iters, lr, weight, basic_params, task_params):
    logger = logging.getLogger('NF-APACT')
    logger.info(" Reconstructing %s with NF-APACT.", task_params['description'])
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'NF-APACT_{}delays_{}iters/'.format(n_delays, n_iters))
    os.makedirs(results_path, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = load_data(sinogram_file=task_params['sinogram'], EIR_file=task_params['EIR'], ring_error_file=task_params['ring_error'])
    
    # Deconvolve EIR.
    if EIR is not None:
        sinogram = deconvolve_sinogram(sinogram, EIR)
    
    # Preparations.
    Nx, Ny = basic_params['Nx'], basic_params['Ny']
    dx, dy = basic_params['dx'], basic_params['dy']
    x_c, y_c = task_params['x_c'], task_params['y_c']
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    v0 = get_water_SOS(task_params['T'])
    l_patch, N_patch = basic_params['l_patch'], basic_params['N_patch']
    R_body = task_params['R_body']
    mean, std = task_params['mean'], task_params['std']
    
    das = DAS(R_ring=basic_params['R_ring'], N_transducer=basic_params['N_transducer'], T_sample=basic_params['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    
    delays = np.linspace(-8e-4, 8e-4, n_delays)
    IP_rec = torch.zeros((Nx, Ny)).cuda()
    delays = torch.tensor(delays).cuda().view(-1,1,1)
    sigma = basic_params['fwhm'] / 4e-5 / np.sqrt(2*np.log(2))
    gaussian_window = torch.tensor(gaussian_kernel(sigma, 80)).cuda()
    
    wavefront_sos = Wavefront_SOS(R_body, v0, x_vec, y_vec, n_points=180)
    wavefront_sos.cuda()
    wavefront_sos.eval()
    
    tf_pact = TF_PACT(n_points=2*N_patch, l=2*l_patch, n_delays=delays.shape[0])
    tf_pact.cuda()
    tf_pact.eval()

    mc_deconv = MultiChannel_Deconv(n_pixels=N_patch, l_patch=l_patch)
    mc_deconv.cuda()
    mc_deconv.eval()
    
    nf_apact = NF_APACT(mode='SIREN', mean=mean, std=std, x_vec=kgrid.x_vec, y_vec=kgrid.y_vec, R=R_body, v0=v0, n_points=N_patch, l=l_patch, n_delays=n_delays, angle_range=(0, 2*torch.pi), lam_tv=weight)
    nf_apact.cuda()
    nf_apact.train()
    logger.info(" Number of learnable parameters: %s", get_total_params(nf_apact))

    optimizer = Adam(params=nf_apact.parameters(), lr=lr)
    DAS_stack, loss_list, SOS_list = [], [], []
    
    t_start = time()
    logger.info(" Running DAS (v_das=%.1fm/s) with %s delays.", v0, n_delays)
    with torch.no_grad():
        for d_delay in tqdm(delays, desc='DAS'):
            recon = das(sinogram=sinogram, v0=v0, d_delay=d_delay, ring_error=ring_error)
            DAS_stack.append(recon)
    DAS_stack = torch.stack(DAS_stack, dim=0)
    DAS_stack = (DAS_stack - DAS_stack.mean()) / DAS_stack.std()
    data_loader = get_jr_dataloader(DAS_stack, l_patch, N_patch)
    
    # Joint Reconstruction.
    for epoch in range(n_iters):
        train_loss = 0.0
        for _, _, x, y, patch_stack in data_loader:
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patch, SOS_rec, loss = nf_apact(x, y, patch_stack, delays)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        loss_list.append(train_loss)
        SOS_list.append(SOS_rec.detach().cpu().numpy())
        logger.info("  [{}/{}]  loss={:0.4g} ".format(epoch+1, n_iters, train_loss/len(data_loader)))
        
    # Deconvolution Using recovered SOS.
    logger.info(" Running deconvolution using recovered SOS.")
    with torch.no_grad():
        for i, j, x, y, patch_stack in tqdm(data_loader, desc='Deconvolution'):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            thetas, wfs = wavefront_sos(x, y, SOS_rec)
            TF_stack = tf_pact(delays, thetas, wfs)
            patch_stack = patch_stack * gaussian_window
            rec_patch = mc_deconv(patch_stack, TF_stack)
            # print(rec_patch.shape, IP_rec.shape)
            IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
    IP_rec = IP_rec.detach().cpu().numpy()
    t_end = time()
    
    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_path, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'img')
    
    # Visualization
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(1,2,1)
    norm_SOS  = Normalize(vmax=1610, vmin=1490)
    plt.imshow(SOS_list[-1], cmap='magma', norm=norm_SOS)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_SOS)
    cb.ax.set_yticks([1500, 1520, 1540, 1560, 1580, 1600])
    cb.ax.tick_params(labelsize=11)
    
    ax = plt.subplot(1,2,2)
    norm_IP = Normalize(vmax=5, vmin=-1)
    plt.imshow(standardize(IP_rec), cmap='gray', norm=norm_IP)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_IP)
    cb.set_ticks([5, -1])
    cb.set_ticklabels(['Max', 'Min'], fontsize=11)
    plt.savefig(os.path.join(results_path, 'vis.png'), bbox_inches='tight')
    
    logger.info(' Results saved to "%s".', results_path)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='numerical', choices=['numerical', 'leaf_phantom', 'mouse_liver'])
    parser.add_argument('--method', type=str, default='NF-APACT', choices=['NF-APACT', 'APACT', 'Deconv', 'Dual-SOS_DAS', 'DAS'])
    parser.add_argument('--v_das', type=float, default=1510.0)
    parser.add_argument('--v_sb', type=float, default=1558.0)
    parser.add_argument('--n_delays', type=int, default=32)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--reg', type=str, default='TV', choices=[None, 'TV'])
    parser.add_argument('--weight', type=float, default=0.3e-11)
    args = parser.parse_args()
    
    
    # Load configuration file.
    config = load_config('config.yaml')
    basic_params, task_params = config['basic_params'], config[args.task]

    
    if args.method == 'NF-APACT':
        nf_apact(n_delays=args.n_delays, n_iters=args.n_iters, lr=args.lr, weight=args.weight,
                 basic_params=basic_params, task_params=task_params)
    elif args.method == 'APACT':
        pass
    elif args.method == 'Deconv':
        apact()
    elif args.method == 'Dual-SOS_DAS':
        dual_sos_das(v_sb=args.v_sb, basic_params=basic_params, task_params=task_params)
    elif args.method == 'DAS':
        das(v_das=args.v_das, basic_params=basic_params, task_params=task_params)
    else:
        raise ValueError("Method not supported.")
    

    