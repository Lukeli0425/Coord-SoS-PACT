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
from models.nf_apact import NF_APACT
from utils.dataio import *
from utils.dataset import get_jr_dataloader
from utils.reconstruction import *
from utils.simulations import get_water_SOS
from utils.utils_torch import get_total_params
from utils.visualization import *

plt.set_loglevel("warning")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_DIR = 'data/'
RESULT_DIR = 'results_new/'


def das(v_das, bps, tps):
    params = 'v_das={:.1f}m·s⁻¹'.format(v_das)
    logger = logging.getLogger('DAS')
    logger.info(" Reconstructing %s with Delay-and-Sum (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'DAS', params)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    # Preparations.
    kgrid = kWaveGrid([tps['Nx'], tps['Ny']], [bps['dx'], bps['dy']])
    x_vec, y_vec = kgrid.x_vec + tps['x_c']*bps['dx'], kgrid.y_vec + tps['y_c']*bps['dy']
    
    das = DAS(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()

    sinogram = torch.from_numpy(sinogram[:,tps['t0']:]).cuda()
    ring_error = torch.from_numpy(ring_error).cuda()
    with torch.no_grad():
        t_start = time()
        IP_rec = das(sinogram=sinogram, v0=v_das, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()

    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    log = {'task':tps['task'], 'method':'DAS', 'v_das':v_das, 'time':t_end-t_start}
    save_log(results_dir, log)
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title(f"DAS Reconstruction ({params})", fontsize=16)
    plt.text(430, 25, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visualization.png'))
    
    logger.info(' Results saved in "%s".', results_dir)

def dual_sos_das(v_body, bps, tps):
    params = 'v_body={:.1f}m·s⁻¹'.format(v_body)
    logger = logging.getLogger('Dual-SOS DAS')
    logger.info(" Reconstructing %s with Dual-SOS DAS (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'Dual-SOS_DAS', params)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    # Preparations.
    Nx, Ny = tps['Nx'], tps['Ny']
    dx, dy = bps['dx'], bps['dy']
    x_c, y_c = tps['x_c'], tps['y_c']
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    v0 = get_water_SOS(tps['T'])
    
    das_dual = Dual_SOS_DAS(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], 
                            x_vec=x_vec, y_vec=y_vec, R_body=tps['R_body'], center=(x_c*dx, y_c*dy), mode='zero')
    das_dual.cuda()
    das_dual.eval()

    sinogram = torch.from_numpy(sinogram[:,tps['t0']:]).cuda()
    ring_error = torch.from_numpy(ring_error).cuda()
    with torch.no_grad():
        t_start = time()
        IP_rec = das_dual(sinogram=sinogram, v0=v0, v1=v_body, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()
        
    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    log = {'task':tps['task'], 'method':'Dual-SOS DAS', 'v_body':v_body, 'time':t_end-t_start}
    save_log(results_dir, log)
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title(f"Dual SOS DAS Reconstruction ({params})", fontsize=16)
    plt.text(430, 25, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visualization.png'))
    
    logger.info(' Results saved in "%s".', results_dir)


def deconv(n_delays, bps, tps):
    params = ''
    logger = logging.getLogger('Deconv')
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'Deconv', params)
    os.makedirs(results_dir, exist_ok=True)
    
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    logger.info(' Results saved to "%s".', results_dir)


def apact(n_delays, lam_tv, n_iters, lr, bps, tps):
    params = f'{n_delays}delays'
    logger = logging.getLogger('APACT')
    logger.info(" Reconstructing %s with APACT (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'APACT', params)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    # Preparations.
    kgrid = kWaveGrid([tps['Nx'], tps['Ny']], [bps['dx'], bps['dy']])
    x_vec, y_vec = kgrid.x_vec+tps['x_c']*bps['dx'], kgrid.y_vec+tps['y_c']*bps['dy']
    v0 = get_water_SOS(tps['T'])
    l_patch, N_patch = bps['l_patch'], bps['N_patch']
    delays = torch.linspace(-8e-4, 8e-4, n_delays).cuda().view(-1,1,1)
    
    das = DAS(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    apact = APACT(delays=delays, lam_tv=lam_tv, R_body=tps['R_body'], v0=v0, Nx=tps['Nx'], Ny=tps['Ny'], dx=bps['dx'], dy=bps['dy'], x_vec=kgrid.x_vec, y_vec=kgrid.y_vec, mean=tps['mean'], std=tps['std'], N_patch=N_patch, 
                  generate_TF=False, dc_range=[-1e-4, 2.6e-4], amp=3.2e-4, step=4e-5, data_path=results_dir)
    apact.cuda()
    apact.eval()
    
    DAS_stack, patch_centers, wf_params_list = [], [], []
    sinogram = torch.from_numpy(sinogram[:,tps['t0']:]).cuda()
    ring_error = torch.from_numpy(ring_error).cuda()
    IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
    
    t_start = time()
    # DAS.
    logger.info(" Running DAS (v_das=%.1fm·s⁻¹) with %s delays.", v0, n_delays)
    with torch.no_grad():
        for d_delay in tqdm(delays, desc='DAS'):
            recon = das(sinogram=sinogram, v0=v0, d_delay=d_delay, ring_error=ring_error)
            DAS_stack.append(recon)
    DAS_stack = torch.stack(DAS_stack, dim=0)
    DAS_stack = (DAS_stack - DAS_stack.mean()) / DAS_stack.std()
    data_loader = get_jr_dataloader(DAS_stack, l_patch, N_patch, shuffle=False)
    
    # Wavefront search.
    logger.info(' Searching for optimal wavefront parameters.')
    with torch.no_grad():
        for i, j, x, y, patch_stack in tqdm(data_loader, desc='Wavefront Search'):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patch, _, wf_coeff, _ = apact(patch_stack)
            IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
            wf_params_list.append(wf_coeff)
            patch_centers.append((x, y))
    IP_rec = IP_rec.detach().cpu().numpy()
    # apact.save_wavefront_params()
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'img')
    wf_params_list = torch.stack(wf_params_list, dim=0).cuda()
    torch.save(wf_params_list, os.path.join(results_dir, 'wf_params.pth'))
    wf_params_list = torch.load(os.path.join(results_dir, 'wf_params.pth')).cuda()
    
    logger.info(' Reconstruting SOS.')
    optimizer = Adam(apact.parameters(), lr=lr)
    # apact.prepare_SOS_reconstruction(torch.tensor(patch_centers).cuda())
    apact.train()
    for idx in range(n_iters):
        train_loss = 0.0
        for _, ((x, y), fourier_params) in enumerate(zip(patch_centers, wf_params_list)):
            loss, SOS_rec = apact.optimize_SOS(x, y, fourier_params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logger.info(' SOS Reconstruction: [{}/{}] loss={:.7e}'.format(idx+1, n_iters, train_loss))
        
    t_end = time()
    logger.info(" Reconstruction completed in {:.1f}s.".format(t_end-t_start))
    
    SOS_rec = SOS_rec.detach().cpu().numpy()
    save_mat(os.path.join(results_dir, 'SOS_rec.mat'), SOS_rec.swapaxes(0,1), 'SOS')
    IP_rec = load_mat(os.path.join(results_dir, 'IP_rec.mat'))
    # Visualization.
    visualize_apact(results_dir, IP_rec, SOS_rec, t_end-t_start, tps['IP_max'], tps['IP_min'], tps['SOS_max'], tps['SOS_min'], params)

    logger.info(' Results saved to "%s".', results_dir)


def nf_apact(n_delays, hidden_layers, hidden_features, pos_encoding, N_freq, lam_tv, reg, lam, n_iters, lr, bps, tps):
    params = f'{n_delays}delays_{hidden_layers}lyrs_{hidden_features}fts' + (f'_PE={N_freq}' if N_freq>0 else '') \
             + ('_TV={:.1e}'.format(lam_tv) if lam_tv != 0 else '') + '_lr={:.1e}_{}iters'.format(lr, n_iters)
    logger = logging.getLogger('NF-APACT')
    logger.info(" Reconstructing %s with NF-APACT (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'NF-APACT', params)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    # Preparations.
    kgrid = kWaveGrid([tps['Nx'], tps['Ny']], [bps['dx'], bps['dy']])
    x_vec, y_vec = kgrid.x_vec+tps['x_c']*bps['dx'], kgrid.y_vec+tps['y_c']*bps['dy']
    v0 = get_water_SOS(tps['T'])
    l_patch, N_patch = bps['l_patch'], bps['N_patch']
    delays = torch.linspace(-8e-4, 8e-4, n_delays).cuda().view(-1,1,1)
    
    das = DAS(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    nf_apact = NF_APACT(mode='SIREN', n_delays=n_delays, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq, lam_tv=lam_tv, reg=reg, lam=lam,
                        x_vec=kgrid.x_vec, y_vec=kgrid.y_vec, R_body=tps['R_body'], v0=v0, mean=tps['mean'], std=tps['std'], N_patch=N_patch, l_patch=l_patch, angle_range=(0, 2*torch.pi))
    nf_apact.cuda()
    nf_apact.train()
    logger.info(" Number of learnable parameters: %s", get_total_params(nf_apact))

    optimizer = Adam(params=nf_apact.parameters(), lr=lr)
    DAS_stack, loss_list, SOS_list, IP_list = [], [], [], []
    sinogram = torch.from_numpy(sinogram[:,tps['t0']:]).cuda()
    ring_error = torch.from_numpy(ring_error).cuda()
    
    t_start = time()
    # DAS.
    logger.info(" Running DAS (v_das=%.1fm·s⁻¹) with %s delays.", v0, n_delays)
    with torch.no_grad():
        for d_delay in tqdm(delays, desc='DAS'):
            recon = das(sinogram=sinogram, v0=v0, d_delay=d_delay, ring_error=ring_error)
            DAS_stack.append(recon)
    DAS_stack = torch.stack(DAS_stack, dim=0)
    DAS_stack = (DAS_stack - DAS_stack.mean()) / DAS_stack.std()
    data_loader = get_jr_dataloader(DAS_stack, l_patch, N_patch)
    
    # Joint Reconstruction.
    for epoch in range(1, n_iters+1):
        train_loss = 0.0
        IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
        for i, j, x, y, patch_stack in data_loader:
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patch, SOS_rec, loss = nf_apact(x, y, patch_stack, delays)
            IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(train_loss)
        SOS_list.append(SOS_rec.detach().cpu().numpy())
        IP_list.append(IP_rec.detach().cpu().numpy())
        logger.info("  [{}/{}]  loss={:0.4g} ".format(epoch, n_iters, train_loss))
        
    # Deconvolution.
    logger.info(" Reconstructing initial pressure via Deconvolution.")
    nf_apact.save_SOS()
    nf_apact.eval()
    test_loss = 0.0
    IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
    with torch.no_grad():
        for i, j, x, y, patch_stack in tqdm(data_loader, desc='Deconvolution'):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patch, SOS_rec, loss = nf_apact(x, y, patch_stack, delays, task='test')
            IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
            test_loss += loss.item()
    t_end = time()
    
    loss_list.append(test_loss)
    SOS_list.append(SOS_rec.detach().cpu().numpy())
    IP_list.append(IP_rec.detach().cpu().numpy())
    
    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    
    # Save final results.
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_list[-1].swapaxes(0,1), 'img')
    save_mat(os.path.join(results_dir, 'SOS_rec.mat'), SOS_list[-1].swapaxes(0,1), 'SOS')
    
    # Save SOS and IP after every iteration to make video.
    os.makedirs(os.path.join(results_dir, 'video'), exist_ok=True)
    for idx, (IP, SOS, loss) in enumerate(zip(IP_list, SOS_list, loss_list)):
        save_mat(os.path.join(results_dir, 'video', f'IP_rec_{idx}.mat'), IP.swapaxes(0,1), 'img')
        save_mat(os.path.join(results_dir, 'video', f'SOS_rec_{idx}.mat'), SOS.swapaxes(0,1), 'SOS')
    
    # Save log.
    log = {'task':tps['task'], 'method':'NF_APACT', 'n_delays':n_delays, 
           'hidden_layers':hidden_layers, 'hidden_features':hidden_features, 'pos_encoding':pos_encoding, 'N_freq':N_freq, 'n_params':get_total_params(nf_apact),
           'reg':reg, 'lam':lam,'n_iters':n_iters, 'lr':lr, 'loss':loss_list, 'time':t_end-t_start}
    save_log(results_dir, log)
    
    # Visualization
    visualize_nf_apact(results_dir, IP_list[-1], SOS_list[-1], loss_list, t_end-t_start, 
                       tps['IP_max'], tps['IP_min'], tps['SOS_max'], tps['SOS_min'], params)
    make_video(results_dir, loss_list, tps)
    
    logger.info(' Results saved to "%s".', results_dir)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
    parser.add_argument('--task', type=str, default='numerical', choices=['numerical', 'phantom', 'in_vivo'], help='Task to be reconstructed.')
    parser.add_argument('--method', type=str, default='NF-APACT', choices=['NF-APACT', 'APACT', 'Deconv', 'Dual-SOS_DAS', 'DAS'], help='Method to be used for reconstruction.')
    parser.add_argument('--v_das', type=float, default=1510.0, help='Speed of sound for DAS.')
    parser.add_argument('--v_body', type=float, default=1560.0, help='Speed of sound in the tissue for Dual-SOS DAS.')
    parser.add_argument('--n_delays', type=int, default=32, help='Number of delays used in NF-APACT.')
    parser.add_argument('--hidden_lyrs', type=int, default=1, help='Number of hidden layers in NF-APACT.')
    parser.add_argument('--hidden_fts', type=int, default=64, help='Number of hidden features in NF-APACT.')
    parser.add_argument('--n_freq', type=int, default=0, help='Number of frequencies used for positioanl encoding in NF-APACT.')
    parser.add_argument('--lam_tv', type=float, default=0.0)
    parser.add_argument('--reg', type=str, default='None', choices=['None', 'Brenner', 'Tenenbaum', 'Variance'])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--n_iters', type=int, default=30, help='Number of training iterations for NF-APACT.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for NF-APACT.')
    args = parser.parse_args()
    
    # Select GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load configuration file.
    config = load_config('config.yaml')
    bps, tps = config['basic_params'], config[args.task]

    # Reconstruction.
    if args.method == 'NF-APACT':
        nf_apact(n_delays=args.n_delays, hidden_layers=args.hidden_lyrs, hidden_features=args.hidden_fts, pos_encoding=args.n_freq>2, N_freq=args.n_freq, 
                 lam_tv=args.lam_tv, reg=args.reg, lam=args.lam, n_iters=args.n_iters, lr=args.lr, bps=bps, tps=tps)
    elif args.method == 'APACT':
        apact(n_delays=args.n_delays, lam_tv=args.lam_tv, n_iters=args.n_iters, lr=args.lr, bps=bps, tps=tps)
    elif args.method == 'Deconv':
        deconv()
    elif args.method == 'Dual-SOS_DAS':
        dual_sos_das(v_body=args.v_body, bps=bps, tps=tps)
    elif args.method == 'DAS':
        das(v_das=args.v_das, bps=bps, tps=tps)
    else:
        raise NotImplementedError("Method not supported.")
    