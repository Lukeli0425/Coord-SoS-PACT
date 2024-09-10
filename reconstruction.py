import argparse
import logging
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from kwave.ktransducer import kWaveGrid
from models.apact import APACT
from models.das import DelayAndSum, DualSOSDelayAndSum
from models.deconv import MultiChannelDeconv
from models.nf_apact import NFAPACT
from models.pact import SOS2Wavefront, Wavefront2TF
from utils.dataio import *
from utils.dataset import get_jr_dataloader
from utils.reconstruction import *
from utils.simulations import get_water_sos
from utils.utils_torch import get_total_params
from utils.visualization import *

plt.set_loglevel("warning")

DATA_DIR = 'data/'
RESULT_DIR = 'results_new/'


def das(v_das:float, bps:dict, tps:dict) -> None:
    """Run delay-and-sum (DAS) reconstruction with given sound speed.

    Args:
        v_das (float): Assumed constant sound speed.
        bps (dict): Basic parameters.
        tps (dict): Task parameters.
    """
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
    
    das = DelayAndSum(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
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


def dual_sos_das(v_body:float, bps:dict, tps:dict) -> None:
    """Run Dual-SOS delay-and-sum (DAS) reconstruction with given tissue sound speed.

    Args:
        v_body (float): Assumed constant sound speed in the tissue.
        bps (dict): Basic parameters.
        tps (dict): Task parameters.
    """
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
    v0 = get_water_sos(tps['T'])
    
    das_dual = DualSOSDelayAndSum(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], 
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


def deconv(n_delays:int, bps:dict, tps:dict) -> None:
    """Run multi-channel deconvolution with given SOS map.

    Args:
        n_delays (int): Array of extra delay distances for DAS.
        bps (dict): Basic parameters.
        tps (dict): Task parameters.
    """
    params = f'{n_delays}delays'
    logger = logging.getLogger('Deconv')
    logger.info(" Reconstructing %s with Multi-channel Deconvolution (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'Deconv', params)
    os.makedirs(results_dir, exist_ok=True)
    
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    SOS = load_mat(os.path.join(DATA_DIR, tps['SOS']))
    
    # Preparations.
    kgrid = kWaveGrid([tps['Nx'], tps['Ny']], [bps['dx'], bps['dy']])
    x_vec, y_vec = kgrid.x_vec+tps['x_c']*bps['dx'], kgrid.y_vec+tps['y_c']*bps['dy']
    v0 = get_water_sos(tps['T'])
    l_patch, N_patch = bps['l_patch'], bps['N_patch']
    delays = torch.linspace(-8e-4, 8e-4, n_delays).cuda().view(-1,1,1)
    
    das = DelayAndSum(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    sos2wavefront = SOS2Wavefront(R_body=tps['R_body'], v0=v0, x_vec=x_vec, y_vec=y_vec, n_thetas=256, N_int=256)
    sos2wavefront.cuda()
    sos2wavefront.eval()
    wavefront2tf = Wavefront2TF(N=2*N_patch, l=2*l_patch, n_delays=n_delays)
    wavefront2tf.cuda()
    wavefront2tf.eval()
    deconv = MultiChannelDeconv()
    deconv.cuda()
    deconv.eval()
    
    sinogram = torch.from_numpy(sinogram[:,tps['t0']:]).cuda()
    ring_error = torch.from_numpy(ring_error).cuda()
    SOS = torch.from_numpy(SOS).cuda()
    sigma = bps['fwhm'] / 4e-5 / np.sqrt(2*np.log(2))
    gaussian_window = torch.from_numpy(get_gaussian_window(sigma, N_patch)).unsqueeze(0).cuda()
    DAS_stack = []
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
    data_loader = get_jr_dataloader(DAS_stack, l_patch, N_patch)
    
    # Deconvolution.
    logger.info(" Reconstructing initial pressure via Deconvolution.")
    IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
    with torch.no_grad():
        for i, j, x, y, patch_stack in tqdm(data_loader, desc='Deconvolution'):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            thetas, wfs = sos2wavefront(x, y, SOS)
            H = wavefront2tf(delays.view(1,-1,1,1), thetas, wfs)
            patch_stack = patch_stack * gaussian_window
            rec_patch, _, _ = deconv(patch_stack, H)
            IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
    t_end = time()
    IP_rec = IP_rec.detach().cpu().numpy()
    SOS = SOS.detach().cpu().numpy()
    
    # Save results.
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'img')
    
    # Save log.
    log = {'task':tps['task'], 'method':'NF_APACT', 'n_delays':n_delays, 'time':t_end-t_start}
    save_log(results_dir, log)
    
    # Visualization
    visualize_deconv(results_dir, IP_rec, SOS, t_end-t_start, tps['IP_max'], tps['IP_min'], tps['SOS_max'], tps['SOS_min'], params)
    
    logger.info(' Results saved to "%s".', results_dir)


def apact(n_delays:int, n_thetas:int, lam_tsv:float, n_iters:int, lr:float, bps:dict, tps:dict) -> None:
    """Run Adaptive Photoacoustic Computed Tomography (APACT) reconstruction.

    Args:
        n_delays (int): Number of extra delay distances for DAS.
        n_thetas (int): Number of angles in wavefront.
        lam_tsv (float): Weight of total-squared-variation (TSV) regularization for SOS reconstruction.
        n_iters (int): Number of iterations for SOS reconstruction.
        lr (float): Learning rate.
        bps (dict): Basic parameters.
        tps (dict): Task parameters.
    """
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
    v0 = get_water_sos(tps['T'])
    l_patch, N_patch = bps['l_patch'], bps['N_patch']
    delays = torch.linspace(-8e-4, 8e-4, n_delays).cuda().view(-1,1,1)
    
    das = DelayAndSum(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    apact = APACT(delays=delays, lam_tsv=lam_tsv, R_body=tps['R_body'], v0=v0, Nx=tps['Nx'], Ny=tps['Ny'], dx=bps['dx'], dy=bps['dy'], x_vec=kgrid.x_vec, y_vec=kgrid.y_vec, mean=tps['mean'], std=tps['std'], n_thetas=n_thetas, N_patch=N_patch, 
                  generate_TF=True, dc_range=[-2.e-4, 4.e-4], amp=3.5e-4, step=5e-5, data_path=results_dir)
    apact.cuda()
    apact.eval()
    apact.generate_tfs()
    
    DAS_stack, patch_centers, wf_params_list, loss_list = [], [], [], []
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
    data_loader = get_jr_dataloader(DAS_stack, 1, l_patch, N_patch, shuffle=False)
    
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
    save_mat(os.path.join(results_dir, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'img')
    wf_params_list = torch.stack(wf_params_list, dim=0).cuda()
    torch.save(wf_params_list, os.path.join(results_dir, 'wf_params.pth'))
    wf_params_list = torch.load(os.path.join(results_dir, 'wf_params.pth')).cuda()
    lr = 50
    logger.info(' Reconstructing SOS.')
    optimizer = Adam(apact.parameters(), lr=lr)
    apact.train()
    for idx in range(n_iters):
        train_loss = 0.0
        for _, ((x, y), fourier_params) in enumerate(zip(patch_centers, wf_params_list)):
            loss, SOS_rec = apact.optimize_sos(x, y, fourier_params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        loss_list.append(train_loss)
        logger.info(' SOS Reconstruction: [{}/{}] loss={:.3e}'.format(idx+1, n_iters, train_loss))
        
    t_end = time()
    logger.info(" Reconstruction completed in {:.1f}s.".format(t_end-t_start))
    
    SOS_rec = SOS_rec.detach().cpu().numpy()
    save_mat(os.path.join(results_dir, 'SOS_rec.mat'), SOS_rec.swapaxes(0,1), 'SOS')
    IP_rec = load_mat(os.path.join(results_dir, 'IP_rec.mat'))
    
    # Save log.
    # log = load_log(os.path.join(results_dir, 'log.json'))
    log = {'task':tps['task'], 'method':'NF_APACT', 'n_delays':n_delays, 'n_thetas':n_thetas,
           'lam_tsv':lam_tsv,'n_iters':n_iters, 'lr':lr, 'loss':loss_list, 'time':t_end-t_start}
    save_log(results_dir, log)
    
    # Visualization.
    visualize_apact(results_dir, IP_rec, SOS_rec, log['time'], tps['IP_max'], tps['IP_min'], tps['SOS_max'], tps['SOS_min'], params)

    logger.info(' Results saved to "%s".', results_dir)


def nf_apact(n_delays:int, hidden_layers:int, hidden_features:int, pos_encoding:bool, N_freq:int, lam_tv:float, reg, lam, 
             n_epochs:int, batch_size:int, lr:float, bps:dict, tps:dict) -> None:
    """Run Neural Fields for Adaptive Photoacoustic Computed Tomography (NF-APACT) reconstruction.

    Args:
        n_delays (int): Number of extra delay distances for DAS.
        hidden_layers (int): Number of hidden layers in the MLP.
        hidden_features (int): Number of hidden features in each layer.
        pos_encoding (bool): Whether to use positional encoding in the SOS representation.
        N_freq (int): Number of frequencies in the positional encoding.
        lam_tv (float): Weight of total-variation (TV) regularization on SOS.
        n_epochs (int): Number of iterations for SOS reconstruction.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        bps (dict): Basic parameters.
        tps (dict): Task parameters.
    """
    params = f'{n_delays}delays_{hidden_layers}lyrs_{hidden_features}fts' + (f'_PE={N_freq}' if N_freq>0 else '') \
             + ('_TV={:.1e}'.format(lam_tv) if lam_tv != 0 else '') + '_{}epochs_bs={}_lr={:.1e}'.format(n_epochs, batch_size, lr)
    logger = logging.getLogger('NF-APACT')
    logger.info(" Reconstructing %s with NF-APACT (%s).", tps['description'], params)
    results_dir = os.path.join(RESULT_DIR, tps['task'], 'NF-APACT', params)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(DATA_DIR, tps['sinogram'], tps['EIR'], tps['ring_error'])
    
    # Preparations.
    kgrid = kWaveGrid([tps['Nx'], tps['Ny']], [bps['dx'], bps['dy']])
    x_vec, y_vec = kgrid.x_vec+tps['x_c']*bps['dx'], kgrid.y_vec+tps['y_c']*bps['dy']
    v0 = get_water_sos(tps['T'])
    l_patch, N_patch = bps['l_patch'], bps['N_patch']
    delays = torch.linspace(-8e-4, 8e-4, n_delays).cuda().view(-1,1,1)
    
    das = DelayAndSum(R_ring=bps['R_ring'], N_transducer=bps['N_transducer'], T_sample=bps['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()
    nf_apact = NFAPACT(n_delays=n_delays, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq, lam_tv=lam_tv, reg=reg, lam=lam,
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
    data_loader = get_jr_dataloader(DAS_stack, batch_size, l_patch, N_patch)
    
    # Joint Reconstruction.
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
        for idx, (i_list, j_list, x, y, patch_stack) in enumerate(data_loader):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patchs, SOS_rec, loss = nf_apact(x, y, patch_stack, delays)
            for i, j, rec_patch in zip(i_list, j_list, rec_patchs):
                IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(train_loss)
        SOS_list.append(SOS_rec.detach().cpu().numpy())
        IP_list.append(IP_rec.detach().cpu().numpy())
        logger.info("  [{}/{}]  loss={:0.4g} ".format(epoch, n_epochs, train_loss))
        
    # Deconvolution.
    logger.info(" Reconstructing initial pressure via Deconvolution.")
    nf_apact.save_sos()
    nf_apact.eval()
    test_loss = 0.0
    IP_rec = torch.zeros((tps['Nx'], tps['Ny'])).cuda()
    with torch.no_grad():
        for i_list, j_list, x, y, patch_stack in tqdm(data_loader, desc='Deconvolution'):
            x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
            rec_patchs, SOS_rec, loss = nf_apact(x, y, patch_stack, delays, task='test')
            for i, j, rec_patch in zip(i_list, j_list, rec_patchs):
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
           'reg':reg, 'lam':lam,'n_epochs':n_epochs, 'lr':lr, 'loss':loss_list, 'time':t_end-t_start}
    save_log(results_dir, log)

    # Visualization
    visualize_nf_apact(results_dir, IP_list[-1], SOS_list[-1], loss_list, t_end-t_start, tps['IP_max'], tps['IP_min'], tps['SOS_max'], tps['SOS_min'], params)
    make_video(results_dir, loss_list, tps)
    # make_video_icon(results_dir, loss_list, tps)
    
    logger.info(' Results saved to "%s".', results_dir)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
    parser.add_argument('--task', type=str, default='numerical', choices=['numerical', 'phantom', 'in_vivo', 'kidney'], help='Task to be reconstructed.')
    parser.add_argument('--method', type=str, default='NF-APACT', choices=['NF-APACT', 'APACT', 'Deconv', 'Dual-SOS_DAS', 'DAS'], help='Method to be used for reconstruction.')
    parser.add_argument('--v_das', type=float, default=1510.0, help='Speed of sound for DAS.')
    parser.add_argument('--v_body', type=float, default=1560.0, help='Speed of sound in the tissue for Dual-SOS DAS.')
    parser.add_argument('--n_delays', type=int, default=32, help='Number of delays used in NF-APACT.')
    parser.add_argument('--n_iters', type=int, default=20, help='Number of iterations for SOS reconstruction in APACT.')
    parser.add_argument('--lam_tsv', type=float, default=5e-15, help='Weight of total squared variation regularization for APACT.')
    parser.add_argument('--hidden_lyrs', type=int, default=1, help='Number of hidden layers in NF-APACT.')
    parser.add_argument('--hidden_fts', type=int, default=64, help='Number of hidden features in NF-APACT.')
    parser.add_argument('--n_freq', type=int, default=0, help='Number of frequencies used for positioanl encoding in NF-APACT.')
    parser.add_argument('--n_thetas', type=int, default=256, help='Number of angles used in wavefront calculation.')
    parser.add_argument('--lam_tv', type=float, default=0.0)
    parser.add_argument('--reg', type=str, default='None', choices=['None', 'Brenner', 'Tenenbaum', 'Variance'])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs for NF-APACT.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for NF-APACT.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for NF-APACT.')
    args = parser.parse_args()
    
    # Select GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load parameters.
    config = load_config('config.yaml')
    bps, tps = config['basic_params'], config[args.task]

    # Run reconstruction.
    if args.method == 'NF-APACT':
        nf_apact(n_delays=args.n_delays, hidden_layers=args.hidden_lyrs, hidden_features=args.hidden_fts, pos_encoding=args.n_freq>2, N_freq=args.n_freq, 
                 lam_tv=args.lam_tv, reg=args.reg, lam=args.lam, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, bps=bps, tps=tps)
    elif args.method == 'APACT':
        apact(n_delays=args.n_delays, n_thetas=args.n_thetas, lam_tsv=args.lam_tsv, n_iters=args.n_iters, lr=args.lr, bps=bps, tps=tps)
    elif args.method == 'Deconv':
        deconv(n_delays=args.n_delays, bps=bps, tps=tps)
    elif args.method == 'Dual-SOS_DAS':
        dual_sos_das(v_body=args.v_body, bps=bps, tps=tps)
    elif args.method == 'DAS':
        das(v_das=args.v_das, bps=bps, tps=tps)
    else:
        raise NotImplementedError("Method not supported.")
    