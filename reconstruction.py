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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = 'data/'
RESULT_DIR = 'results_new/'


def das(v_das, basic_params, task_params):
    params = 'v_das={:.1f}m·s⁻¹'.format(v_das)
    logger = logging.getLogger('DAS')
    logger.info(" Reconstructing %s with Delay-and-Sum (%s).", task_params['description'], params)
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'DAS', params)
    os.makedirs(results_path, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(data_dir=DATA_DIR, 
                                             sinogram_file=task_params['sinogram'], 
                                             EIR_file=task_params['EIR'], 
                                             ring_error_file=task_params['ring_error'])
    # Preparations.
    Nx, Ny = task_params['Nx'], task_params['Ny']
    dx, dy = basic_params['dx'], basic_params['dy']
    x_c, y_c = task_params['x_c'], task_params['y_c']
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    
    das = DAS(R_ring=basic_params['R_ring'], N_transducer=basic_params['N_transducer'], T_sample=basic_params['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    das.cuda()
    das.eval()

    sinogram = torch.tensor(sinogram[:,task_params['t0']:]).cuda()
    ring_error = torch.tensor(ring_error).cuda()
    with torch.no_grad():
        t_start = time()
        IP_rec = das(sinogram=sinogram, v0=v_das, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()

    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_path, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title(f"DAS Reconstruction ({params})", fontsize=16)
    plt.text(430, 25, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'visualization.png'))
    
    logger.info(' Results saved in "%s".', results_path)

def dual_sos_das(v_body, basic_params, task_params):
    params = 'v_body={:.1f}m·s⁻¹'.format(v_body)
    logger = logging.getLogger('Dual-SOS DAS')
    logger.info(" Reconstructing %s with Dual-SOS DAS (%s).", task_params['description'], params)
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'Dual-SOS_DAS', params)
    os.makedirs(results_path, exist_ok=True)
    
    # Load data.
    sinogram, EIR, ring_error = prepare_data(data_dir=DATA_DIR, 
                                             sinogram_file=task_params['sinogram'], 
                                             EIR_file=task_params['EIR'], 
                                             ring_error_file=task_params['ring_error'])
    # Preparations.
    Nx, Ny = task_params['Nx'], task_params['Ny']
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
        IP_rec = das_dual(sinogram=sinogram, v0=v0, v1=v_body, d_delay=0, ring_error=ring_error).detach().cpu().numpy()
        t_end = time()
        
    logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    save_mat(os.path.join(results_path, 'IP_rec.mat'), IP_rec.swapaxes(0,1), 'IP')
    
    # Visualization.
    plt.figure(figsize=(7,7))
    plt.imshow(IP_rec, cmap='gray')
    plt.title(f"Dual SOS DAS Reconstruction ({params})", fontsize=16)
    plt.text(430, 25, "t = {:.4f} s".format(t_end-t_start), color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'visualization.png'))
    
    logger.info(' Results saved in "%s".', results_path)


def deconv(basic_params, task_params):
    logger = logging.getLogger('Deconv')
    logger.info(' Results saved to "%s".', results_path)

def apact(basic_params, task_params):
    logger = logging.getLogger('APACT')
    logger.info(" Reconstructing %s with APACT.", task_params['task'])

def nf_apact(n_delays, hidden_layers, hidden_features, pos_encoding, N_freq, lam_tv, reg_IP, lam_ip, n_iters, lr, basic_params, task_params):
    title = '{}delays_{}lyrs_{}fts_PE={}_TV={:.1e}_{}={:.1e}_lr={:.1e}_{}iters'.format(n_delays, hidden_layers, hidden_features, N_freq, lam_tv, reg_IP, lam_ip, lr, n_iters)
    logger = logging.getLogger('NF-APACT')
    logger.info(" Reconstructing %s with NF-APACT (%s).", task_params['description'], title)
    results_path = os.path.join(RESULT_DIR, task_params['task'], 'NF-APACT', title)
    os.makedirs(results_path, exist_ok=True)
    
    # # Load data.
    # sinogram, EIR, ring_error = prepare_data(data_dir=DATA_DIR, 
    #                                          sinogram_file=task_params['sinogram'], 
    #                                          EIR_file=task_params['EIR'], 
    #                                          ring_error_file=task_params['ring_error'])
    # # Preparations.
    # Nx, Ny = task_params['Nx'], task_params['Ny']
    # dx, dy = basic_params['dx'], basic_params['dy']
    # x_c, y_c = task_params['x_c'], task_params['y_c']
    # kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    # x_vec, y_vec = kgrid.x_vec+x_c*dx, kgrid.y_vec+y_c*dy
    # v0 = get_water_SOS(task_params['T'])
    # l_patch, N_patch = basic_params['l_patch'], basic_params['N_patch']
    # R_body = task_params['R_body']
    # mean, std = task_params['mean'], task_params['std']
    
    # das = DAS(R_ring=basic_params['R_ring'], N_transducer=basic_params['N_transducer'], T_sample=basic_params['T_sample'], x_vec=x_vec, y_vec=y_vec, mode='zero')
    # das.cuda()
    # das.eval()
    
    # delays = np.linspace(-8e-4, 8e-4, n_delays)
    # IP_rec = torch.zeros((Nx, Ny)).cuda()
    # delays = torch.tensor(delays).cuda().view(-1,1,1)
    # sigma = basic_params['fwhm'] / 4e-5 / np.sqrt(2*np.log(2))
    # gaussian_window = torch.tensor(get_gaussian_window(sigma, 80)).cuda()

    # wavefront_sos = Wavefront_SOS(R_body, v0, kgrid.x_vec, kgrid.y_vec, n_points=180)
    # wavefront_sos.cuda()
    # wavefront_sos.eval()
    
    # tf_pact = TF_PACT(N=2*N_patch, l=2*l_patch, n_delays=n_delays)
    # tf_pact.cuda()
    # tf_pact.eval()

    # mc_deconv = MultiChannel_Deconv(N_patch=N_patch, l_patch=l_patch)
    # mc_deconv.cuda()
    # mc_deconv.eval()
    
    # nf_apact = NF_APACT(mode='SIREN', n_delays=n_delays, hidden_layers=hidden_layers, hidden_features=hidden_features, pos_encoding=pos_encoding, N_freq=N_freq, lam_tv=lam_tv, lam_ip=lam_ip,
    #                     x_vec=kgrid.x_vec, y_vec=kgrid.y_vec, R_body=R_body, v0=v0, mean=mean, std=std, N_patch=N_patch, l_patch=l_patch, angle_range=(0, 2*torch.pi))
    # nf_apact.cuda()
    # nf_apact.train()
    # logger.info(" Number of learnable parameters: %s", get_total_params(nf_apact))

    # optimizer = Adam(params=nf_apact.parameters(), lr=lr)
    # DAS_stack, loss_list, SOS_list, IP_list = [], [], [], []
    
    # sinogram = torch.tensor(sinogram[:,task_params['t0']:]).cuda()
    # ring_error = torch.tensor(ring_error).cuda()
    # t_start = time()
    # logger.info(" Running DAS (v_das=%.1fm·s⁻¹) with %s delays.", v0, n_delays)
    # with torch.no_grad():
    #     for d_delay in tqdm(delays, desc='DAS'):
    #         recon = das(sinogram=sinogram, v0=v0, d_delay=d_delay, ring_error=ring_error)
    #         DAS_stack.append(recon)
    # DAS_stack = torch.stack(DAS_stack, dim=0)
    # DAS_stack = (DAS_stack - DAS_stack.mean()) / DAS_stack.std()
    # data_loader = get_jr_dataloader(DAS_stack, l_patch, N_patch)
    
    # # Joint Reconstruction.
    # for epoch in range(n_iters+1):
    #     train_loss = 0.0
    #     IP_rec = torch.zeros((Nx, Ny)).cuda()
    #     for i, j, x, y, patch_stack in data_loader:
    #         x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
    #         rec_patch, SOS_rec, loss = nf_apact(x, y, patch_stack, delays)
    #         IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
    #         train_loss += loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #     loss_list.append(train_loss)
    #     SOS_list.append(SOS_rec.detach().cpu().numpy())
    #     IP_list.append(IP_rec.detach().cpu().numpy())
        
    #     logger.info("  [{}/{}]  loss={:0.4g} ".format(epoch, n_iters, train_loss))
        
    # # # Deconvolution Using recovered SOS.
    # # logger.info(" Running deconvolution using recovered SOS.")
    # # with torch.no_grad():
    # #     for i, j, x, y, patch_stack in tqdm(data_loader, desc='Deconvolution'):
    # #         x, y, patch_stack = x.cuda(), y.cuda(), patch_stack.cuda()
    # #         thetas, wfs = wavefront_sos(x, y, SOS_rec)
    # #         TF_stack = tf_pact(delays, thetas, wfs)
    # #         patch_stack = patch_stack * gaussian_window
    # #         rec_patch = mc_deconv(patch_stack, TF_stack)
    # #         IP_rec[20*i:20*i+80, 20*j:20*j+80] += rec_patch.squeeze(0).squeeze(0)
    # # IP_rec = IP_rec.detach().cpu().numpy()
    # t_end = time()
    
    # logger.info(" Reconstruction completed in {:.4f}s.".format(t_end-t_start))
    # save_mat(os.path.join(results_path, 'IP_rec.mat'), IP_list[-1].swapaxes(0,1), 'img')
    # save_mat(os.path.join(results_path, 'SOS_rec.mat'), SOS_list[-1].swapaxes(0,1), 'SOS')
    
    # for idx, (IP, SOS, loss) in enumerate(zip(IP_list, SOS_list, loss_list)):
    #     save_mat(os.path.join(results_path, 'video', f'IP_rec_{idx}.mat'), IP.swapaxes(0,1), 'img')
    #     save_mat(os.path.join(results_path, 'video', f'SOS_rec_{idx}.mat'), SOS.swapaxes(0,1), 'SOS')
        
    # # Visualization
    # fig = plt.figure(figsize=(13,11))
    # gs = gridspec.GridSpec(2, 2)
    # ax = plt.subplot(gs[0:1,:])
    # plt.plot(range(0, n_iters+1), loss_list, '-o', markersize=4.5, linewidth=2, label='loss')
    # # plt.title("", fontsize=16)
    # plt.title("t = {:.2f} s".format(t_end-t_start), loc='right', x=0.94, y=0.91, color='black', fontsize=15)
    # plt.xlabel("Iteration", fontsize=15)
    # plt.ylabel("Loss", fontsize=15)
    
    # ax = plt.subplot(gs[1:2,0:1])
    # norm_IP = Normalize(vmax=task_params['IP_max'], vmin=task_params['IP_min'])
    # plt.imshow(standardize(IP_list[-1]), cmap='gray', norm=norm_IP)
    # plt.title("Reconstructed Initial Pressure", fontsize=16)
    # plt.axis('off')
    # cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    # cb = plt.colorbar(cax=cax, norm=norm_IP)
    # cb.set_ticks([task_params['IP_max'], task_params['IP_min']])
    # cb.set_ticklabels(['Max', 'Min'], fontsize=13)
    
    
    # ax = plt.subplot(gs[1:2,1:2])
    # norm_SOS  = Normalize(vmax=task_params['SOS_max'], vmin=task_params['SOS_min'])
    # plt.imshow(SOS_list[-1], cmap='magma', norm=norm_SOS)
    # plt.title("Reconstructed Speed of Sound", fontsize=16)
    # plt.axis('off')
    # cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    # cb = plt.colorbar(cax=cax, norm=norm_SOS)
    # # cb.ax.set_yticks([1500, 1520, 1540, 1560, 1580, 1600])
    # cb.ax.tick_params(labelsize=12)
    # cb.set_label("$m \cdot s^{-1}$", fontsize=12)
    # plt.savefig(os.path.join(results_path, 'visualization.png'), bbox_inches='tight')
    
    # logger.info(' Results saved to "%s".', results_path)
    loss_list = np.arange(0, n_iters+1)
    make_video(results_path, loss_list, task_params)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='numerical', choices=['numerical', 'phantom', 'in_vivo'])
    parser.add_argument('--method', type=str, default='NF-APACT', choices=['NF-APACT', 'APACT', 'Deconv', 'Dual-SOS_DAS', 'DAS'])
    parser.add_argument('--v_das', type=float, default=1510.0)
    parser.add_argument('--v_body', type=float, default=1560.0)
    parser.add_argument('--n_delays', type=int, default=32)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--hidden_features', type=int, default=96)
    parser.add_argument('--N_freq', type=int, default=4)
    parser.add_argument('--lam_tv', type=float, default=1.2e-4)
    parser.add_argument('--reg_IP', type=str, default='Brenner', choices=['Brenner', 'Tenenbaum', 'Variance'])
    parser.add_argument('--lam_IP', type=float, default=0.0)
    parser.add_argument('--n_iters', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--reg', type=str, default='TV', choices=[None, 'TV'])
    args = parser.parse_args()
    
    
    # Load configuration file.
    config = load_config('config.yaml')
    basic_params, task_params = config['basic_params'], config[args.task]

    
    if args.method == 'NF-APACT':
        nf_apact(n_delays=args.n_delays, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features, pos_encoding=args.N_freq>2, N_freq=args.N_freq, 
                 lam_tv=args.lam_tv, reg_IP=args.reg_IP, lam_ip=args.lam_IP, 
                 n_iters=args.n_iters, lr=args.lr,
                 basic_params=basic_params, task_params=task_params)
    elif args.method == 'APACT':
        pass
    elif args.method == 'Deconv':
        apact()
    elif args.method == 'Dual-SOS_DAS':
        dual_sos_das(v_body=args.v_body, basic_params=basic_params, task_params=task_params)
    elif args.method == 'DAS':
        das(v_das=args.v_das, basic_params=basic_params, task_params=task_params)
    else:
        raise ValueError("Method not supported.")
    

    