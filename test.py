import argparse
import json
import logging
import os
import time

import torch
from tqdm import tqdm

from models.ResUNet import ResUNet
from models.WienerNet import WienerNet
from models.Unrolled_ADMM import Unrolled_ADMM
from utils.dataset import get_dataloader
from utils.utils_test import delta_2D, estimate_shear


def test_shear(model_name, n_iters, model_file, n_gal, snrs, data_path, result_path):
    logger = logging.getLogger('Shear Test')
    logger.info(' Testing method: %s', method)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    psf_delta = delta_2D(48, 48)
    
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')
    
    # Load the model.
    if 'Unrolled_ADMM' in model_name:
        model = Unrolled_ADMM (n_iters=n_iters, n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'WienerNet' in model_name:
        model = WienerNet(n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'ResUNet' in model_name:
        model = ResUNet(in_nc=8, out_nc=1, nc=[nc, nc*2, nc*4, nc*8])
    model.to(device)

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    for snr in snrs:
        logger.info(' Running shear test with %s SNR=%s galaxies.\n', n_gal, snr)
        test_loader = get_dataloader(data_path=data_path, train=False,
                                     obs_folder=f'obs_{snr}/', gt_folder=f'gt_{snr}/')
        
        rec_shear, gt_shear = [], []
        for ((obs, psf, alpha), gt), idx in zip(test_loader, tqdm(range(n_gal))):
            with torch.no_grad():
                if method == 'No_Deconv':
                    gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    gt_shear.append(estimate_shear(gt, psf_delta))
                    rec_shear.append(estimate_shear(obs, psf_delta))
                elif method == 'FPFS':
                    psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(obs, psf))
                elif method == 'Wiener':
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                elif 'Richard-Lucy' in method:
                    obs, psf = obs.to(device), psf.to(device)
                    rec = model(obs, psf) 
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
                else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                    obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                    rec = model(obs, psf, alpha)
                    rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                    rec_shear.append(estimate_shear(rec, psf_delta))
        
        # Save results.
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(" Successfully loaded in %s.", results_file)
        except:
            results = {} 
            logger.critical(" Failed loading in %s.", results_file)
            
        if str(snr) not in results:
            results[str(snr)] = {}
        results[str(snr)]['rec_shear'] = rec_shear
        if method == 'No_Deconv':
            results[str(snr)]['gt_shear'] = gt_shear
        
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.info(" Shear test results saved to %s.\n", results_file)
    


def test_time(method, n_iters, model_file, n_gal, data_path, result_path):  
    """Test the time consumption of different methods."""
    logger = logging.getLogger('Time Test')
    logger.info(' Running time test with %s galaxies.', n_gal)
    logger.info(' Testing method: %s', method)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False)
    
    psf_delta = delta_2D(48, 48)
    
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')

    # Load the model.
    model = None
    if method == 'Wiener':
        model = Wiener()
    elif 'Richard-Lucy' in method:
        model = Richard_Lucy(n_iters=n_iters)
    elif method == 'Tikhonet':
        model = Tikhonet(filter='Identity')
    elif method == 'ShapeNet' or 'Laplacian' in method:
        model = Tikhonet(filter='Laplacian')
    elif 'Gaussian' in method:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Gaussian', PnP=True)
    else:
        model = Unrolled_ADMM(n_iters=n_iters, llh='Poisson', PnP=True)

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()

    rec_shear = []
    time_start = time.time()
    for ((obs, psf, alpha), gt), idx in zip(test_loader, tqdm(range(n_gal))):
        with torch.no_grad():
            if method == 'No_Deconv':
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(obs, psf_delta))
            elif method == 'FPFS':
                psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(obs, psf))
            elif method == 'Wiener':
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha)
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            elif 'Richard-Lucy' in method:
                obs, psf = obs.to(device), psf.to(device)
                rec = model(obs, psf) 
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha)
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
                
    time_end = time.time()
    logger.info(' Tested %s on %s galaxies: Time = {:.4g}s.'.format(time_end-time_start),method, n_gal)

    # Save test results.
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(" Successfully loaded in %s.", results_file)
    except:
        results = {} 
        logger.critical(" Failed loading in %s.", results_file)
    results['time'] = (time_end-time_start, n_gal)
    
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logger.info(" Time test results saved to %s.\n", results_file)
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='shear', choices=['PSNR', 'SSIM', 'time'])
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--result_path', type=str, default='results/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    # Uncomment the methods to be tested.
    methods = {
        'No_Deconv': (0, None), 
        'ResUNet': (0, None), 
        'WienerNet': (0, None), 
        'Unrolled_ADMM(4)': (4, "saved_models/Gaussian_PnP_ADMM_4iters_MultiScale_20epochs.pth"), 
        'Unrolled_ADMM(8)': (8, "saved_models/Gaussian_PnP_ADMM_8iters_MultiScale_20epochs.pth"),
    }
    

    if opt.test == 'shear':
        snrs = [20, 40, 60, 80, 100, 150, 200]
        for method, (n_iters, model_file) in methods.items():
            test_shear(model_name=opt.model, n_iters=n_iters, model_file=model_file, n_gal=opt.n_gal, snrs=snrs,
                       data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    elif opt.test == 'time':
        for method, (n_iters, model_file) in methods.items():
            for i in range(3): # Run 2 dummy test first to warm up the GPU.
                test_time(model_name=opt.model, n_iters=n_iters, model_file=model_file, n_gal=opt.n_gal,
                          data_path='/mnt/WD6TB/tianaoli/dataset/LSST_23.5_deconv/', result_path=opt.result_path)
    else:
        raise ValueError("Invalid test type.")
