import argparse
import json
import logging
import os
import time

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from models.ResUNet import ResUNet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.deconv import WienerNet
from utils.dataset import get_dataloader

# from utils.utils_train import get_method


def test_metric(method, n_iters, nc, model_file, n_samples, metric, data_path, result_path):
    logger = logging.getLogger('Performance Test')
    logger.info(' Metric: %s', metric)
    logger.info(' Testing method: %s', method)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')
    
    # Load the model.
    model = None
    if 'Unrolled_ADMM' in method:
        model = Unrolled_ADMM (n_iters=n_iters, n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'WienerNet' in method:
        model = WienerNet(n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'ResUNet' in method:
        model = ResUNet(in_nc=8, out_nc=1, nc=[nc, nc*2, nc*4, nc*8])

    if model is not None:
        model.to(device)
        try: # Load the pretrained wieghts.
            model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
            logger.info(' Successfully loaded in %s.', model_file)
        except:
            raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    evaluate = ssim if metric == 'SSIM' else (psnr if metric == 'PSNR' else None)
    
    logger.info(' Running %s test with %s samples.\n', metric, n_samples)
    test_loader = get_dataloader(data_path=data_path, train=False, obs_folder=f'obs/', gt_folder=f'gt/')
    
    rec_metric = []
    with torch.no_grad():
        for (obs, gt), idx in zip(test_loader, tqdm(range(n_samples))):
            gt = gt.squeeze(0).squeeze(0).detach().cpu().numpy()
            if method == 'No_Deconv':
                obs = obs.squeeze(0).squeeze(0).detach().cpu().numpy()
                rec_metric.append(evaluate(obs[3], gt, data_range=2.))
            else: # Unrolled ADMM, WienerNet, ResUNet
                obs = obs.to(device)
                rec = model(obs)
                rec = rec.squeeze(0).squeeze(0).detach().cpu().numpy()
                # print(rec.shape, gt.shape)
                rec_metric.append(evaluate(rec, gt, data_range=2))
    
    # Save results.
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(" Successfully loaded in %s.", results_file)
    except:
        results = {} 
        logger.critical(" Failed loading in %s.", results_file)
        
    if metric not in results:
        results[metric] = {}
    results[metric]['rec'] = rec_metric

    
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logger.info(" Shear test results saved to %s.\n", results_file)
    


def test_time(method, n_iters, nc, model_file, n_samples, data_path, result_path):  
    """Test the time consumption of different methods."""
    logger = logging.getLogger('Time Test')
    logger.info(' Running time test with %s samples.', n_samples)
    logger.info(' Testing method: %s', method)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = get_dataloader(data_path=data_path, train=False)
    
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')

    # Load the model.
    model = None
    if 'Unrolled_ADMM' in method:
        model = Unrolled_ADMM (n_iters=n_iters, n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'WienerNet' in method:
        model = WienerNet(n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'ResUNet' in method:
        model = ResUNet(in_nc=8, out_nc=1, nc=[nc, nc*2, nc*4, nc*8])

    if model is not None:
        model.to(device)
        try: # Load the pretrained wieghts.
            model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
            logger.info(' Successfully loaded in %s.', model_file)
        except:
            raise Exception('Failed loading in %s', model_file)
        model.eval()

    time_start = time.time()
    with torch.no_grad():
        for (obs, gt), idx in zip(test_loader, tqdm(range(n_samples))):
            if method == 'No_Deconv':
                rec = obs.squeeze(0).squeeze(0).detach().cpu().numpy()
            else: # Unrolled ADMM, WienerNet, ResUNet
                obs = obs.to(device)
                rec = model(obs)
                rec = rec.squeeze(0).squeeze(0).detach().cpu().numpy()
                
    time_end = time.time()
    logger.info(' Tested %s on %s samples: Time = {:.4g}s.'.format(time_end-time_start),method, n_samples)

    # Save test results.
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(" Successfully loaded in %s.", results_file)
    except:
        results = {} 
        logger.critical(" Failed loading in %s.", results_file)
    results['time'] = (time_end-time_start, n_samples)
    
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logger.info(" Time test results saved to %s.\n", results_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='time', choices=['PSNR', 'SSIM', 'time'])
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--result_path', type=str, default='results/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    # Uncomment the methods to be tested.
    methods = {
        'No_Deconv': (0, 0, None), 
        'ResUNet_8channels': (0, 8, './pretrained_models/ResUNet_8channels_MSE_92epochs.pth'), 
        'ResUNet_16channels': (0, 16, './pretrained_models/ResUNet_16channels_MSE_80epochs.pth'), 
        'ResUNet_32channels': (0, 32, './pretrained_models/ResUNet_32channels_MSE_66epochs.pth'),
        'WienerNet_8channels': (0, 8, './pretrained_models/WienerNet_8channels_MSE_97epochs.pth'), 
        'WienerNet_16channels': (0, 16, './pretrained_models/WienerNet_16channels_MSE_65epochs.pth'),
        'WienerNet_32channels': (0, 32, './pretrained_models/WienerNet_32channels_MSE_55epochs.pth'), 
        'Unrolled_ADMM_4iters_8channels': (4, 8, './pretrained_models/Unrolled_ADMM_4iters_8channels_MSE_92epochs.pth'), 
        'Unrolled_ADMM_4iters_16channels': (4, 16, './pretrained_models/Unrolled_ADMM_4iters_16channels_MSE_71epochs.pth'), 
        'Unrolled_ADMM_4iters_32channels': (4, 32, './pretrained_models/Unrolled_ADMM_4iters_32channels_MSE_71epochs.pth'), 
        # 'Unrolled_ADMM_8iters_8channels': (8, 8, './pretrained_models/Unrolled_ADMM_8iters_8channels_MSE_99epochs.pth'),
        # 'Unrolled_ADMM_8iters_16channels': (8, 16, './pretrained_models/Unrolled_ADMM_8iters_16channels_MSE_96epochs.pth'),
        # 'Unrolled_ADMM_8iters_32channels': (8, 32, './pretrained_models/Unrolled_ADMM_8iters_32channels_MSE_44epochs.pth'),
    }
    
    # data_path = '/Users/luke/Downloads/SkinVessel_PACT/' # '/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/'
    data_path = '/home/mist/SkinVessel_PACT/'
    
    if opt.test in ['PSNR', 'SSIM']:
        for method, (n_iters, nc, model_file) in methods.items():
            test_metric(method=method, n_iters=n_iters, nc=nc, model_file=model_file, n_samples=opt.n_samples, metric=opt.test,
                        data_path=data_path, result_path=opt.result_path)
    elif opt.test == 'time':
        for method, (n_iters, nc, model_file) in methods.items():
            for i in range(4): # Run several dummy tests to warm up the GPU.
                test_time(method=method, n_iters=n_iters, nc=nc, model_file=model_file, n_samples=opt.n_samples, 
                          data_path=data_path, result_path=opt.result_path)
    else:
        raise ValueError("Invalid test type.")
