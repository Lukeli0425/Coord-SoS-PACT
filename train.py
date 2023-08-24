import argparse
import logging
import os

import torch
from torch.optim import Adam

from models.Double_ADMM import Double_ADMM
from models.DUBLID import DUBLID
from models.ResUNet import FT_ResUNet, ResUNet
from models.Unrolled_ADMM import Unrolled_ADMM
from models.WienerNet import WienerNet
from utils.dataset import get_dataloader
from utils.utils_plot import plot_loss
from utils.utils_train import SSIM, MultiScaleLoss, get_model_name

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train(model_name='DUBLID', n_iters=4, nc=32,
          n_epochs=100, lr=2e-4, loss='MSE',
          data_path='/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/', train_val_split=0.9, batch_size=32,
          model_save_path='./saved_models/', pretrained_epochs=0):
    
    model_name = get_model_name(method=model_name, n_iters=n_iters, nc=nc, loss=loss)
    logger = logging.getLogger('Train')
    logger.info(' Start training %s on %s data for %s epochs.', model_name, data_path, n_epochs)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    train_loader, val_loader = get_dataloader(data_path=data_path, train=True, train_val_split=train_val_split, batch_size=batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if 'Double_ADMM' in model_name:
        model = Double_ADMM(n_iters=n_iters, n_delays=8)
    elif 'Unrolled_ADMM' in model_name:
        model = Unrolled_ADMM (n_iters=n_iters, n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    elif 'WienerNet' in model_name:
        model = WienerNet(n_delays=8, nc=[nc, nc*2, nc*4, nc*8])
    # elif 'FT_ResUNet' in model_name:
    #     model = FT_ResUNet(in_nc=8, out_nc=1)
    elif 'ResUNet' in model_name:
        model = ResUNet(in_nc=8, out_nc=1, nc=[nc, nc*2, nc*4, nc*8])
    model.to(device)
    # model = DataParallel(model, device_ids=[0, 1])
    
    if pretrained_epochs > 0:
        try:
            pretrained_file = os.path.join(model_save_path, f'{model_name}_{pretrained_epochs}epochs.pth')
            model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
            logger.info(' Successfully loaded in %s.', pretrained_file)
        except:
            raise Exception(' Failed loading in %s!', pretrained_file)

    if loss == 'SSIM':
        loss_fn = SSIM()
    elif loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss == 'MultiScale':
        loss_fn = MultiScaleLoss()
    
    optimizer = Adam(params=model.parameters(), lr = lr)

    train_loss_list, val_loss_list = [], []
    val_loss_min, epoch_min = 1.e9, 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, (obs, gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, gt = obs.to(device), gt.to(device)
            rec = model(obs)
            loss = loss_fn(gt, rec)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            
            # Evaluate on valid dataset.
            if (idx+1) % 20 == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for _, (obs, gt) in enumerate(val_loader):
                        obs, gt = obs.to(device), gt.to(device)
                        rec = model(obs)
                        loss = loss_fn(gt, rec)
                        val_loss += loss.item()

                logger.info(" [{}: {}/{}]  train_loss={:.4g}  val_loss={:.4g}".format(
                                epoch+1, idx+1, len(train_loader),
                                train_loss,
                                val_loss/len(val_loader)))
    
        # Evaluate on train & valid dataset after every epoch.
        train_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, (obs, gt) in enumerate(train_loader):
                obs, gt = obs.to(device), gt.to(device)
                rec = model(obs)
                loss = loss_fn(gt, rec)
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(train_loader))
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, (obs, gt) in enumerate(val_loader):
                obs, gt = obs.to(device), gt.to(device)
                rec = model(obs)
                loss = loss_fn(gt, rec)
                val_loss += loss.item()
            val_loss_list.append(val_loss/len(val_loader))

        logger.info(" [{}: {}/{}]  train_loss={:.4g}  val_loss={:.4g}".format(
                        epoch+1, len(train_loader), len(train_loader),
                        train_loss/len(train_loader),
                        val_loss/len(val_loader)))
        
        # Save model.
        if val_loss_min > val_loss:
            val_loss_min = val_loss
            epoch_min = epoch
            model_file_name = f'{model_name}_{epoch+1+pretrained_epochs}epochs.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, model_file_name))
            logger.info(' Model saved to %s', os.path.join(model_save_path, model_file_name))

        # Plot loss curve.
        plot_loss(train_loss_list, val_loss_list, epoch_min, model_save_path, model_name)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument('--model', type=str, default='Unrolled_ADMM', choices=['Unrolled_ADMM', 'Double_ADMM', 'WienerNet', 'FT_ResUNet', 'ResUNet'])
    parser.add_argument('--n_iters', type=int, default=4)
    parser.add_argument('--nc', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'MultiScale', 'SSIM'])
    parser.add_argument('--train_val_split', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_epochs', type=int, default=0)
    opt = parser.parse_args()

    data_path = '/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT_new/'

    train(model_name=opt.model, n_iters=opt.n_iters, nc=opt.nc,
          n_epochs=opt.n_epochs, lr=opt.lr, loss=opt.loss,
          data_path=data_path, train_val_split=opt.train_val_split, batch_size=opt.batch_size,
          model_save_path='./saved_models_new/', pretrained_epochs=opt.pretrained_epochs)
