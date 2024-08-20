import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class JR_Dataset(Dataset):
    def __init__(self, IP_stack, l_patch, N_patch, stride):
        self.IP_stack = IP_stack
        self.l_patch = l_patch
        self.N_patch = N_patch
        self.stride = stride
        self.nx = (self.IP_stack.shape[-2]-self.N_patch) // self.stride + 1
        self.ny = (self.IP_stack.shape[-1]-self.N_patch) // self.stride + 1

    def __len__(self):
        return self.nx * self.ny
    
    def __getitem__(self, idx):
        i, j = idx // self.nx, idx % self.ny
        x, y = (j-self.ny//2)*self.l_patch / 4, (self.nx//2-i)*self.l_patch / 4
        return i, j, torch.tensor(x), torch.tensor(y), self.IP_stack[:,self.stride*i:self.stride*i+self.N_patch, self.stride*j:self.stride*j+self.N_patch]
    
    
def get_jr_dataloader(IP_stack, l_patch=3.2e-3, N_patch=80, stride=20, batch_size=1, shuffle=True):
    dataset = JR_Dataset(IP_stack.clone(), l_patch, N_patch, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# class PACT_Dataset(Dataset):
#     """Simulated PACT Dataset inherited from `torch.utils.data.Dataset`."""
#     def __init__(self, data_path, train=True, n_train=130*169, obs_folder='obs/', gt_folder='gold/', psf_folder='psf/'):
#         """Construction function for the PyTorch PACT Dataset.

#         Args:
#             data_path (str, optional): Path to the dataset. Defaults to `'/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/'`.
#             train (bool, optional): Whether the dataset is generated for training or testing. Defaults to `True`.
#             obs_folder (str, optional): Path to the observed image folder. Defaults to `'obs/'`.
#             gt_folder (str, optional): Path to the ground truth image folder. Defaults to `'gt/'`.
#             psf_folder (str, optional): Path to the PSF folder. Defaults to `'psf/'`.
#         """
#         super(PACT_Dataset, self).__init__()
        
#         self.logger = logging.getLogger('Dataset')
        
#         # Initialize parameters
#         self.train = train
#         self.data_path = data_path
#         self.gt_path = os.path.join(self.data_path, gt_folder)
#         self.obs_path = os.path.join(self.data_path, obs_folder)
#         self.psf_path = os.path.join(data_path, psf_folder)
        
#         self.n_gt = len(os.listdir(self.gt_path))
#         self.n_obs = len(os.listdir(self.obs_path))
#         # self.n_psf = 169 
#         if self.n_gt == self.n_obs:
#             self.n_samples = self.n_gt
#         else:
#             self.n_samples = min(self.n_gt, self.n_obs)
#             self.logger.warning("Inequal number of ground truth samples and observation samples.")
#         self.n_train = n_train
#         self.n_test = self.n_samples - self.n_train
        
#         self.logger.info(" Successfully constructed %s dataset. Total Samples: %s.",
#                          'train' if self.train else 'test', self.n_train if self.train else self.n_test)

#     def __len__(self):
#         return self.n_train if self.train else self.n_test

#     def __getitem__(self, idx):
#         idx = idx if self.train else idx + self.n_train ## TODD ##
#         gt = torch.from_numpy(np.load(os.path.join(self.gt_path, f"gt_{idx}.npy"))).unsqueeze(0).float()
#         # gt = gt / gt.sum() # Normalize flux to 1.
        
#         obs = torch.from_numpy(np.load(os.path.join(self.obs_path, f"obs_{idx}.npy"))).float()
#         # obs = obs / obs.sum(dim=[-2,-1]).unsqueeze(-1).unsqueeze(-1) # Normalize flux to 1.
#         # gt = (gt - gt.min()) / (gt.max() - gt.min()) # Normalize to [0, 1].
#         # obs = (obs - gt.min()) / (gt.max() - gt.min()) # Normalize to [0, 1].
        
#         # psf_idx = idx # Pick corresponding PSF for each patch.
#         psf = torch.load(os.path.join(self.psf_path, f"psf_{idx}.pth"))

#         gt /= gt.abs().mean()
#         obs /= obs.abs().mean()

#         return obs, psf, gt
    
    
    
# def get_dataloader(data_path='/mnt/WD6TB/tianaoli/dataset/Mice_new1/', train=True, train_val_split=0.875, batch_size=64, num_workers=18, pin_memory=True,
#                    obs_folder='obs/', gt_folder='gold/', psf_folder='psf/'):
#     """Generate PyTorch dataloaders for training or testing.

#     Args:
#         data_path (str, optional): Path the dataset. Defaults to `'/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/'`.
#         train (bool, optional): Whether to generate train dataloader or test dataloader. Defaults to True.
#         train_val_split (float, optional): Proportion of data used in train dataloader in train dataset, the rest will be used in valid dataloader. Defaults to `0.875`.
#         batch_size (int, optional): Batch size for training dataset. Defaults to 16.
#         obs_folder (str, optional): Path to the observed image folder. Defaults to `'obs/'`.
#         gt_folder (str, optional): Path to the ground truth image folder. Defaults to `'gt/'`.

#     Returns:
#         train_loader (`torch.utils.data.DataLoader`): PyTorch dataloader for train dataset.
#         val_loader (`torch.utils.data.DataLoader`): PyTorch dataloader for valid dataset.
#         test_loader (`torch.utils.data.DataLoader`): PyTorch dataloader for test dataset.
#     """
#     if train:
#         train_dataset = PACT_Dataset(data_path=data_path, train=True, obs_folder=obs_folder, gt_folder=gt_folder, psf_folder=psf_folder)
#         train_size = int(train_val_split * len(train_dataset))
#         val_size = len(train_dataset) - train_size
#         train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
#         return train_loader, val_loader
#     else:
#         test_dataset = PACT_Dataset(data_path=data_path, train=False, obs_folder=obs_folder, gt_folder=gt_folder, psf_folder=psf_folder)
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#         return test_loader
    
    
    
if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(batch_size=1)
    test_loader = get_dataloader(train=False)
    pos, neg = 0, 0
    for idx, (obs, psf, gt) in enumerate(train_loader):
        pass
            
    # print(pos, neg)
        