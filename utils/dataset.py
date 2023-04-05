import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import logging




def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
        
class PACT_Dataset(Dataset):
    """Simulated PACT Dataset inherited from `torch.utils.data.Dataset`."""
    def __init__(self, data_path='/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/', train=True,
                 obs_folder='obs/', gt_folder='gt/'):
        """Construction function for the PyTorch PACT Dataset.

        Args:
            data_path (str, optional): Path to the dataset. Defaults to '/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/'.
            train (bool, optional): Whether the dataset is generated for training or testing. Defaults to True.
            obs_folder (str, optional): Path to the observed image folder. Defaults to 'obs/'.
            gt_folder (str, optional): Path to the ground truth image folder. Defaults to 'gt/'.
        """
        super(PACT_Dataset, self).__init__()
        
        self.logger = logging.getLogger('Dataset')
        
        # Initialize parameters
        self.train = train
        self.data_path = os.path.join(data_path, 'train' if train else 'test')
        self.gt_path = os.path.join(self.data_path, gt_folder)
        self.obs_path = os.path.join(self.data_path, obs_folder)
        self.n_gt = len(os.listdir(self.gt_path))
        self.n_obs = len(os.listdir(self.obs_path))
        if self.n_gt == self.n_obs:
            self.n_samples = self.n_gt
        else:
            self.n_samples = min(self.n_gt, self.n_obs)
            self.logger.warning("Inequal number of ground truth samples and observation samples.")
        self.n_samples = 3200
        self.logger.info(" Successfully constructed %s dataset. Total Samples: %s.", 'train' if self.train else 'test', self.n_samples)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        
        obs = torch.from_numpy(np.load(os.path.join(self.obs_path, f"obs_{idx}.npy"))).float()
        gt = torch.from_numpy(np.load(os.path.join(self.gt_path, f"gt_{idx}.npy"))).unsqueeze(0).float()

        return obs, gt
    
    
    
def get_dataloader(data_path='/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/', train=True, train_val_split=0.875, batch_size=16,
                   obs_folder='obs/', gt_folder='gt/'):
    """Generate PyTorch dataloaders for training or testing.

    Args:
        data_path (str, optional): Path the dataset. Defaults to `'/mnt/WD6TB/tianaoli/dataset/SkinVessel_PACT/'`.
        train (bool, optional): Whether to generate train dataloader or test dataloader. Defaults to True.
        train_val_split (float, optional): Proportion of data used in train dataloader in train dataset, the rest will be used in valid dataloader. Defaults to `0.875`.
        batch_size (int, optional): Batch size for training dataset. Defaults to 16.
        obs_folder (str, optional): Path to the observed image folder. Defaults to `'obs/'`.
        gt_folder (str, optional): Path to the ground truth image folder. Defaults to `'gt/'`.

    Returns:
        train_loader (`torch.utils.data.DataLoader`):  PyTorch dataloader for train dataset.
        val_loader (`torch.utils.data.DataLoader`): PyTorch dataloader for valid dataset.
        test_loader (`torch.utils.data.DataLoader`): PyTorch dataloader for test dataset.
    """
    if train:
        train_dataset = PACT_Dataset(data_path=data_path, train=True)
        train_size = int(train_val_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        test_dataset = PACT_Dataset(data_path=data_path, train=False, obs_folder=obs_folder, gt_folder=gt_folder)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return test_loader
    
    
    
if __name__ == '__main__':
    train_loader, val_loader = get_dataloader()
    for idx, (obs, gt) in enumerate(train_loader):
        print(obs.shape, gt.shape)
        break