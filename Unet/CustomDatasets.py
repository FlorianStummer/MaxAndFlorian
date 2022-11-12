from torch.utils.data import Dataset
import os
import numpy as np
from functions import create_unet_images

class MagnetDataset(Dataset):
    def __init__(self, path_to_root, partition=(0,1), transforms=None, maximum_elements=10000):
        self.path_to_root = path_to_root
        self.transforms = transforms
        self.partition = partition
        self.lowest_idx = int(self.partition[0]*(len(os.listdir(self.path_to_root)) - 1))
        self.highest_idx = int(self.partition[1]*(len(os.listdir(self.path_to_root)) - 1)) - 1
        self.length = min(maximum_elements,self.highest_idx - self.lowest_idx)
        self.dim_list = np.genfromtxt(os.path.join(self.path_to_root,"random_magnet_list.csv"), delimiter=',', unpack=True)[1:,1:]
        self.inputs = np.load('../../MagnetDataset.npz')['input'][:self.length]
        self.targets = np.load('../../MagnetDataset.npz')['target'][:self.length]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.lowest_idx + idx
        
        input = self.inputs[idx]
        target = self.inputs[idx]

        return input, target
