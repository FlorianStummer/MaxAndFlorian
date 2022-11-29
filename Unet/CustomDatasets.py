from torch.utils.data import Dataset
import os
import numpy as np
from functions import create_unet_images

class MagnetDataset(Dataset):
    def __init__(self, path_to_root, partition=(0,1), transforms=None, maximum_elements=10000):
        self.path_to_root = path_to_root
        self.transforms = transforms
        self.partition = partition
        self.maximum_elements = maximum_elements
        self.lowest_idx = int(self.partition[0]*(self.maximum_elements) - 1)
        self.highest_idx = int(self.partition[1]*(self.maximum_elements) - 1)
        self.length = min(self.maximum_elements,self.highest_idx - self.lowest_idx)
        # self.dim_list = np.genfromtxt(os.path.join(self.path_to_root,"random_magnet_list.csv"), delimiter=',', unpack=True)[1:,1:]
        self.inputs = np.load('../../MagnetDataset.npz')['input'][:self.maximum_elements]
        self.targets = np.load('../../MagnetDataset.npz')['target'][:self.maximum_elements]
        self.metainfos = np.load('../../MagnetDataset.npz')['metainfo'][:self.maximum_elements]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.lowest_idx + idx
        
        input = self.inputs[actual_idx]
        target = self.targets[actual_idx]
        
        # data augmentation
        # if(model in trainmode):
            # Mirrored in X
            # if(np.random.normal(0,1,1) <= 0.5):
            #     mirror in x
            # Negative current
            # if(np.random.normal(0,1,1) <= 0.5):
            #     invert current

        return input, target
