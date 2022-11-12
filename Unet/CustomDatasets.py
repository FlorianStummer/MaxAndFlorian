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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.lowest_idx + idx
        filename = "Stage1_" + str(actual_idx).zfill(6) + ".csv"
        Bx,By = np.genfromtxt(os.path.join(self.path_to_root,filename), delimiter=',', unpack=True)[2:4]
        Bx = np.resize(Bx[1:], (81,121)).T
        By = np.resize(By[1:], (81,121)).T
        Bx[np.abs(Bx) < 0.01] = 0
        By[np.abs(By) < 0.01] = 0
        target = np.stack([Bx, By], axis=0)

        inputs = create_unet_images(self.dim_list[:,actual_idx])

        inputs = np.asarray(inputs)[:,:120,:80]
        target = np.asarray(target)[:,:120,:80]

        return inputs, target
