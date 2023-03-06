from torch.utils.data import Dataset as torch_dataset
import os
import numpy as np

class Dataset(torch_dataset):
    def __init__(self, path_to_root, partition=(0,1), transforms=None, maximum_elements=10000):
        self.path_to_root = path_to_root
        self.transforms = transforms
        self.partition = partition
        self.maximum_elements = maximum_elements
        self.lowest_idx = int(self.partition[0]*(self.maximum_elements))
        self.highest_idx = int(self.partition[1]*(self.maximum_elements) - 1)
        self.length = min(self.maximum_elements,self.highest_idx - self.lowest_idx)
        self.train = self.partition[0] == 0 # this assumes the dataset partition starts with the train segment

        self.meta = np.load(os.path.join(self.path_to_root,"meta.npz"))["metainfos"][self.lowest_idx:self.highest_idx,:]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.lowest_idx + idx

        sample = np.load(os.path.join(self.path_to_root,str(actual_idx)+".npz"))
        
        #TODO data augmentation

        inp = sample["input"]
        tar = sample["target"]

        return inp, tar