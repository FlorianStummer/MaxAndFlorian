from torch.utils.data import Dataset as torch_dataset
import os
import numpy as np
import matplotlib.pyplot as plt


class Dataset_Bend_H(torch_dataset):
    def __init__(self, path_to_root, transforms=None, maximum_elements=128000):
        self.path_to_root = path_to_root
        self.transforms = transforms
        self.npzlist = []
        n = 0
        for file in os.listdir(self.path_to_root):
            if file.endswith(".npz"):
                self.npzlist.append(file)
                n += 1
            if n == maximum_elements:
                break
        self.npzlist.sort()
        # shuffle the list of npz files with a fixed seed
        np.random.seed(42)
        np.random.shuffle(self.npzlist)
        self.maximum_elements = min(maximum_elements, len(self.npzlist))
        self.length = self.maximum_elements

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = idx
        # load the npz file
        filename = os.path.join(self.path_to_root, self.npzlist[actual_idx])
        inp, tar = np.load(filename)['input'], np.load(filename)['target']

        # TODO data augmentation
        
        return inp, tar
        # return {"inp": inp, "metadata": metadata}, tar


def plot_to_axis(axs, img, title):
    axs.imshow(img.T, origin='lower')
    axs.set_title(title)
    axs.set_aspect('equal')
    axs.axis('off')

def plot_ds(ds, idx):
    inp, tar = ds[idx]
    fig, axs = plt.subplots(4, 6)
    plot_to_axis(axs[0, 0], inp[0,:,:], "Yoke bool")
    plot_to_axis(axs[0, 1], inp[1,:,:], "Yoke distance")
    plot_to_axis(axs[0, 2], inp[2,:,:], "Yoke distance norm")
    plot_to_axis(axs[0, 3], inp[3,:,:], "Coil bool")
    plot_to_axis(axs[0, 4], inp[4,:,:], "Coil distance")
    plot_to_axis(axs[0, 5], inp[5,:,:], "Coil distance norm")
    plot_to_axis(axs[1, 0], inp[6,:,:], "Coil distance in yoke")
    plot_to_axis(axs[1, 1], inp[7,:,:], "Coil distance in yoke norm")
    plot_to_axis(axs[1, 2], inp[8,:,:], "Coil distance in yoke x")
    plot_to_axis(axs[1, 3], inp[9,:,:], "Field strength x")
    plot_to_axis(axs[1, 4], inp[10,:,:], "Field strength y")
    plot_to_axis(axs[1, 5], inp[11,:,:], "Coil distance in yoke x norm")
    plot_to_axis(axs[2, 0], inp[12,:,:], "Coil distance in yoke y norm")
    plot_to_axis(axs[2, 1], inp[13,:,:], "Current density")
    plot_to_axis(axs[2, 2], inp[14,:,:], "Distance to edge")
    plot_to_axis(axs[2, 3], inp[15,:,:], "Distance to edge norm")
    plot_to_axis(axs[2, 4], inp[16,:,:], "DNN prediction")
    plot_to_axis(axs[2, 5], inp[17,:,:], "Priority map")
    plot_to_axis(axs[3, 0], tar[0,:,:], "Bx")
    plot_to_axis(axs[3, 1], tar[1,:,:], "By")
    axs[3, 2].axis('off')
    axs[3, 3].axis('off')
    axs[3, 4].axis('off')
    axs[3, 5].axis('off')
    plt.show()
