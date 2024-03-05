from torch.utils.data import Dataset as torch_dataset
import os
import numpy as np
import magnetdesigner
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Dataset_Dipole_H(torch_dataset):
    def __init__(self, path_to_root, transforms=None, maximum_elements=128, mdfile="md_hshaped_v1"):
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
        allmagnets = magnetdesigner.designer.magnetdataset()
        allmagnets = allmagnets.load(os.path.join(self.path_to_root, mdfile))
        self.magnetinfos = magnetdesigner.designer.magnetdataset()
        for element in self.npzlist:
            self.magnetinfos.append_magnet(allmagnets.get_magnet_by_name(element[:-4]))
        self.maximum_elements = min(maximum_elements, len(self.npzlist))
        self.length = self.maximum_elements 

        # get the spacing from the first npz file
        dummy = np.load(os.path.join(self.path_to_root, self.npzlist[0]))['data']
        xlist = list(set(dummy[:,:,0].flatten()))
        xlist.sort()
        ylist = list(set(dummy[:,:,1].flatten()))
        ylist.sort()
        self.spacing = np.round(np.abs(xlist[0] - xlist[1]), 5) * 1e-3

        # get the name of the largest magnet covered by the npzs from magnetinfos
        _, self.maxX, _, self.maxY = self.magnetinfos.get_largest_magnet_dimensions()
        self.xbins = int(np.ceil(self.maxX / self.spacing))
        self.ybins = int(np.ceil(self.maxY / self.spacing))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = idx
        # load the npz file
        filename = os.path.join(self.path_to_root, 'prepared', self.npzlist[actual_idx])
        inp, tar = np.load(filename)['input'], np.load(filename)['target']

        # TODO data augmentation
        
        return inp, tar

    def prepare(self, actual_idx):
        # prepare the input
        dipole = self.magnetinfos.get_magnet_by_name(self.npzlist[actual_idx][:-4])
        magnet = dipole.magnet2D
        inp = np.zeros((self.xbins, self.ybins, 5))
        # create input image 1: all the bins are 1 inside the yoke and 0 outside
        for yoke_i in magnet.yokeCoords:
            yoke = Polygon(yoke_i)
            for i in range(self.xbins):
                for j in range(self.ybins):
                    if yoke.contains(Point(i*self.spacing, j*self.spacing)):
                        inp[i, j, 0] = 1
        # create input image 2: all the bins in the coils are the current density
        for index, coil_i in enumerate(magnet.coilCoords):
            coil_i_array = np.array(coil_i)
            coil_i_array_x = coil_i_array[:,0]
            coil_i_array_y = coil_i_array[:,1]
            coil_i_array_x = np.arange(np.min(coil_i_array_x), np.max(coil_i_array_x), self.spacing * 0.2)
            coil_i_array_y = np.arange(np.min(coil_i_array_y), np.max(coil_i_array_y), self.spacing * 0.2)
            coil = Polygon(coil_i)
            for i in range(self.xbins):
                for j in range(self.ybins):
                    if coil.contains(Point(i*self.spacing, j*self.spacing)):
                        inp[i, j, 1] = magnet.currentDensities[index]
        # create input image 3 and 4: the field strength in x and y direction including the sign
                    dx = np.min(np.abs(i*self.spacing-coil_i_array_x))
                    dy = np.min(np.abs(j*self.spacing-coil_i_array_y))
                    # get the distance and angle to the coil
                    r = np.sqrt(dx**2 + dy**2)
                    if r == 0:
                        r = 1e-2
                    if dx == 0:
                        alpha = np.pi/2
                    else:
                        alpha = np.arctan(dy/dx)
                    # get the sign of the field strength
                    if j*self.spacing < min(coil_i_array[:,1]):
                        sign_x = 1
                    elif j*self.spacing > max(coil_i_array[:,1]):
                        sign_x = -1
                    else:
                        sign_x = 0
                    if i*self.spacing < min(coil_i_array[:,0]):
                        sign_y = -1
                    elif i*self.spacing > max(coil_i_array[:,0]):
                        sign_y = 1
                    else:
                        sign_y = 0
                    inp[i, j, 2] += sign_x * np.sin(alpha) * magnet.currentDensities[index] / r
                    inp[i, j, 3] += sign_y * np.cos(alpha) * magnet.currentDensities[index] / r
        # create input image 5: good field region form magnet.aper_x and magnet.aper_y
        for i in range(self.xbins):
            for j in range(self.ybins):
                if i*self.spacing < dipole.aper_x * 0.5 * 2.0 and j*self.spacing < dipole.aper_y * 0.5:
                    inp[i, j, 4] = 1

        # prepare the target
        tar_unprepared = np.load(os.path.join(self.path_to_root, self.npzlist[actual_idx]))['data']
        tar = np.zeros((self.xbins, self.ybins, 2))
        for i in range(tar_unprepared.shape[0]):
            for j in range(tar_unprepared.shape[1]):
                x = int(np.round(tar_unprepared[i, j, 0] / (self.spacing*1e3)))
                y = int(np.round(tar_unprepared[i, j, 1] / (self.spacing*1e3)))
                tar[x, y, 0] = tar_unprepared[i, j, 2]
                tar[x, y, 1] = tar_unprepared[i, j, 3]
        # set fields to zero if field is smaller than 1e-4
        tar[np.abs(tar) < 1e-4] = 0

        # move channels to the front
        inp = np.moveaxis(inp, -1, 0)
        tar = np.moveaxis(tar, -1, 0)

        # save inp and tar in npz file
        np.savez(os.path.join(self.path_to_root, 'prepared', self.npzlist[actual_idx]), input=inp, target=tar)

    def prepare_all(self, overwrite=False):
        for i in range(self.maximum_elements):
            if not os.path.exists(os.path.join(self.path_to_root, 'prepared', self.npzlist[i])) or overwrite:
                self.prepare(i)

def plot_to_axis(axs, img, title):
    axs.imshow(img.T, origin='lower')
    axs.set_title(title)
    axs.set_aspect('equal')
    axs.axis('off')


def plot_ds(ds, idx):
    import matplotlib.pyplot as plt
    inp, tar = ds[idx]
    fig, axs = plt.subplots(2, 4)
    plot_to_axis(axs[0, 0], inp[0,:,:], "Yoke")
    plot_to_axis(axs[1, 0], inp[1,:,:], "Current Density")
    plot_to_axis(axs[0, 1], inp[4,:,:], "Good Field Region")
    axs[1, 1].axis('off')
    plot_to_axis(axs[0, 2], inp[2,:,:], "Field Strength X")
    plot_to_axis(axs[1, 2], inp[3,:,:], "Field Strength Y")
    plot_to_axis(axs[0, 3], tar[0,:,:], "Field Strength X Target")
    plot_to_axis(axs[1, 3], tar[1,:,:], "Field Strength Y Target")

    plt.show()

def main():
    ds = Dataset_Dipole_H("data/raw/npz_select_1cmSpacing")
    plot_ds(ds, 0)

if __name__ == "__main__":
    main()