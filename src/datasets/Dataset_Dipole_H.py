from torch.utils.data import Dataset as torch_dataset
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.measurement import distance
from shapely.affinity import scale
import matplotlib.pyplot as plt
import pandas as pd
import magnetdesigner
import magnetoptimiser


class Dataset_Dipole_H(torch_dataset):
    def __init__(self, path_to_root, transforms=None, maximum_elements=128000, mdfile="md_hshaped_v1", prepareFolder="prepared"):
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
        allmagnets = magnetdesigner.designer.magnetdataset()
        allmagnets = allmagnets.load(os.path.join(self.path_to_root, mdfile))
        self.magnetinfos = magnetdesigner.designer.magnetdataset()
        # shuffle the list of npz files with a fixed seed
        np.random.seed(42)
        np.random.shuffle(self.npzlist)
        for element in self.npzlist:
            self.magnetinfos.append_magnet(allmagnets.get_magnet_by_name(element[:-4]))
        self.maximum_elements = min(maximum_elements, len(self.npzlist))
        self.length = self.maximum_elements 
        self.prepareFolder = prepareFolder

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
        filename = os.path.join(self.path_to_root, self.prepareFolder, self.npzlist[actual_idx])
        inp, tar = np.load(filename)['input'], np.load(filename)['target']

        # TODO data augmentation
        
        return inp, tar
        # return {"inp": inp, "metadata": metadata}, tar

    def prepare(self, actual_idx):
        xbinsFixed = 224
        ybinsFixed = 160
        # prepare the input
        dipole = self.magnetinfos.get_magnet_by_name(self.npzlist[actual_idx][:-4])
        magnet = dipole.magnet2D
        # inp = np.zeros((self.xbins, self.ybins, 5))
        inp = np.zeros((xbinsFixed, ybinsFixed, 18))
        # channel 0, 1 and 2: the yoke region and the distance to the yoke boundary (absolute and normalised)
        for yoke_i in magnet.yokeCoords:
            coords = np.array(yoke_i)
            coords[np.abs(coords) < 1e-3] = 0
            yoke = Polygon(coords)
            inverse_polygon = Polygon([(-xbinsFixed*self.spacing, -ybinsFixed*self.spacing), (-xbinsFixed*self.spacing, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, -ybinsFixed*self.spacing)])
            inverse_polygon = inverse_polygon.difference(yoke)
            inverse_polygon = inverse_polygon.difference(scale(yoke, xfact=-1, yfact=1, origin=(0, 0)))
            inverse_polygon = inverse_polygon.difference(scale(yoke, xfact=1, yfact=-1, origin=(0, 0)))
            inverse_polygon = inverse_polygon.difference(scale(yoke, xfact=-1, yfact=-1, origin=(0, 0)))
            for i in range(self.xbins):
                for j in range(self.ybins):
                    if yoke.contains(Point(i*self.spacing, j*self.spacing)):
                        inp[i, j, 0] = 1
                        inp[i, j, 1] = distance(inverse_polygon, Point(i*self.spacing, j*self.spacing))
            inp[:,:,2] = inp[:,:,1] / np.max(inp[:,:,1])
        # channel 3, 4 and 5: the coil region and the distance to the coil boundary (absolute and normalised)
        for index, coil_i in enumerate(magnet.coilCoords):
            coil_i_array = np.array(coil_i)
            coil_i_array_x = coil_i_array[:,0]
            coil_i_array_y = coil_i_array[:,1]
            coil_i_array_x = np.arange(np.min(coil_i_array_x), np.max(coil_i_array_x), self.spacing * 0.2)
            coil_i_array_y = np.arange(np.min(coil_i_array_y), np.max(coil_i_array_y), self.spacing * 0.2)
            coil = Polygon(coil_i)
            inverse_polygon = Polygon([(0, 0), (0, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, 0)])
            inverse_polygon = inverse_polygon.difference(coil)
            for i in range(xbinsFixed):
                for j in range(ybinsFixed):
                    if coil.contains(Point(i*self.spacing, j*self.spacing)):
                        inp[i, j, 3] = 1
                        inp[i, j, 4] = distance(inverse_polygon, Point(i*self.spacing, j*self.spacing))
                    inp[i, j, 6] = distance(coil, Point(i*self.spacing, j*self.spacing))
                        # print(Point(i*self.spacing, j*self.spacing))
                        # print(inverse_polygon)
                        # inp[i, j, 6] = magnet.currentDensities[index]
        # channel 9 and 10: the field strength in x and y direction including the sign
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
                    inp[i, j, 9] += sign_x * np.sin(alpha) * magnet.currentDensities[index] / r
                    inp[i, j, 10] += sign_y * np.cos(alpha) * magnet.currentDensities[index] / r
        inp[:,:,5] = inp[:,:, 4] / np.max(inp[:,:,4])
        inp[:,:,7] = inp[:,:, 6] / np.max(inp[:,:,6])
        inp[:,:,7] = (np.ones_like(inp[:,:,7]) - inp[:,:,7]) * (inp[:,:,3] != 1)
        # channel 8: the distance to the coil in the yoke region
        inp[:,:,8] = inp[:,:, 7] * inp[:,:, 0]
        min_nonzero = np.min(inp[:,:,8][inp[:,:,8] > 0])
        max_nonzero = np.max(inp[:,:,8])
        inp[:,:,8] = (inp[:,:,8] - min_nonzero) / (max_nonzero - min_nonzero) * (inp[:,:,8] > 0)
        # channel 11: normalised distance to coil in yoke in x direction
        inp[:,:,11] = inp[:,:, 8] * (inp[:,:, 9] != 0)
        inp[:,:,11] = inp[:,:, 11] / np.max(inp[:,:,11])
        # channel 12: normalised distance to coil in yoke in y direction
        inp[:,:,12] = inp[:,:, 8] * (inp[:,:, 10] != 0)
        inp[:,:,12] = inp[:,:, 12] / np.max(inp[:,:,12])
        # channel 13: current density
        inp[:,:,13] = magnet.currentDensities[0]
        # channel 14 and 15: distance to the edge of the magnet (absolute and normalised)
        polygon = Polygon([(0, 0), (0, dipole.yoke_y * 0.5), (dipole.yoke_x * 0.5, dipole.yoke_y * 0.5), (dipole.yoke_x * 0.5, 0)])
        inverse_polygon = Polygon([(-xbinsFixed*self.spacing, -ybinsFixed*self.spacing), (-xbinsFixed*self.spacing, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, ybinsFixed*self.spacing), (xbinsFixed*self.spacing, -ybinsFixed*self.spacing)])
        inverse_polygon = inverse_polygon.difference(polygon)
        for i in range(self.xbins):
            for j in range(self.ybins):
                inp[i, j, 14] = distance(inverse_polygon, Point(i*self.spacing, j*self.spacing))
        inp[:,:,15] = inp[:,:,14] / np.max(inp[:,:,14])

        # channel 16: DNN prediction in the aperture
        B0, gfr_x, gfr_y = get_B0_and_GFR(dipole)
        for i in range(self.xbins):
            for j in range(self.ybins):
                if i*self.spacing <= gfr_x * 0.5 and j*self.spacing <= gfr_y * 0.5:
                    inp[i, j, 16] = B0
        


        # create input image 17: priority map (Magnet --> 1, rest / outside --> 0)
        for i in range(xbinsFixed):
            for j in range(ybinsFixed):
                inp[i, j, 17] = 0
                if i*self.spacing < dipole.yoke_x * 0.5 and j*self.spacing < dipole.yoke_y * 0.5:
                    inp[i, j, 17] = 0.1 / (dipole.yoke_x * dipole.yoke_y * 0.25) 
                # if i*self.spacing < dipole.aper_x * 0.5 * 2.0 and j*self.spacing < dipole.aper_y * 0.5:
                #     inp[i, j, 17] = 100 / (dipole.aper_x * dipole.aper_y * 0.25)

        # prepare the target
        tar_unprepared = np.load(os.path.join(self.path_to_root, self.npzlist[actual_idx]))['data']
        tar = np.zeros((xbinsFixed, ybinsFixed, 2))
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
        if not os.path.exists(os.path.join(self.path_to_root, self.prepareFolder)):
            os.makedirs(os.path.join(self.path_to_root, self.prepareFolder))
        np.savez(os.path.join(self.path_to_root, self.prepareFolder, self.npzlist[actual_idx]), input=inp, target=tar)

    def prepare_all(self, overwrite=False):
        for i in range(self.maximum_elements):
            if not os.path.exists(os.path.join(self.path_to_root, self.prepareFolder, self.npzlist[i])) or overwrite:
                self.prepare(i)


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

def get_B0_and_GFR(magnet):
    trainer = magnetoptimiser.trainers.DefaultTrainer_Dipole_Hshape
    inputcolumns = ['gfr_x', 'gfr_y', 'gfr_margin', 'maxCurrentDensity', 'fieldTolerance', 'aper_x', 'aper_y', 'aper_x_poleoverhang', 'aper_y_distFromCoil', 'aper_x_tapering', 'aper_x_taperingstop', 'B_design', 'B_design_margin', 'B_real', 'coil_width', 'coil_height', 'yoke_x', 'yoke_y', 'maxBmaterial', 'fillfactor', 'windings', 'w', 'w_leg', 'totalCurrent', 'totalCurrentMax', 'coilAreaTotal', 'coilWeightTotal', 'coilVolumeTotal', 'length', 'yokeAreaTotal', 'yokeWeightTotal', 'yokeVolumeTotal', 'shims_False', 'shape_H', 'material_coil_Copper', 'material_yoke_Pure Iron', 'symmetry_reflectxydipole', 'coolingRequirementMax_Liquid Nitrogen', 'coolingRequirementMax_None', 'coolingRequirementMax_Water']
    # prop dict
    prop = magnet.get_properties()
    prop = pd.DataFrame(prop, index=[0])
    prop = pd.get_dummies(prop, columns=['shims', 'shape', 'material_coil', 'material_yoke', 'symmetry', 'coolingRequirementMax'])
    # check if 'coolingRequirementMax_Liquid Nitrogen', 'coolingRequirementMax_Water', 'coolingRequirementMax_None' are in the columns
    if 'coolingRequirementMax_Liquid Nitrogen' not in prop.columns:
        prop['coolingRequirementMax_Liquid Nitrogen'] = 0
    if 'coolingRequirementMax_Water' not in prop.columns:
        prop['coolingRequirementMax_Water'] = 0
    if 'coolingRequirementMax_None' not in prop.columns:
        prop['coolingRequirementMax_None'] = 0
    prop = prop.drop(columns=['name'])
    prop = prop[inputcolumns]
    prop = prop.values.astype(float)
    output = trainer.predict(prop)
    B0 = output[0, 0]
    gfr_x = output[0, 16]
    gfr_y = output[0, 32]
    return B0, gfr_x, gfr_y

def main():
    ds = Dataset_Dipole_H("data/raw/npz_select_1cmSpacing")
    ds.prepare_all(overwrite=True)
    plot_ds(ds, 0)

def trolo():
    # generate a polygon with 7 points
    md = magnetdesigner.designer.magnetdataset()
    md = md.load("data/raw/npz_select_1cmSpacing/md_hshaped_v1")
    dipole = md.get_magnet(23000)
    magnet = dipole.magnet2D
    magnet.yokeCoords[0] = np.array(magnet.yokeCoords[0])
    # round all coordinates smaller than 1e-3 to 0
    magnet.yokeCoords[0][np.abs(magnet.yokeCoords[0]) < 1e-3] = 0
    polygon = Polygon(magnet.yokeCoords[0])
    # print(polygon)
    # difference poligon and mirrored polygons
    inverse_polygon = Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]).difference(polygon)
    inverse_polygon = inverse_polygon.difference(scale(polygon, xfact=-1, yfact=1, origin=(0, 0)))
    inverse_polygon = inverse_polygon.difference(scale(polygon, xfact=1, yfact=-1, origin=(0, 0)))
    inverse_polygon = inverse_polygon.difference(scale(polygon, xfact=-1, yfact=-1, origin=(0, 0)))

    # generate an image that is 1 inside the polygon and 0 outside
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            # Z[i, j] = polygon.contains(Point(X[i, j], Y[i, j])) * distance(polygon, Point(X[i, j], Y[i, j]))
            # Z[i, j] = distance(polygon, Point(X[i, j], Y[i, j]))
            Z[i, j] = distance(inverse_polygon, Point(X[i, j], Y[i, j]))
            
            
    # plot the polygon
    x, y = polygon.exterior.xy
    fig, ax = plt.subplots()
    ax.imshow(Z, origin='lower', extent=(-1, 1, -1, 1))
    ax.plot(x, y)
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # main()
    # trolo()
    dataset = Dataset_Dipole_H("data/raw/npz_select_1cmSpacing")
    print(dataset[0])