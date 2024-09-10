import numpy as np
import pandas as pd
import os
import sys
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
from shapely.measurement import distance
import magnetdesigner
import magnetoptimiser

class PrepareDataset:
    def __init__(self, path_to_root, prepareFolder, N):
        self.path_to_root = path_to_root
        self.prepareFolder = prepareFolder
        dsetfolders = ['random_5mm', 'random_large_5mm', 'random_small_5mm', 'straight_5mm']
        self.npzlist = []
        for dsetfolder in dsetfolders:
            self.npzlist += [os.path.join(path_to_root, dsetfolder, f) for f in os.listdir(os.path.join(self.path_to_root, dsetfolder)) if f.endswith('.npz')]
        self.npzlist.sort()

        md_random = magnetdesigner.designer.magnetdataset()
        md_random = md_random.load(os.path.join(self.path_to_root, 'md_dipole_hshaped_v2_random.pkl'))
        md_random_large = magnetdesigner.designer.magnetdataset()
        md_random_large = md_random_large.load(os.path.join(self.path_to_root, 'md_dipole_hshaped_v2_random_large.pkl'))
        md_random_small = magnetdesigner.designer.magnetdataset()
        md_random_small = md_random_small.load(os.path.join(self.path_to_root, 'md_dipole_hshaped_v2_random_small.pkl'))
        md_straight = magnetdesigner.designer.magnetdataset()
        md_straight = md_straight.load(os.path.join(self.path_to_root, 'md_dipole_hshaped_v2_straight.pkl'))
        print("Magnetdesigner loaded")

        datacollection = []
        for npz in self.npzlist:
            num = npz.split('/')[-1][:-4]
            if 'random_5mm' in npz:
                datacollection.append({'magnet': md_random.get_magnet_by_name(num), 'npz': npz})
            elif 'random_large_5mm' in npz:
                datacollection.append({'magnet': md_random_large.get_magnet_by_name(num), 'npz': npz})
            elif 'random_small_5mm' in npz:
                datacollection.append({'magnet': md_random_small.get_magnet_by_name(num), 'npz': npz})
            elif 'straight_5mm' in npz:
                datacollection.append({'magnet': md_straight.get_magnet_by_name(num), 'npz': npz})
        print("Datacollection length:", len(datacollection))

        allmagnets = magnetdesigner.designer.magnetdataset()
        for data in datacollection:
            allmagnets.append_magnet(data['magnet'])
        allmagnets = magnetdesigner.designer.magnetdataset()
        allmagnets.magnets = md_random.magnets + md_random_large.magnets + md_random_small.magnets + md_straight.magnets
        print("Allmagnets length:", len(allmagnets.magnets))

        # get the spacing from the first npz file
        dummy = np.load(os.path.join(self.npzlist[0]))['data']
        xlist = list(set(dummy[:,:,0].flatten()))
        xlist.sort()
        ylist = list(set(dummy[:,:,1].flatten()))
        ylist.sort()
        self.spacing = np.round(np.abs(xlist[0] - xlist[1]), 5) * 1e-3

        # get the name of the largest magnet covered by the npzs from magnetinfos
        _, self.maxX, _, self.maxY = allmagnets.get_largest_magnet_dimensions()
        self.xbins = int(np.ceil(self.maxX / self.spacing))
        self.ybins = int(np.ceil(self.maxY / self.spacing))

        self.xbinsFixed = 224
        self.ybinsFixed = 160

        print("Spacing:", self.spacing)
        print("MaxX:", self.maxX)
        print("MaxY:", self.maxY)
        print("Xbins:", self.xbins)
        print("Ybins:", self.ybins)
        print(allmagnets.get_magnets_number())

        if not os.path.exists(self.prepareFolder):
            os.makedirs(self.prepareFolder)
        for idx, entry in enumerate(datacollection):
            if N * 100 <= idx < (N + 1) * 100:
                print("Preparing", idx)
                self.prepare(entry['magnet'], entry['npz'], str(idx).zfill(8))


    def prepare(self, dipole, npz, idx):
        magnet = dipole.magnet2D
        # prepare the input
        inp = np.zeros((self.xbinsFixed, self.ybinsFixed, 18))
        # channel 0, 1 and 2: the yoke region and the distance to the yoke boundary (absolute and normalised)
        for yoke_i in magnet.yokeCoords:
            coords = np.array(yoke_i)
            coords[np.abs(coords) < 1e-3] = 0
            yoke = Polygon(coords)
            inverse_polygon = Polygon([(-self.xbinsFixed*self.spacing, -self.ybinsFixed*self.spacing), (-self.xbinsFixed*self.spacing, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, -self.ybinsFixed*self.spacing)])
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
            inverse_polygon = Polygon([(0, 0), (0, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, 0)])
            inverse_polygon = inverse_polygon.difference(coil)
            for i in range(self.xbinsFixed):
                for j in range(self.ybinsFixed):
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
        inverse_polygon = Polygon([(-self.xbinsFixed*self.spacing, -self.ybinsFixed*self.spacing), (-self.xbinsFixed*self.spacing, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, self.ybinsFixed*self.spacing), (self.xbinsFixed*self.spacing, -self.ybinsFixed*self.spacing)])
        inverse_polygon = inverse_polygon.difference(polygon)
        for i in range(self.xbins):
            for j in range(self.ybins):
                inp[i, j, 14] = distance(inverse_polygon, Point(i*self.spacing, j*self.spacing))
        inp[:,:,15] = inp[:,:,14] / np.max(inp[:,:,14])

        # channel 16: DNN prediction in the aperture
        B0, gfr_x, gfr_y = self.get_B0_and_GFR(dipole)
        for i in range(self.xbins):
            for j in range(self.ybins):
                if i*self.spacing <= gfr_x and j*self.spacing <= gfr_y:
                    inp[i, j, 16] = B0
        


        # create input image 17: priority map (Magnet --> 1, rest / outside --> 0)
        for i in range(self.xbinsFixed):
            for j in range(self.ybinsFixed):
                inp[i, j, 17] = 0
                if i*self.spacing < dipole.yoke_x * 0.5 and j*self.spacing < dipole.yoke_y * 0.5:
                    inp[i, j, 17] = 0.1 / (dipole.yoke_x * dipole.yoke_y * 0.25) 
                # if i*self.spacing < dipole.aper_x * 0.5 * 2.0 and j*self.spacing < dipole.aper_y * 0.5:
                #     inp[i, j, 17] = 100 / (dipole.aper_x * dipole.aper_y * 0.25)

        # prepare the target
        tar_unprepared = np.load(npz)['data']
        tar = np.zeros((self.xbinsFixed, self.ybinsFixed, 2))
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
        # if not os.path.exists(self.prepareFolder):
        #     os.makedirs(self.prepareFolder)
        # np.savez(os.path.join(self.prepareFolder, idx), input=inp, target=tar)
        np.savez(".", input=inp, target=tar)

    def get_B0_and_GFR(self, dipole):
        trainer = magnetoptimiser.wrappers.Wrapper_Dipole_Hshape_RegNet_v002
        inputcolumns = ['fieldTolerance', 'maxCurrentDensity', 'rho0', 'usedPowerInPercent', 'w_leg_factor', 'B_design', 'aper_x', 'aper_y']
        prop = dipole.get_properties()
        prop = pd.DataFrame(prop, index=[0])
        # only keep the columns that are in the inputcolumns
        prop = prop[inputcolumns]
        prop = prop.values.astype(float)
        output = trainer.predict(prop)
        B0 = output[0, 0]
        gfr_x = output[0, 17]
        gfr_y = output[0, 33]
        return B0, gfr_x, gfr_y
        
        
    # def get_B0_and_GFR(magnet):
    #     trainer = magnetoptimiser.trainers.DefaultTrainer_Dipole_Hshape
    #     inputcolumns = ['gfr_x', 'gfr_y', 'gfr_margin', 'maxCurrentDensity', 'fieldTolerance', 'aper_x', 'aper_y', 'aper_x_poleoverhang', 'aper_y_distFromCoil', 'aper_x_tapering', 'aper_x_taperingstop', 'B_design', 'B_design_margin', 'B_real', 'coil_width', 'coil_height', 'yoke_x', 'yoke_y', 'maxBmaterial', 'fillfactor', 'windings', 'w', 'w_leg', 'totalCurrent', 'totalCurrentMax', 'coilAreaTotal', 'coilWeightTotal', 'coilVolumeTotal', 'length', 'yokeAreaTotal', 'yokeWeightTotal', 'yokeVolumeTotal', 'shims_False', 'shape_H', 'material_coil_Copper', 'material_yoke_Pure Iron', 'symmetry_reflectxydipole', 'coolingRequirementMax_Liquid Nitrogen', 'coolingRequirementMax_None', 'coolingRequirementMax_Water']
    #     # prop dict
    #     prop = magnet.get_properties()
    #     prop = pd.DataFrame(prop, index=[0])
    #     prop = pd.get_dummies(prop, columns=['shims', 'shape', 'material_coil', 'material_yoke', 'symmetry', 'coolingRequirementMax'])
    #     # check if 'coolingRequirementMax_Liquid Nitrogen', 'coolingRequirementMax_Water', 'coolingRequirementMax_None' are in the columns
    #     if 'coolingRequirementMax_Liquid Nitrogen' not in prop.columns:
    #         prop['coolingRequirementMax_Liquid Nitrogen'] = 0
    #     if 'coolingRequirementMax_Water' not in prop.columns:
    #         prop['coolingRequirementMax_Water'] = 0
    #     if 'coolingRequirementMax_None' not in prop.columns:
    #         prop['coolingRequirementMax_None'] = 0
    #     prop = prop.drop(columns=['name'])
    #     prop = prop[inputcolumns]
    #     prop = prop.values.astype(float)
    #     output = trainer.predict(prop)
    #     B0 = output[0, 0]
    #     gfr_x = output[0, 16]
    #     gfr_y = output[0, 32]
    #     return B0, gfr_x, gfr_y

def main(N):
    # PrepareDataset("data/bend_h", "data/bend_h/prepared", N)
    PrepareDataset("/eos/experiment/shadows/user/flstumme/ai/data/bend_h/raw", "/eos/experiment/shadows/user/flstumme/ai/data/bend_h/prepared", N)

if __name__ == "__main__":
    # get N from command line
    N = int(sys.argv[1])
    print("N:", N)
    # N = 0
    main(N)