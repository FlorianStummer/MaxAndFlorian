# load pyg4ometry
# import pyg4ometry
import numpy as np
import pandas as pd
import torch
import os
import pybdsim
import pyg4ometry.geant4 as _g4
import pyg4ometry.gdml as _gdml
import pyg4ometry.visualisation as _vis
from Unet_MaxProkop import UNet
import functions

class StageMagnetBuilder:
    def __init__(self, name="StageMagnet"):
        self.name = name

        # registry to store gdml data
        self.reg  = _g4.Registry()

        # variables
        self.safety=_gdml.Constant("safety",0.0002 * 1000.0,self.reg)
        self.stage_I=_gdml.Constant("stage_I",250.0,self.reg)
        self.stage_windings=_gdml.Constant("stage_windings",80.0,self.reg)
        self.stage_hole_x=_gdml.Constant("stage_hole_x",0.3 * 1000.0,self.reg)
        self.stage_hole_y=_gdml.Constant("stage_hole_y",0.3 * 1000.0,self.reg)
        self.stage_hole_dist=_gdml.Constant("stage_hole_dist",0.6 * 1000.0,self.reg)
        self.stage_topyoke_y=_gdml.Constant("stage_topyoke_y",0.3 * 1000.0,self.reg)
        self.stage_sideyoke_left_x=_gdml.Constant("stage_sideyoke_left_x",0.3 * 1000.0,self.reg)
        self.stage_sideyoke_right_x=_gdml.Constant("stage_sideyoke_right_x",0.6 * 1000.0,self.reg)
        self.stage_x=_gdml.Constant("stage_x",self.stage_hole_x+self.stage_sideyoke_left_x+self.stage_sideyoke_right_x,self.reg)
        self.stage_y=_gdml.Constant("stage_y",self.stage_hole_dist+2*self.stage_hole_y+2*self.stage_topyoke_y,self.reg)
        self.stage_z=_gdml.Constant("stage_z",3.0 * 1000.0,self.reg)
        self.stage_hole_centre=_gdml.Constant("stage_hole_centre",self.stage_hole_dist/2.0+self.stage_hole_y/2.0,self.reg)
        self.sign_leftright = _gdml.Constant("sign_leftright", (self.stage_sideyoke_left_x-self.stage_sideyoke_right_x)/np.abs(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x),self.reg)
        self.parameterlist = [self.stage_hole_x.eval()/1000., self.stage_hole_y.eval()/1000., self.stage_hole_dist.eval()/1000., self.stage_topyoke_y.eval()/1000., self.stage_sideyoke_left_x.eval()/1000., self.stage_sideyoke_right_x.eval()/1000., self.stage_I.eval()]
        
        # world solid and logical
        self.ws_01 = _g4.solid.Box("ws_01",self.stage_x+np.abs(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x),self.stage_y,self.stage_z,self.reg)
        self.ws_02 = _g4.solid.Box("ws_02",self.stage_x+np.abs(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x),self.stage_y+5*self.safety,self.stage_z+5*self.safety,self.reg)
        self.ws   = _g4.solid.Subtraction("ws",self.ws_01,self.ws_02,[[0,0,0],[self.sign_leftright*self.stage_x,0,0]],self.reg)
        
        self.wl    = _g4.LogicalVolume(self.ws,"G4_Galactic","wl",self.reg)
        
        self.wg_coil1_solid = _g4.solid.Box("wg_coil1_solid",self.stage_hole_x-self.safety,self.stage_hole_y-self.safety,self.stage_z-self.safety,self.reg);
        self.wg_coil2_solid = _g4.solid.Box("wg_coil2_solid",self.stage_hole_x-self.safety,self.stage_hole_y-self.safety,self.stage_z-self.safety,self.reg);
        
        self.wg_coil1_logical    = _g4.LogicalVolume(self.wg_coil1_solid,"G4_Cu","wg_coil1_logical",self.reg)
        self.wg_coil1_physical   = _g4.PhysicalVolume([0,0,0],[0,self.stage_hole_centre,0],self.wg_coil1_logical,"wg_coil1_physical",self.wl,self.reg)
        self.wg_coil2_logical    = _g4.LogicalVolume(self.wg_coil2_solid,"G4_Cu","wg_coil2_logical",self.reg)
        self.wg_coil2_physical   = _g4.PhysicalVolume([0,0,0],[0,-self.stage_hole_centre,0],self.wg_coil2_logical,"wg_coil2_physical",self.wl,self.reg)

        # magnet yoke and coil  solids
        self.wg_yoke_01     = _g4.solid.Box("wg_yoke_01",self.stage_x-self.safety,self.stage_y-self.safety,self.stage_z-self.safety,self.reg)
        self.wg_hole1_cutout_solid = _g4.solid.Box("wg_hole1_cutout_solid",self.stage_hole_x+self.safety,self.stage_hole_y+self.safety,self.stage_z+5*self.safety,self.reg);
        self.wg_hole2_cutout_solid = _g4.solid.Box("wg_hole2_cutout_solid",self.stage_hole_x+self.safety,self.stage_hole_y+self.safety,self.stage_z+5*self.safety,self.reg);

        self.wg_yoke_02    = _g4.solid.Subtraction("wg_yoke_02",self.wg_yoke_01,self.wg_hole1_cutout_solid,[[0,0,0],[(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x)/2.0, self.stage_hole_centre,0]],self.reg)
        self.wg_yoke_solid       = _g4.solid.Subtraction("wg_yoke",self.wg_yoke_02,self.wg_hole2_cutout_solid,[[0,0,0],[(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x)/2.0,-self.stage_hole_centre,0]],self.reg)
        
        self.wg_yoke_logical    = _g4.LogicalVolume(self.wg_yoke_solid,"G4_Fe","wg_yoke_logical",self.reg)
        self.wg_yoke_physical   = _g4.PhysicalVolume([0,0,0],[-(self.stage_sideyoke_left_x-self.stage_sideyoke_right_x)/2.0,0,0],self.wg_yoke_logical,"wg_yoke_physical",self.wl,self.reg)

        self.reg.setWorld(self.wl.name)
        

    def view(self):
        v = _vis.VtkViewer()
        # self.reg.setWorld(self.wl.name)
        v.addLogicalVolume(self.wl)
        self.wl.checkOverlaps(recursive=True, coplanar=False)
        # self.wg_yoke_logical.checkOverlaps(recursive=True, coplanar=False)
        v.view()

    def update_parameters(self, param_list, z, windings=80.0):
        self.stage_I.setExpression(param_list[6])
        self.stage_windings.setExpression(windings)
        self.stage_hole_x.setExpression(param_list[0] * 1000.0)
        self.stage_hole_y.setExpression(param_list[1] * 1000.0)
        self.stage_hole_dist.setExpression(param_list[2] * 1000.0)
        self.stage_topyoke_y.setExpression(param_list[3] * 1000.0)
        self.stage_sideyoke_left_x.setExpression(param_list[4] * 1000.0)
        self.stage_sideyoke_right_x.setExpression(param_list[5] * 1000.0)
        self.stage_z.setExpression(z * 1000.0)

        self.reg.logicalVolumeDict['wl'].setSolid(self.ws)
        self.reg.logicalVolumeDict['wg_coil1_logical'].setSolid(self.wg_coil1_solid)
        self.reg.logicalVolumeDict['wg_coil2_logical'].setSolid(self.wg_coil2_solid)
        self.reg.logicalVolumeDict['wg_yoke_logical'].setSolid(self.wg_yoke_solid)

    def save_gdml(self, filename='magnet', foldername='./'):
        # create folder
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        # create gdml
        w = _gdml.Writer()
        w.addDetector(self.reg)
        w.write(foldername+filename+'.gdml')
        return w
        
    def save_fieldmap(self, filename='magnet', foldername='./'):
        # create folder
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        # create fieldmap
        f=pd.read_csv('../../Stage_magnet/Stage1_000000.csv')
        X=np.unique(f['X'].values)
        Y=np.unique(f['Y'].values)
        nx=X.size
        ny=Y.size
        X,Y=np.meshgrid(X[:120],Y[:80])
        X = X.T
        Y = Y.T
        # load trained unet
        model = UNet(in_channels=4,out_channels=2,depth=4)
        model.load_state_dict(torch.load("last_checkpoint.pth"))
        model.eval()
        model.to('cpu')
        # create inputs for unet
        parameterlist=[self.stage_hole_x.eval()/1000., self.stage_hole_y.eval()/1000., self.stage_hole_dist.eval()/1000., self.stage_topyoke_y.eval()/1000., self.stage_sideyoke_left_x.eval()/1000., self.stage_sideyoke_right_x.eval()/1000., self.stage_I.eval()]
        x = functions.create_unet_images(parameterlist)
        x = np.asarray(x)[:,:120,:80]
        # predict the fields with the unet
        fields = model(torch.tensor(x).unsqueeze(0).float()).squeeze().detach().numpy() 
        # merge fieldmap array
        fm = np.stack((X, Y, fields[0], fields[1], np.zeros(X.shape)), axis = 2)
        # save fieldmap in bdsimformat
        f = pybdsim.Field.Field2D(fm)
        f.Write(foldername+filename+'.dat')

    def save(self, gdmlname='magnet', fieldmapname='magnet', foldername='magnet/'):
        # create folder
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        # create gdml
        w = self.save_gdml(gdmlname, foldername)
        w.writeGmadTester(foldername+gdmlname+'.gmad', gdmlname+'.gdml')
        # create fieldmap
        self.save_fieldmap(fieldmapname, foldername)
        # update tester to include the field
        line = ['magnet_field: field, type="bmap2d", integrator = "g4classicalrk4", magneticFile = "bdsim2d:{}.dat", magneticInterpolator = "linear", bScaling=1;\n'.format(fieldmapname)]
        with open(foldername+gdmlname+'.gmad', 'r') as f:
            content = f.readlines()
            print(content[0])
            content[0] = content[0].replace('*mm;', '*mm, fieldAll="magnet_field";')
        with open(foldername+gdmlname+'.gmad', 'w') as g:
            g.writelines(line + content)