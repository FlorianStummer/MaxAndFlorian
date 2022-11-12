import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as _patches
import torch
from Unet_MaxProkop import UNet
import functions

magnetfolder = "../../Stage_magnet"
data_columns=['stage1_hole_x', 'stage1_hole_y', 'stage1_hole_dist', 'stage1_topyoke_y', 'stage1_sideyoke_left_x', 'stage1_sideyoke_right_x', 'stage1_I']

stage_init=[0.65, 0.05, 1.75, 1.05, 0.85, 1.35, 250.0]

delta_length = 0.05
delta_current = 10.0

magnetcolor='None'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# create the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
plt.subplots_adjust(left=0.25, bottom=0.40)
titlesize = 9
ticklabelsize=6
net = UNet(in_channels=4,out_channels=2,depth=4)
net.load_state_dict(torch.load("last_checkpoint.pth"))
net.eval()
net.to('cpu')

f=pd.read_csv(magnetfolder+'/Stage1_000000.csv')
X=np.unique(f['X'].values/100.)
Y=np.unique(f['Y'].values/100.)
nx=X.size
ny=Y.size
X,Y=np.meshgrid(X,Y)

skip = (slice(None, None, 1), slice(None, None, 1))

X_0 = functions.create_unet_images(stage_init)[:,:120,:80]

net.double()
pred = net(torch.tensor(X_0).unsqueeze(0).double())

Bx_pred = pred[0,0,:,:].detach().numpy().T * 2.5
By_pred = pred[0,1,:,:].detach().numpy().T * 2.5
BB_pred=np.sqrt(Bx_pred**2+By_pred**2)
Bxdir_pred,Bydir_pred=np.divide(Bx_pred,BB_pred),np.divide(By_pred,BB_pred)

I = ax.imshow(BB_pred,extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)],cmap='gist_rainbow', vmin=0.0)#, vmax=2.0)
Q = ax.quiver(X[skip][:80,:120],Y[skip][:80,:120],Bxdir_pred[skip],Bydir_pred[skip])
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")


l_magnet_yoke   = ax.add_patch(_patches.Rectangle( (-stage_init[0]/2.0-stage_init[4],-(2.0*stage_init[3]+2*stage_init[1]+stage_init[2])/2.0), stage_init[0]+stage_init[4]+stage_init[5], 2.0*stage_init[3]+2*stage_init[1]+stage_init[2], edgecolor='black',facecolor=magnetcolor, linewidth=0.5))
l_magnet_hole_u = ax.add_patch(_patches.Rectangle( (-stage_init[0]/2.0, stage_init[2]/2.0), stage_init[0], stage_init[1], edgecolor='black',facecolor='none', linewidth=0.5))
l_magnet_hole_d = ax.add_patch(_patches.Rectangle( (-stage_init[0]/2.0, -stage_init[2]/2.0-stage_init[1]), stage_init[0], stage_init[1],  edgecolor='black',facecolor='none', linewidth=0.5))
l_error         = ax.add_patch(_patches.Rectangle( (-3.0,-2.0), 6.0, 4.0,  edgecolor='white',facecolor='None', linewidth=2))
ax.set_xlim([-3.0,3.0])
ax.set_ylim([-2.0,2.0])
# ax.axis('off')

fig.colorbar(I, ax=ax, label="Magnetic flux density [T]", extend="max")

axcolor = 'lightgoldenrodyellow'
ax_hole_x =             plt.axes([0.25, 0.30, 0.65, 0.02], facecolor=axcolor)
ax_hole_y =             plt.axes([0.25, 0.26, 0.65, 0.02], facecolor=axcolor)
ax_hole_dist =          plt.axes([0.25, 0.22, 0.65, 0.02], facecolor=axcolor)
ax_topyoke_y =          plt.axes([0.25, 0.18, 0.65, 0.02], facecolor=axcolor)
ax_sideyoke_left_x =    plt.axes([0.25, 0.14, 0.65, 0.02], facecolor=axcolor)
ax_sideyoke_right_x =   plt.axes([0.25, 0.10, 0.65, 0.02], facecolor=axcolor)
ax_I =                  plt.axes([0.25, 0.06, 0.65, 0.02], facecolor=axcolor)

s_hole_x = Slider(ax_hole_x, 'hole x [m]', delta_length, 3.0, valinit=stage_init[0], valstep=delta_length)
s_hole_y = Slider(ax_hole_y, 'hole y [m]', delta_length, 3.0, valinit=stage_init[1], valstep=delta_length)
s_hole_dist = Slider(ax_hole_dist, 'hole distance [m]', delta_length, 4.0, valinit=stage_init[2], valstep=delta_length)
s_topyoke_y = Slider(ax_topyoke_y, 'top yoke [m]', delta_length, 4.0, valinit=stage_init[3], valstep=delta_length)
s_sideyoke_left_x = Slider(ax_sideyoke_left_x, 'side yoke left [m]', delta_length, 3.0, valinit=stage_init[4], valstep=delta_length)
s_sideyoke_right_x = Slider(ax_sideyoke_right_x, 'side yoke right [m]', delta_length, 3.0, valinit=stage_init[5], valstep=delta_length)
s_I = Slider(ax_I, 'current I [A]', 50.0, 250.0, valinit=stage_init[6], valstep=delta_current)

def update(val):
    hole_x = s_hole_x.val
    hole_y = s_hole_y.val
    hole_dist = s_hole_dist.val
    topyoke_y = s_topyoke_y.val
    sideyoke_left_x = s_sideyoke_left_x.val
    sideyoke_right_x = s_sideyoke_right_x.val
    current = s_I.val

    stage_list = [hole_x, hole_y,hole_dist,topyoke_y,sideyoke_left_x,sideyoke_right_x,current]
    
    l_magnet_yoke.set_bounds(-stage_list[0]/2.0-stage_list[4],-(2.0*stage_list[3]+2*stage_list[1]+stage_list[2])/2.0, stage_list[0]+stage_list[4]+stage_list[5], 2.0*stage_list[3]+2*stage_list[1]+stage_list[2])
    l_magnet_hole_u.set_bounds(-stage_list[0]/2.0, stage_list[2]/2.0, stage_list[0], stage_list[1])
    l_magnet_hole_d.set_bounds(-stage_list[0]/2.0, -stage_list[2]/2.0-stage_list[1], stage_list[0], stage_list[1])
    
    if(stage_list[0]+stage_list[4]+stage_list[5]>3.0 or 2.0*stage_list[3]+2*stage_list[1]+stage_list[2]>4.0 or stage_list[0]*stage_list[1]<0.03):
        l_error.set(edgecolor='red')
    else:  
        l_error.set(edgecolor='None')

    I.set_data(np.sqrt((net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,0,:,:].detach().numpy().T * 2.5)**2+(net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,1,:,:].detach().numpy().T * 2.5)**2))
    Q.set_UVC(np.divide(net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,0,:,:].detach().numpy().T * 2.5,np.sqrt((net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,0,:,:].detach().numpy().T * 2.5)**2+(net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,1,:,:].detach().numpy().T * 2.5)**2))[skip],np.divide(net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,1,:,:].detach().numpy().T * 2.5,np.sqrt((net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,0,:,:].detach().numpy().T * 2.5)**2+(net(torch.tensor(functions.create_unet_images(stage_list)[:,:120,:80]).unsqueeze(0).double())[0,1,:,:].detach().numpy().T * 2.5)**2))[skip])
   
    fig.canvas.draw_idle()

s_hole_x.on_changed(update)
s_hole_y.on_changed(update)
s_hole_dist.on_changed(update)
s_topyoke_y.on_changed(update)
s_sideyoke_left_x.on_changed(update)
s_sideyoke_right_x.on_changed(update)
s_I.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.03])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_hole_x.reset()
    s_hole_y.reset()
    s_hole_dist.reset()
    s_topyoke_y.reset()
    s_sideyoke_left_x.reset()
    s_sideyoke_right_x.reset()
    s_I.reset()
button.on_clicked(reset)

plt.show()