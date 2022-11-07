import numpy as np
    
# ---------------------------------------------------------------
# Create input images/arrays for U-Net
# ---------------------------------------------------------------
def get_bins(length, granularity = 0.05):
    return int(length/granularity)

def round_nearest(x, a):
    return round(round(x / a) * a,5)
    
def create_unet_images(dimension_list, windings=80, granularity = 0.05):
    for i in range(len(dimension_list[:6])):
        dimension_list[i]=round_nearest(dimension_list[i],granularity)
    dimension_dict = dict(zip(['stage1_hole_x','stage1_hole_y','stage1_hole_dist','stage1_topyoke_y','stage1_sideyoke_left_x','stage1_sideyoke_right_x','stage1_I'],dimension_list))
    
    # calculate bins for binblocks in images
    bins_x = 121
    bins_y = 81
    
    bins_yoke_x = int(get_bins(dimension_dict['stage1_sideyoke_left_x']+dimension_dict['stage1_hole_x']+dimension_dict['stage1_sideyoke_right_x']))
    bins_yoke_y = int(get_bins(2*dimension_dict['stage1_hole_y']+2*dimension_dict['stage1_topyoke_y']+dimension_dict['stage1_hole_dist']))
    bins_hole_x = int(get_bins(dimension_dict['stage1_hole_x']))
    bins_hole_y = int(get_bins(dimension_dict['stage1_hole_y']))
    
    bins_x0 = int((bins_x-1)/2 - ( get_bins(dimension_dict['stage1_sideyoke_left_x'])+(get_bins(dimension_dict['stage1_hole_x'])-1)/2 ))
    bins_hole_x0 = int((bins_x+1)/2 - (get_bins(dimension_dict['stage1_hole_x'])-1)/2)
    bins_y0 = int((bins_y+1)/2 - ( get_bins(dimension_dict['stage1_topyoke_y'])+get_bins(dimension_dict['stage1_hole_y'])+(get_bins(dimension_dict['stage1_hole_dist'])+1)/2 ))
    bins_hole_y0_a = int((bins_y+1)/2 - ( get_bins(dimension_dict['stage1_hole_y'])+(get_bins(dimension_dict['stage1_hole_dist'])-1)/2 ))
    bins_hole_y0_b = int(bins_hole_y0_a + get_bins(dimension_dict['stage1_hole_y']+dimension_dict['stage1_hole_dist']))
    
    current = dimension_dict['stage1_I']*windings
    current_density = current/(dimension_dict['stage1_hole_x']*dimension_dict['stage1_hole_y']*1e6)
    # mat_Air = 0.0
    mat_ARMCO = 1.0
    # mat_Cu = -1.0
    mat_Cu = 0.0
    
    # create material image/array
    img_material = np.zeros((bins_x,bins_y))
    img_material[bins_x0:bins_x0+bins_yoke_x,bins_y0:bins_y0+bins_yoke_y] = mat_ARMCO*np.ones((bins_yoke_x,bins_yoke_y))
    img_material[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] = mat_Cu*np.ones((bins_hole_x,bins_hole_y))
    img_material[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = mat_Cu*np.ones((bins_hole_x,bins_hole_y))

    # create current density image/array
    img_currentdensity = np.zeros((bins_x,bins_y))
    img_currentdensity[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] = current_density*np.ones((bins_hole_x,bins_hole_y))
    img_currentdensity[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = -current_density*np.ones((bins_hole_x,bins_hole_y))

    # create material image/array
    fs_unmagnatizable=1e-6
    coil_x =   np.arange(bins_hole_x0,bins_hole_x0+bins_hole_x)
    coil_y_a = np.arange(bins_hole_y0_a,bins_hole_y0_a+bins_hole_y)
    coil_y_b = np.arange(bins_hole_y0_b,bins_hole_y0_b+bins_hole_y)

    img_fieldstrength = np.zeros((bins_x,bins_y))
    for xb in range(bins_x):
        for yb in range(bins_y):
            denominator_a = np.sqrt(np.min(np.abs(xb-coil_x)*granularity)**2+np.min(np.abs(yb-coil_y_a)*granularity)**2)
            denominator_b = np.sqrt(np.min(np.abs(xb-coil_x)*granularity)**2+np.min(np.abs(yb-coil_y_b)*granularity)**2)
            denominator_a = (denominator_a==0)+denominator_a
            denominator_b = (denominator_b==0)+denominator_b
            img_fieldstrength[xb,yb] = img_fieldstrength[xb,yb] + current/denominator_a + current/denominator_b
    img_fieldstrength[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    img_fieldstrength[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    img_fieldstrength = img_fieldstrength/np.max(np.abs(img_fieldstrength))

    # plt.imshow(img_material.T)
    # plt.imshow(img_currentdensity.T)
    # plt.imshow(img_fieldstrength.T)
    # plt.colorbar()
    # plt.show()

    unet_img_set = np.stack([img_material, img_currentdensity, img_fieldstrength], axis=0)
    
    return unet_img_set


# ---------------------------------------------------------------
# Get list of magnet variables from magnet list
# ---------------------------------------------------------------
def get_dimlist_from_magnetlist(ID, file = "../Stage_magnet/random_magnet_list.csv"):
    return np.genfromtxt(file, delimiter=',', unpack=True)[1:,ID+1].tolist()
