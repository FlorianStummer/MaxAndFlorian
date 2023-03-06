import numpy as np

def get_bins(length, granularity = 0.05):
    return int(round(length/granularity, 5))

def round_nearest(x, a):
    return round(round(x / a) * a, 5)
    
def create_unet_images(dimension_list, windings=80, granularity = 0.05):
    for i in range(len(dimension_list[:6])):
        dimension_list[i]=round_nearest(dimension_list[i],granularity)
        # correct rounding errors
        if i in [0,2]:
            if(get_bins(dimension_list[i])%2==0):
                dimension_list[i]=round(dimension_list[i]-granularity,5)
    dimension_dict = dict(zip(['stage1_hole_x','stage1_hole_y','stage1_hole_dist','stage1_topyoke_y','stage1_sideyoke_left_x','stage1_sideyoke_right_x','stage1_I'],dimension_list))
    # calculate bins for binblocks in images
    bins_x = 121
    bins_y = 81
    
    bins_yoke_x = int(get_bins(dimension_dict['stage1_sideyoke_left_x'])+get_bins(dimension_dict['stage1_hole_x'])+get_bins(dimension_dict['stage1_sideyoke_right_x']))
    bins_yoke_y = int(2*get_bins(dimension_dict['stage1_hole_y'])+2*get_bins(dimension_dict['stage1_topyoke_y'])+get_bins(dimension_dict['stage1_hole_dist']))
    bins_hole_x = int(get_bins(dimension_dict['stage1_hole_x']))
    bins_hole_y = int(get_bins(dimension_dict['stage1_hole_y']))
    
    bins_x0 = int((bins_x-1)/2 - ( get_bins(dimension_dict['stage1_sideyoke_left_x'])+(get_bins(dimension_dict['stage1_hole_x'])-1)/2 ))
    bins_hole_x0 = int(bins_x0 + get_bins(dimension_dict['stage1_sideyoke_left_x']))
    bins_y0 = int((bins_y-1)/2 - ( get_bins(dimension_dict['stage1_topyoke_y'])+get_bins(dimension_dict['stage1_hole_y'])+(get_bins(dimension_dict['stage1_hole_dist'])-1)/2 ))
    if(2*bins_y0+bins_yoke_y < 81):
        bins_y0 = bins_y0+1
    if(2*bins_y0+bins_yoke_y > 81):
        bins_y0 = bins_y0-1
    bins_hole_y0_a = int(bins_y0 + get_bins(dimension_dict['stage1_topyoke_y']))
    bins_hole_y0_b = int(bins_hole_y0_a + get_bins(dimension_dict['stage1_hole_y']) + get_bins(dimension_dict['stage1_hole_dist']))
    
    current = dimension_dict['stage1_I']*windings
    current_max = 250.0*windings
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
    img_currentdensity[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] =  current_density*np.ones((bins_hole_x,bins_hole_y))
    img_currentdensity[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = -current_density*np.ones((bins_hole_x,bins_hole_y))

    # create material image/array
    coil_x =   np.arange(bins_hole_x0,bins_hole_x0+bins_hole_x)
    coil_y_a = np.arange(bins_hole_y0_a,bins_hole_y0_a+bins_hole_y)
    coil_y_b = np.arange(bins_hole_y0_b,bins_hole_y0_b+bins_hole_y)

    img_fieldstrength_x = np.zeros((bins_x,bins_y))
    img_fieldstrength_y = np.zeros((bins_x,bins_y))

    for xb in range(bins_x):
        for yb in range(bins_y):
            dx   = np.min(np.abs(xb-coil_x)*granularity)
            dy_a = np.min(np.abs(yb-coil_y_a)*granularity)
            dy_b = np.min(np.abs(yb-coil_y_b)*granularity)
            if ( dx==0 ):
                alpha_a = np.pi/2
            else:
                alpha_a = np.arctan(dy_a/dx)            
            if ( dx==0 ):
                alpha_b = np.pi/2 
            else:
                alpha_b = np.arctan(dy_b/dx)
            sign_x = np.sign((xb-coil_x)[np.where(np.abs(xb-coil_x) == np.min(np.abs(xb-coil_x)))])
            sign_y_a = np.sign((yb-coil_y_a)[np.where(np.abs(yb-coil_y_a) == np.min(np.abs(yb-coil_y_a)))])
            sign_y_b = np.sign((yb-coil_y_b)[np.where(np.abs(yb-coil_y_b) == np.min(np.abs(yb-coil_y_b)))])
            denominator_a = np.sqrt(dx**2+dy_a**2)
            denominator_b = np.sqrt(dx**2+dy_b**2)
            denominator_a = (denominator_a==0)+denominator_a
            denominator_b = (denominator_b==0)+denominator_b
            img_fieldstrength_x[xb,yb] = img_fieldstrength_x[xb,yb] + current*( sign_y_a*np.sin(alpha_a))/denominator_a + (-current)*( sign_y_b*np.sin(alpha_b))/denominator_b
            img_fieldstrength_y[xb,yb] = img_fieldstrength_y[xb,yb] + current*(-sign_x  *np.cos(alpha_a))/denominator_a + (-current)*(-sign_x  *np.cos(alpha_b))/denominator_b
    # img_fieldstrength_x[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    # img_fieldstrength_x[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    # img_fieldstrength_y[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_a:bins_hole_y0_a+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    # img_fieldstrength_y[bins_hole_x0:bins_hole_x0+bins_hole_x,bins_hole_y0_b:bins_hole_y0_b+bins_hole_y] = np.zeros((bins_hole_x,bins_hole_y))
    img_fieldstrength_x = img_fieldstrength_x*img_material/np.max(np.abs(img_fieldstrength_x)) #*current/current_max
    img_fieldstrength_y = img_fieldstrength_y*img_material/np.max(np.abs(img_fieldstrength_y)) #*current/current_max
    img_fieldstrength_x[np.abs(img_fieldstrength_x) < 0.01] = 0
    img_fieldstrength_y[np.abs(img_fieldstrength_y) < 0.01] = 0

    unet_img_set = np.flip(np.stack([img_material, img_currentdensity, img_fieldstrength_x, img_fieldstrength_y], axis=0), axis=1)
    
    return unet_img_set

def optimized_Unet_input(dimension_list, x_bins=121, y_bins=81, bin_size = 0.05):
    dimension_dict = dict(zip(['stage1_hole_x','stage1_hole_y','stage1_hole_dist','stage1_topyoke_y','stage1_sideyoke_left_x','stage1_sideyoke_right_x','stage1_I'],dimension_list))
    bin_idxs = {}
    bin_idxs["yoke_size"] = \
             {"x":(dimension_dict['stage1_sideyoke_left_x'] + dimension_dict['stage1_hole_x'] + dimension_dict['stage1_sideyoke_right_x'])/bin_size, 
              "y":(2 * dimension_dict['stage1_hole_y'] + 2 * dimension_dict['stage1_topyoke_y'] + dimension_dict['stage1_hole_dist'])/bin_size}
    # TODO 
