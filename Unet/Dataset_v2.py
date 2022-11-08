import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import functions

magnetfolder = functions.magnetfolder

class MagnetDataset(Dataset):
    def __init__(self, imagePaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.transforms = transforms
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        
        # input pictures
        image_prep = functions.create_unet_images(functions.get_dimlist_from_magnetlist(idx,magnetfolder+"/random_magnet_list.csv"))[:,:120,:80]
        image = np.stack([image_prep[0]*image_prep[2], image_prep[1]], axis=0)
        
        # output pictures (for training)
        # load the fields from csv
        Bx,By = np.genfromtxt(imagePath, delimiter=',', unpack=True)[2:4]
        Bx = np.resize(Bx[1:], (81,121)).T
        By = np.resize(By[1:], (81,121)).T
        mask = np.stack([Bx[:120,:80]*image_prep[0], By[:120,:80]*image_prep[0]], axis=0)
        mask = np.asarray(mask)[:,:,:]/2.5

        
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        # # plot to check the output (uncomment to have a look)
        # fig,ax = plt.subplots(2,2)
        # ax[0][0].imshow(image[0].T)
        # ax[0][1].imshow(image[1].T)
        # # ax[0][2].imshow(image[2].T)
        # ax[1][0].imshow(mask[0].T)
        # ax[1][1].imshow(mask[1].T)
        # # ax[1][2].axis('off')
        # plt.show()

        # return a tuple of the image and its mask
        return (image, mask)