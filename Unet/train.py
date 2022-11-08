import argparse
import sys
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../Stage_magnet')
import functions
from Unet_MaxProkop import UNet

magnetfolder = "../../Stage_magnet"

class SegmentationDataset(Dataset):
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

        # output pictures (for training)
        # load the fields from csv
        Bx,By = np.genfromtxt(imagePath, delimiter=',', unpack=True)[2:4]
        # maybe add samplesize increasements (data augumentation)
        # like -Bx/-By for negative I (x2)
        # or rotate by 180deg for sideyoke swap (x2)
        # or cropping (96x64 for example)
        # dont forget to synchronize with mask
        Bx = np.resize(Bx[1:], (81,121)).T
        By = np.resize(By[1:], (81,121)).T
        mask = np.stack([Bx, By], axis=0)
        mask = np.asarray(mask)[:,:120,:80]/2.5

        # input pictures
        image = functions.create_unet_images(functions.get_dimlist_from_magnetlist(idx,magnetfolder+"/random_magnet_list.csv"))[:,:120,:80]

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        # # plot to check the output (uncomment to have a look)
        # fig,ax = plt.subplots(2,3)
        # ax[0][0].imshow(image[0].T)
        # ax[0][1].imshow(image[1].T)
        # ax[0][2].imshow(image[2].T)
        # ax[1][0].imshow(mask[0].T)
        # ax[1][1].imshow(mask[1].T)
        # ax[1][2].axis('off')
        # plt.show()

        # return a tuple of the image and its mask
        return (image, mask)


def train_net(net,
              device,
              epochs: int = 40,
              batch_size: int = 16,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              ):
    # 1. Create dataset
    dataset = SegmentationDataset(np.sort(glob.glob(magnetfolder+'/Stage1_0*.csv'))[:8960], None)
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = []
        for X, Y in train_loader:

            Y = Y.to(device=device, dtype=torch.float32)
            X = X.to(device=device, dtype=torch.float32)

            pred = net(X)
            # print(pred.shape)
            loss = criterion(pred, Y)

            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_hist.append(np.mean(np.array(epoch_loss)))
        
        net.eval()
        epoch_loss = []
        for X, Y in val_loader:

            Y = Y.to(device=device, dtype=torch.float32)
            X = X.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(X)
                loss = criterion(pred, Y)
            epoch_loss.append(loss.item())
        val_loss_hist.append(np.mean(np.array(epoch_loss)))
        
        print('Epoch '+str(epoch)+': \ttrain_loss='+str(train_loss_hist[epoch-1])[0:10]+', \tval_loss='+str(val_loss_hist[epoch-1])[0:10])
    
    torch.save(net.state_dict(), "stagemagnet_unet_trained.pth")
    
    plt.plot([*range(1,epochs+1)],train_loss_hist)
    plt.plot([*range(1,epochs+1)],val_loss_hist)
    plt.yscale('log')
    plt.show()
    plt.close()
    
    net.to('cpu')

    f=pd.read_csv(magnetfolder+'/Stage1_000000.csv')
    X=np.unique(f['X'].values)
    Y=np.unique(f['Y'].values)
    nx=X.size
    ny=Y.size
    X,Y=np.meshgrid(X,Y)
    
    skip = (slice(None, None, 1), slice(None, None, 1))

    X_0, Y_0 = train_set[0]
    Bx = Y_0[0,:,:].T * 2.5
    By = Y_0[1,:,:].T * 2.5
    BB=np.sqrt(Bx**2+By**2)
    Bxdir,Bydir=np.divide(Bx,BB),np.divide(By,BB)

    fig = plt.figure()
    ax=fig.add_subplot(1, 2, 1)
    I_1 = ax.imshow(BB,extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)],cmap='gist_rainbow', vmin=0.0, vmax=2.0)
    Q_1 = ax.quiver(X[skip][:80,:120],Y[skip][:80,:120],Bxdir[skip],Bydir[skip])
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    # ax.set_title("I = {} A (80 windings)".format(dim_femm.stage1_I))
    fig.colorbar(I_1, ax=ax, label="Magnetic flux density [T]", extend="max")

    net.double()
    pred = net(torch.tensor(X_0).unsqueeze(0).double())

    Bx_pred = pred[0,0,:,:].detach().numpy().T * 2.5
    By_pred = pred[0,1,:,:].detach().numpy().T * 2.5
    BB_pred=np.sqrt(Bx_pred**2+By_pred**2)
    Bxdir_pred,Bydir_pred=np.divide(Bx_pred,BB_pred),np.divide(By_pred,BB_pred)

    ax2=fig.add_subplot(1, 2, 2)
    I_2 = ax2.imshow(BB_pred,extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)],cmap='gist_rainbow', vmin=0.0)#, vmax=2.0)
    Q_2 = ax2.quiver(X[skip][:80,:120],Y[skip][:80,:120],Bxdir_pred[skip],Bydir_pred[skip])
    ax2.set_xlabel("x [cm]")
    # ax2.set_ylabel("y [cm]")
    
    fig.colorbar(I_2, ax=ax2, label="Magnetic flux density [T]", extend="max")

    plt.show()








def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

# -------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(in_channels=3, out_channels=args.classes)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                #   img_scale=args.scale,
                  val_percent=args.val / 100,
                #   amp=args.amp
                  )
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        raise