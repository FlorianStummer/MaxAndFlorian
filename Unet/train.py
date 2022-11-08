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


sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../Stage_magnet')
import functions
from Unet_MaxProkop import UNet
from Dataset_v2 import MagnetDataset

magnetfolder = functions.magnetfolder




def train_net(net,
              device,
              epochs: int = 40,
              batch_size: int = 16,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              ):
    # 1. Create dataset
    dataset = MagnetDataset(np.sort(glob.glob(magnetfolder+'/Stage1_0*.csv'))[:5120], None)    # 8960
    
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
    net.double()
    
    # plot to check the output (first row: input, second row: desired output, trird row: predicted output)
    fig,ax = plt.subplots(3,3)

    image, mask = train_set[0]
    mask_pred = net(torch.tensor(image).unsqueeze(0).double())

    ax[0][0].imshow(image[0].T)
    ax[0][1].imshow(image[1].T)
    # ax[0][2].imshow(image[2].T)
    ax[0][2].axis('off')
    ax[1][0].imshow(mask[0].T)
    ax[1][1].imshow(mask[1].T)
    # ax[1][2].axis('off')
    ax[2][0].imshow(mask_pred[0,0,:,:].detach().numpy().T)
    ax[2][1].imshow(mask_pred[0,1,:,:].detach().numpy().T)
    # ax[2][2].axis('off')
    plt.show()


    # # Old plot
    # f=pd.read_csv(magnetfolder+'/Stage1_000000.csv')
    # X=np.unique(f['X'].values)
    # Y=np.unique(f['Y'].values)
    # nx=X.size
    # ny=Y.size
    # X,Y=np.meshgrid(X,Y)
    
    # skip = (slice(None, None, 1), slice(None, None, 1))

    # X_0, Y_0 = train_set[0]
    # Bx = Y_0[0,:,:].T * 2.5
    # By = Y_0[1,:,:].T * 2.5
    # BB=np.sqrt(Bx**2+By**2)
    # Bxdir,Bydir=np.divide(Bx,BB),np.divide(By,BB)

    # fig = plt.figure()
    # ax=fig.add_subplot(1, 2, 1)
    # I_1 = ax.imshow(BB,extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)],cmap='gist_rainbow', vmin=0.0, vmax=2.0)
    # Q_1 = ax.quiver(X[skip][:80,:120],Y[skip][:80,:120],Bxdir[skip],Bydir[skip])
    # ax.set_xlabel("x [cm]")
    # ax.set_ylabel("y [cm]")
    # # ax.set_title("I = {} A (80 windings)".format(dim_femm.stage1_I))
    # fig.colorbar(I_1, ax=ax, label="Magnetic flux density [T]", extend="max")

    # net.double()
    # pred = net(torch.tensor(X_0).unsqueeze(0).double())

    # Bx_pred = pred[0,0,:,:].detach().numpy().T * 2.5
    # By_pred = pred[0,1,:,:].detach().numpy().T * 2.5
    # BB_pred=np.sqrt(Bx_pred**2+By_pred**2)
    # Bxdir_pred,Bydir_pred=np.divide(Bx_pred,BB_pred),np.divide(By_pred,BB_pred)

    # ax2=fig.add_subplot(1, 2, 2)
    # I_2 = ax2.imshow(BB_pred,extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)],cmap='gist_rainbow', vmin=0.0)#, vmax=2.0)
    # Q_2 = ax2.quiver(X[skip][:80,:120],Y[skip][:80,:120],Bxdir_pred[skip],Bydir_pred[skip])
    # ax2.set_xlabel("x [cm]")
    # # ax2.set_ylabel("y [cm]")
    
    # fig.colorbar(I_2, ax=ax2, label="Magnetic flux density [T]", extend="max")

    plt.show()








def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=512, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-4,
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
    net = UNet(in_channels=2, out_channels=args.classes)

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