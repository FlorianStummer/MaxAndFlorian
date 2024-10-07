import os
from torch.utils.data import DataLoader
from datasets.Dataset_Bend_H import Dataset_Bend_H
from models.UNet import UNet
from training.Trainer_Dipole_H import Trainer_Dipole_H
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models.LossFunctions import prioritized_loss

def get_metrics(arr):
    if type(arr) == int:
        arr = np.array([arr])
    # Flatten the 2D array to 1D
    flattened_array = arr.flatten()
    # Filter out the zero values
    
    if len(flattened_array[flattened_array != 0]) > 0:
        non_zero_values = flattened_array[flattened_array != 0]
    else:
        non_zero_values = flattened_array
    # Calculate mean, median, and max
    mean_value = np.mean(non_zero_values)
    median_value = np.median(non_zero_values)
    max_value = np.max(non_zero_values)
    # Calculate the 95th percentile
    perc95 = np.percentile(non_zero_values, 95)
    return mean_value, median_value, max_value, perc95, 1

def get_errors(model, test_loader, figureOfMerit="all", metric="mean", norm=None, device="cuda"):
    # create a histogram that shows the mean error for each sample
    error_x = []
    error_y = []
    error_mag = []
    error_ang = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        
        for sample in range(x.shape[0]):
            if figureOfMerit == "all":
                mask = x[sample, 17, :, :] > 0
            elif figureOfMerit == "aperture":
                mask = x[sample, 16, :, :] > 0
            elif figureOfMerit == "yoke":
                mask = x[sample, 0, :, :] > 0
            elif figureOfMerit == "coil":
                mask = x[sample, 3, :, :] > 0
            elif figureOfMerit == "rest":
                mask = (x[sample, 17, :, :] > 0) * (x[sample, 16, :, :] == 0) * (x[sample, 0, :, :] == 0) * (x[sample, 3, :, :] == 0)
            else:
                raise ValueError("Unknown figure of merit")
            mask = mask.cpu().detach().numpy().T

            with torch.no_grad():
                y_pred = model(x)
            tar_x = y[sample, 0, :, :].cpu().detach().numpy().T
            tar_y = y[sample, 1, :, :].cpu().detach().numpy().T
            pred_x = y_pred[sample, 0, :, :].cpu().detach().numpy().T
            pred_y = y_pred[sample, 1, :, :].cpu().detach().numpy().T
                
            tar_mag = np.sqrt(tar_x**2 + tar_y**2)
            tar_ang = np.arctan2(tar_y, tar_x)
            pred_mag = np.sqrt(pred_x**2 + pred_y**2)
            pred_ang = np.arctan2(pred_y, pred_x)

            tar_x = tar_x * mask
            tar_y = tar_y * mask
            pred_x = pred_x * mask
            pred_y = pred_y * mask
            tar_mag = tar_mag * mask
            tar_ang = tar_ang * mask
            pred_mag = pred_mag * mask
            pred_ang = pred_ang * mask
            
            # calculate the error
            err_x = np.abs(tar_x - pred_x)
            err_y = np.abs(tar_y - pred_y)
            err_mag = np.abs(tar_mag - pred_mag)
            err_ang = np.abs(np.mod(tar_ang - pred_ang + np.pi, 2*np.pi) - np.pi)

            if norm == "all":
                err_x = np.abs(np.divide(err_x, tar_x))
                err_y = np.abs(np.divide(err_y, tar_y))
                err_mag = np.abs(np.divide(err_mag, tar_mag))
                err_ang = np.abs(np.divide(err_ang, tar_ang))
            elif norm == "max":
                err_x = err_x / np.max(np.abs(tar_x))
                err_y = err_y / np.max(np.abs(tar_y))
                err_mag = err_mag / np.max(np.abs(tar_mag))
                err_ang = err_ang / np.max(np.abs(tar_ang))
            elif norm == "zero":
                err_x = err_x / np.abs(tar_x[0,0]) if tar_x[0,0] != 0 else 0
                err_y = err_y / np.abs(tar_y[0,0]) if tar_y[0,0] != 0 else 0
                err_mag = err_mag / np.abs(tar_mag[0,0]) if tar_mag[0,0] != 0 else 0
                err_ang = err_ang / np.abs(tar_ang[0,0]) if tar_ang[0,0] != 0 else 0
            elif norm == "zeroonly":
                err_x = np.array([err_x[0,0] / np.abs(tar_x[0,0])]) if tar_x[0,0] != 0 else 0
                err_y = np.array([err_y[0,0] / np.abs(tar_y[0,0])]) if tar_y[0,0] != 0 else 0
                err_mag = np.array([err_mag[0,0] / np.abs(tar_mag[0,0])]) if tar_mag[0,0] != 0 else 0
                err_ang = np.array([err_ang[0,0] / np.abs(tar_ang[0,0])]) if tar_ang[0,0] != 0 else 0
            else:
                pass

            # # plot mask
            # img = plt.imshow(err_x, origin='lower', vmin=0, vmax=0.1)
            # plt.colorbar(img)
            # plt.show()
            
            # get metrics
            mean_x, median_x, max_x, perc95_x, bvc_x = get_metrics(err_x)
            mean_y, median_y, max_y, perc95_y, bvc_y = get_metrics(err_y)
            mean_mag, median_mag, max_mag, perc95_mag, bvc_mag = get_metrics(err_mag)
            mean_ang, median_ang, max_ang,perc95_ang, bvc_ang = get_metrics(err_ang)

            if metric=="mean":
                error_x.append(mean_x)
                error_y.append(mean_y)
                error_mag.append(mean_mag)
                error_ang.append(mean_ang)
            elif metric=="median":
                error_x.append(median_x)
                error_y.append(median_y)
                error_mag.append(median_mag)
                error_ang.append(median_ang)
            elif metric=="max":
                error_x.append(max_x)
                error_y.append(max_y)
                error_mag.append(max_mag)
                error_ang.append(max_ang)
            elif metric=="perc95":
                error_x.append(perc95_x)
                error_y.append(perc95_y)
                error_mag.append(perc95_mag)
                error_ang.append(perc95_ang)
            elif metric=="bvc":
                error_x.append(bvc_x)
                error_y.append(bvc_y)
                error_mag.append(bvc_mag)
                error_ang.append(bvc_ang)
            else:
                raise ValueError("Unknown metric")
            

    error_x = np.array(error_x)
    error_y = np.array(error_y)
    error_mag = np.array(error_mag)
    error_ang = np.array(error_ang)
    return error_x, error_y, error_mag, error_ang


def main():
    dataset_path = "/eos/experiment/shadows/user/flstumme/ai/data/bend_h/prepared"
    path_to_root = "/eos/experiment/shadows/user/flstumme/ai/models/"
    if not os.path.exists(path_to_root):
        os.makedirs(path_to_root)
    maximum_elements = 75445
    traintestsplit = 0.1

    # hyperparameters
    batch_size = 128
    learning_rate = 0.001
    depth = 5
    wf = 4
    padding = True
    up_mode = 'upconv'
    batch_norm = True
    drop_out = 0
    
    # load dataset
    dataset = Dataset_Bend_H(dataset_path, maximum_elements=maximum_elements)
    print("Dataset loaded")

    # plot_ds(dataset, 3)

    # split dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*traintestsplit), int(len(dataset)*traintestsplit)], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_loader))
    print(len(test_loader))
    test_loader.dataset.dataset.print_info()

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
        print("Memory Allocated:", torch.cuda.memory_allocated(0))
        print("Memory Cached:", torch.cuda.memory_reserved(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using", device)
    
    # load model
    model = UNet(in_channels=18, out_channels=2, depth=depth, wf=wf, padding=padding, up_mode=up_mode, batch_norm=batch_norm, drop_out=drop_out)
    print(model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(optimizer)

    # set criterion
    # criterion = torch.nn.MSELoss()
    criterion = prioritized_loss
    print(criterion)

    # create trainer
    trainer = Trainer_Dipole_H(model, optimizer, criterion, device)
    trainer.load_model("/eos/experiment/shadows/user/flstumme/ai/models/UNet_Dipole_H_v2_epoch0019.pt")
    print("Trainer created")



    err = get_errors(model, test_loader, figureOfMerit="yoke", metric="perc95", norm=None, device=device)
    pickle.dump(err, open("errors_yoke.pkl", "wb"))

    # getErrorHistograms(model, test_loader, figureOfMerit="yoke", metric="perc95")

if __name__ == "__main__":
    main()
