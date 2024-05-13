
from torch.utils.data import DataLoader
from datasets.Dataset_Dipole_H import Dataset_Dipole_H
from models.UNet import UNet
from models.LossFunctions import prioritized_loss
from training.Trainer_Dipole_H import Trainer_Dipole_H
import torch
import matplotlib.pyplot as plt
import pickle


def main():
    # dataset parameters
    dataset_path = "data/raw/npz_select_1cmSpacing"
    mdfile = "md_hshaped_v1"
    maximum_elements = 19200
    # maximum_elements = 2560
    # maximum_elements = 80
    traintestsplit = 0.1
    # prepareFolder = "prepared"
    # prepareFolder = "prepared_v1_weighted_o1"
    prepareFolder = "prepared_v2_weighted_o0_areaconsidered"

    # hyperparameters
    batch_size = 64
    # batch_size = 8
    learning_rate = 0.001
    num_epochs = 301
    # num_epochs = 11
    depth = 5
    wf = 4
    padding = True
    up_mode = 'upconv'
    batch_norm = True
    drop_out = 0
    
    # load dataset
    dataset = Dataset_Dipole_H(dataset_path, maximum_elements=maximum_elements, mdfile=mdfile, prepareFolder=prepareFolder)
    print("Dataset loaded")

    # split dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*traintestsplit), int(len(dataset)*traintestsplit)])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_loader))
    print(len(test_loader))

    # set device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using", device)

    # load model
    model = UNet(in_channels=5, out_channels=2, depth=depth, wf=wf, padding=padding, up_mode=up_mode, batch_norm=batch_norm, drop_out=drop_out)
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
    print("Trainer created")

    # train model
    trainer.run_epochs(train_loader, test_loader, num_epochs)

    # evaluate model
    trainer.load_model("UNet_Dipole_H.pt")
    with open("train_hist.pkl", "rb") as f:
        train_hist = pickle.load(f)
    with open("test_hist.pkl", "rb") as f:
        test_hist = pickle.load(f)
    with open("errors.pkl", "rb") as f:
        errors = pickle.load(f)
    print("Model loaded")

    trainer.getErrorHistograms(test_loader=test_loader, figureOfMerit="aperture")
    trainer.getErrorHistograms(test_loader=test_loader, figureOfMerit="magnet")
    trainer.getErrorHistograms(test_loader=test_loader, figureOfMerit="outer")

    # plot training history
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_hist)
    ax[0].set_title("Training Loss")
    ax[1].plot(test_hist)
    ax[1].set_title("Test Loss")
    plt.show()


if __name__ == "__main__":
    main()