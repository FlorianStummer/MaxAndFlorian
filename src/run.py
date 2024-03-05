
from torch.utils.data import DataLoader
from datasets.Dataset_Dipole_H import Dataset_Dipole_H
from models.UNet import UNet
from training.Trainer_Dipole_H import Trainer_Dipole_H
import torch
import matplotlib.pyplot as plt


def main():
    # dataset parameters
    dataset_path = "data/raw/npz_select_1cmSpacing"
    mdfile = "md_hshaped_v1"
    maximum_elements = 400
    traintestsplit = 0.1

    # hyperparameters
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 10
    depth = 3
    wf = 4
    padding = True
    up_mode = 'upconv'
    drop_out = 0
    
    # load dataset
    dataset = Dataset_Dipole_H(dataset_path, maximum_elements=maximum_elements, mdfile=mdfile)
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
    model = UNet(in_channels=5, out_channels=2, depth=depth, wf=wf, padding=padding, batch_norm=True, up_mode=up_mode, drop_out=drop_out)
    print(model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(optimizer)

    # set criterion
    criterion = torch.nn.MSELoss()
    print(criterion)

    # create trainer
    trainer = Trainer_Dipole_H(model, optimizer, criterion, device)
    print("Trainer created")

    # train model
    train_hist = trainer.train(train_loader, num_epochs)
    
    # test model
    test_hist = trainer.test(test_loader)

    # save model
    trainer.save_model("UNet_Dipole_H.pt")

    # plot training history
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_hist)
    ax[0].set_title("Training Loss")
    ax[1].plot(test_hist)
    ax[1].set_title("Test Loss")
    plt.show()


if __name__ == "__main__":
    main()