
import torch
import time
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class Trainer_Dipole_H:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def run_epochs(self, train_loader, test_loader, epochs, model_name="UNet", path_to_root=""):
        time_start = time.time()
        timestamp = time.time()
        times = []
        train_hist = []
        test_hist = []
        for epoch in tqdm(range(epochs)):
            train_hist.append(self.train(train_loader))
            test_hist.append(self.test(test_loader))
            times.append(time.time() - timestamp)
            timestamp = time.time()
            if epoch % 1 == 0:
                self.save_model("{}{}_epoch{}.pt".format(path_to_root, model_name, str(epoch).zfill(4)))
                self.save_hist(train_hist, test_hist, times, path_to_root)
                # self.evaluate(test_loader)
        self.save_model("{}{}_epoch{}.pt".format(path_to_root, model_name, str(epochs).zfill(4)))
        print('Total time:', time.time() - time_start)
    
    def getweights(self, data):
        weights = data[:, 17, :, :]
        # weights[weights == 1] = 0.0
        weights = weights.unsqueeze(1).expand(-1, 2, -1, -1)
        weights = weights.to(self.device, dtype=torch.float32)
        return weights
    
    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        for data, target in train_loader:
            # skip sample if it is None (e.g. due to an error during loading)
            if data is None:
                continue
            # run training step
            weights = self.getweights(data)
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(data)
            # loss = self.criterion(output, target)
            loss = self.criterion(output, target, weights)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                # skip sample if it is None (e.g. due to an error during loading)
                if data is None:
                    continue
                # run test step
                weights = self.getweights(data)
                data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
                output = self.model(data)
                # test_loss += self.criterion(output, target).item()
                test_loss += self.criterion(output, target, weights).item()
        return test_loss
    
    def get_errors(self, test_loader, figureOfMerit="all"):
        self.model.eval()
        if figureOfMerit == "all":
            cl = None
        elif figureOfMerit == "aperture":
            cl = 10
        elif figureOfMerit == "magnet":
            cl = 3
        elif figureOfMerit == "outer":
            cl = 1
        else:
            raise ValueError("Unknown figure of merit")
        # create a histogram that shows the mean error for each sample
        error_x = []
        error_y = []
        error_mag = []
        error_ang = []
        for i, (x, y) in enumerate(test_loader):
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            for sample in range(x.shape[0]):
                with torch.no_grad():
                    y_pred = self.model(x)
                tar_x = y[sample, 0, :, :].cpu().detach().numpy().T
                tar_y = y[sample, 1, :, :].cpu().detach().numpy().T
                pred_x = y_pred[sample, 0, :, :].cpu().detach().numpy().T
                pred_y = y_pred[sample, 1, :, :].cpu().detach().numpy().T
                if cl is not None:
                    mask = x[sample, 4, :, :] == cl
                    mask = mask.cpu().detach().numpy().T
                else:
                    mask = None
                err_x = np.abs(tar_x - pred_x)
                err_y = np.abs(tar_y - pred_y)
                err_mag = np.abs(np.sqrt(tar_x**2 + tar_y**2) - np.sqrt(pred_x**2 + pred_y**2))
                err_ang = np.abs(((np.arctan2(tar_y, tar_x) - np.arctan2(pred_y, pred_x)) + np.pi) % (2*np.pi) - np.pi)
                error_x.append(np.mean(np.ma.masked_array(err_x, mask=mask)))
                error_y.append(np.mean(np.ma.masked_array(err_y, mask=mask)))
                error_mag.append(np.mean(np.ma.masked_array(err_mag, mask=mask)))
                error_ang.append(np.mean(np.ma.masked_array(err_ang, mask=mask)))
        error_x = np.array(error_x)
        error_y = np.array(error_y)
        error_mag = np.array(error_mag)
        error_ang = np.array(error_ang)
        return error_x, error_y, error_mag, error_ang
    
    def evaluate(self, test_loader):
        self.model.eval()
        # load or create error dictionary
        if os.path.exists("errors.pkl"):
            with open("errors.pkl", "rb") as file:
                error_dict = pickle.load(file)
        else:
            error_dict = {"aperture": {"x": [], "y": [], "mag": [], "ang": []},
                          "magnet": {"x": [], "y": [], "mag": [], "ang": []},
                          "outer": {"x": [], "y": [], "mag": [], "ang": []}}
        # append new errors
        errors_aperture = self.get_errors(test_loader, "aperture")
        errors_magnet = self.get_errors(test_loader, "magnet")
        errors_outer = self.get_errors(test_loader, "outer")
        error_dict["aperture"]["x"].append(errors_aperture[0])
        error_dict["aperture"]["y"].append(errors_aperture[1])
        error_dict["aperture"]["mag"].append(errors_aperture[2])
        error_dict["aperture"]["ang"].append(errors_aperture[3])
        error_dict["magnet"]["x"].append(errors_magnet[0])
        error_dict["magnet"]["y"].append(errors_magnet[1])
        error_dict["magnet"]["mag"].append(errors_magnet[2])
        error_dict["magnet"]["ang"].append(errors_magnet[3])
        error_dict["outer"]["x"].append(errors_outer[0])
        error_dict["outer"]["y"].append(errors_outer[1])
        error_dict["outer"]["mag"].append(errors_outer[2])
        error_dict["outer"]["ang"].append(errors_outer[3])
        # save error dictionary
        with open("errors.pkl", "wb") as file:
            pickle.dump(error_dict, file)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print('Model saved at', path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print('Model loaded from', path)
        return self.model
    
    def save_hist(self, train_hist, test_hist, times, path_to_root=""):
        with open("{}train_hist.pkl".format(path_to_root), "wb") as file:
            pickle.dump(train_hist, file)
        with open("{}test_hist.pkl".format(path_to_root), "wb") as file:
            pickle.dump(test_hist, file)
        with open("{}times.pkl".format(path_to_root), "wb") as file:
            pickle.dump(times, file)
    
    def get_model(self):
        return self.model
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_criterion(self):
        return self.criterion
    
    def get_device(self):
        return self.device
    
    def set_model(self, model):
        self.model = model
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def _plot_histogram(self, ax, data, title, bins, range):
        ax.set_xlim(range)
        counts, bin_edges = np.histogram(data, bins=bins, range=range)
        ax.set_title(title)
        ax.set_xlabel("Mean Error")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        err = np.sqrt(counts)
        ax.errorbar(bin_centres, counts, yerr=err, fmt='o')

    def plot_histograms(self, error_x, error_y, error_mag, error_ang):
        fig, ax = plt.subplots(2, 2, figsize=(30, 10))
        self._plot_histogram(ax[0, 0], error_x, "Error X", 100, (0, 0.5))
        self._plot_histogram(ax[0, 1], error_y, "Error Y", 100, (0, 0.5))
        self._plot_histogram(ax[1, 0], error_mag, "Error Magnitude", 100, (0, 0.5))
        self._plot_histogram(ax[1, 1], error_ang, "Error Angle", 180, (0, np.pi))
        plt.show()

    def getErrorHistograms(self, test_loader, figureOfMerit="all"):
        error_x, error_y, error_mag, error_ang = self.get_errors(test_loader, figureOfMerit)
        self.plot_histograms(error_x, error_y, error_mag, error_ang)
        print("Error histograms for '{}': ".format(figureOfMerit))
        print("Mean error x:", np.mean(error_x))
        print("Mean error y:", np.mean(error_y))
        print("Mean error magnitude:", np.mean(error_mag))
        print("Mean error angle:", np.mean(error_ang))
        print("Mean error x std:", np.std(error_x))
        print("Mean error y std:", np.std(error_y))
        print("Mean error magnitude std:", np.std(error_mag))
        print("Mean error angle std:", np.std(error_ang))