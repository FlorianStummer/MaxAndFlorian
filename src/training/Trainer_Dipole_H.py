
import torch
import time
from tqdm import tqdm

class Trainer_Dipole_H:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def run_epoch(self, train_loader, test_loader, epochs):
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
        print('Total time:', time.time() - time_start)
        return train_hist, test_hist
    
    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
        return test_loss
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print('Model saved at', path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print('Model loaded from', path)
        return self.model
    
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

