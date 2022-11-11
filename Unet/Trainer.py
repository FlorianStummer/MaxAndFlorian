from copy import deepcopy
import time
import sys
import torch

class Trainer:
    def __init__(self, optimizer, optimizer_parameters, loss_fun, device, model, train_dataloader, val_dataloader, num_epochs, 
    priority_mask_mode='field_strength',min_priority=0.1):
        self.optimizer_parameters = optimizer_parameters
        self.model = model

        self.optimizer = optimizer(params=self.model.parameters(), **self.optimizer_parameters)
        self.loss_fun = loss_fun
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs

        if not priority_mask_mode in ['field_strength', 'material']:
            raise ValueError("unsupported priority mask mode '" + str(priority_mask_mode) + "'. has to be 'field_strength' or 'material'")
        self.priority_mask_mode = priority_mask_mode
        self.min_priority = min_priority


    def run_epoch(self, train, dataloader):
        loss_list = []
        loss_fun = self.loss_fun()

        if train:
            self.model.train()
        else:
            self.model.eval()

        self.model.to(self.device)

        for x, y in dataloader:

            x = x.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)

            pred = self.model(x)

            if self.priority_mask_mode == 'field_strength':
                y_swapped = torch.stack([y[:,1,:,:],y[:,0,:,:]])
                field_strengths =  torch.sqrt(y*y + y_swapped*y_swapped)
                priority_mask = torch.max(torch.ones_like(y)*self.min_priority, field_strengths)
            elif self.priority_mask_mode == 'material':
                material_stack = torch.stack([x[:,0,:,:],x[:,0,:,:]],1)
                priority_mask = torch.max(torch.ones_like(y)*self.min_priority,material_stack)

            # the loss that is used for training
            loss_with_prio = loss_fun(pred*priority_mask, y*priority_mask)

            # the loss that we use to compare models
            loss = loss_fun(pred, y)            

            loss_list.append(loss.item())
            if train:
                self.optimizer.zero_grad()
                loss_with_prio.backward()
                self.optimizer.step()

        return torch.tensor(loss_list).mean()

    def train(self):
        train_hist = []
        val_hist = []

        for i in range(1, self.num_epochs + 1):
            try:
                train_loss = self.run_epoch(train=True, dataloader=self.train_dataloader).item()
                with torch.no_grad():
                    val_loss = self.run_epoch(train=False, dataloader=self.val_dataloader).item()


                train_hist.append(train_loss)
                val_hist.append(val_loss)

                prec = 6

                now = time.gmtime()
                now_string = str(now.tm_mday)+":"+str(now.tm_hour)+":"+str(now.tm_min)+":"+str(now.tm_sec)

                print("\t Epoch " + str(i)
                    + ":\t Train_loss: " + (str(train_loss)[0:prec])
                    + "\t Val_loss: " + (str(val_loss)[0:prec])
                    + "\t" + now_string)
                sys.stdout.flush()
            except KeyboardInterrupt:
                return train_hist, val_hist

        return train_hist, val_hist
    