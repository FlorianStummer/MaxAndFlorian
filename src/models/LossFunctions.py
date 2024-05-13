import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics as tm
import matplotlib.pyplot as plt

def prioritized_loss(output, target, weights, criterion=nn.MSELoss(reduction='none')):
    """
    Compute a prioritized loss where individual values are weighted by priority.

    :param output: torch.Tensor: Predicted values. (b, c, h, w)
    :type output: torch.Tensor
    :param target: torch.Tensor: Ground truth values. (b, c, h, w)
    :type target: torch.Tensor
    :param weights: torch.Tensor: Weights for each value. (b, c, h, w)
    :type weights: torch.Tensor
    :param criterion: torch.nn.Module: Loss function. Defaults to nn.MSELoss().
    :type criterion: torch.nn.Module
    :return: torch.Tensor: Prioritized loss. (1,)
    """
    loss = criterion(output, target)  # Compute the base loss
    weighted_loss = loss * weights    # Apply weights
    return weighted_loss.mean()       # Return mean of mean loss
    # return loss.mean()

if __name__ == "__main__":
    # Create a random dataset
    data = torch.randn(10, 5, 32, 32)
    target = torch.randn(10, 2, 32, 32)
    weights = torch.ones(10, 2, 32, 32)
    
    mask = torch.ones(10, 2, 5, 5) * 0.1
    weights[:, :, 10:15, 10:15] = mask

    # Create a model
    model = nn.Sequential(
        nn.Conv2d(5, 10, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(10, 2, 3, padding=1),
    )

    # Compute the loss
    loss = prioritized_loss(model(data), target, weights)
    loss.backward()
    print(loss)
    # tensor(1.0000, grad_fn=<MeanBackward0>)