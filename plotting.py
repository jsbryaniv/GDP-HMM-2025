
# Import libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# Plot results of a single slice
def plot_prediction(scan, target, prediction, z=None):
    """
    Plot results of a single slice.
    Args:
        scan (array): Input scan.
        target (array): Ground truth.
        prediction (array): Predicted.
        z (int): Slice to plot.
    """

    # Convert to numpy
    if isinstance(scan, torch.Tensor):
        scan = scan.cpu().detach().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()

    # Set up figure
    fig, ax = plt.subplots(1, 3)
    plt.ion()
    plt.show()

    # Find middle slice
    if z is None:
        z = target.shape[-3] // 2

    # Plot data
    ax[0].set_title('Input')
    ax[0].imshow(scan[0, 0, z, :, :], cmap='gray')
    ax[1].set_title('Target')
    ax[1].imshow(target[0, 0, z, :, :], cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].imshow(prediction[0, 0, z, :, :] > .5, cmap='gray')

    # Finalize plot
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

# Plot training and validation losses
def plot_losses(losses_train, losses_val):
    """
    Plot training and validation losses.
    Args:
        losses_train (list): Training losses.
        losses_val (list): Validation losses.
    """

    # Set up figure
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    # Plot losses
    ax.set_title('Losses')
    ax.plot(losses_train, label='Train')
    ax.plot(losses_val, label='Validation')

    # Finalize plot
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

