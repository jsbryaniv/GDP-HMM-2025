
# Import libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# Plot ROC curve
def plot_roc(fpr, tpr, auc=None):
    """
    Plot ROC curve.
    Args:
        fpr (np.array): False positive rate.
        tpr (np.array): True positive rate.
        auc (float): Area under the curve.
    """

    # Set up figure
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    # Plot ROC curve
    if auc is None:
        label = 'ROC'
    else:
        label = f'ROC (AUC={auc:.5f})'
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.plot(fpr, tpr, label=label)

    # Finalize plot
    ax.legend()
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

# Plot results of a single slice
def plot_prediction(scan, mask, pred, z=None):
    """
    Plot results of a single slice.
    Args:
        scan (array): Input scan.
        mask (array): Ground truth mask.
        pred (array): Predicted mask.
        z (int): Slice to plot.
    """

    # Convert to numpy
    if isinstance(scan, torch.Tensor):
        scan = scan.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()

    # Set up figure
    fig, ax = plt.subplots(1, 3)
    plt.ion()
    plt.show()

    # Find z slice where the mask is present
    if z is None:
        if mask.any():
            z = np.where(mask[0].any(axis=(0, 1)))[0][0]
        else:
            z = mask.shape[-1] // 2

    # Plot data
    ax[0].set_title('Scan')
    ax[0].imshow(scan[0, 0, :, :, z], cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].imshow(mask[0, :, :, z], cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].imshow(pred[0, :, :, z] > .5, cmap='gray')

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

