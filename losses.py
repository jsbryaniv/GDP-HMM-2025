
# Import libraries
import os
import math
import torch
import numpy as np
import torch.nn.functional as F


# Define competition loss function
def competition_loss(pred, target, body):
    """
    This loss replicates the error calculation provided by the competition using torch tensors.
    ```
    # the mask include the body AND the region where the dose/prediction is higher than 5Gy
    isodose_5Gy_mask = ((ref_dose > 5) | (prediction > 5)) & (body > 0) 
    diff = ref_dose - prediction
    error = np.mean(np.abs(diff)[isodose_5Gy_mask > 0])
    ```
    """

    # Get mask
    mask = ((target > 5) | (pred > 5)) & (body > 0)

    # Calculate loss
    loss = (target[mask] - pred[mask]).abs().mean()

    # # Plot error
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 4)
    # plt.ion()
    # plt.show()
    # z_slice = pred.shape[2] // 2
    # ax[0].set_title('Target')
    # ax[0].imshow(target[0, 0, z_slice].cpu().detach().numpy())
    # ax[1].set_title('Prediction')
    # ax[1].imshow(pred[0, 0, z_slice].cpu().detach().numpy())
    # ax[2].set_title('Mask')
    # ax[2].imshow(mask[0, 0, z_slice].cpu().detach().numpy())
    # ax[3].set_title('Error')
    # ax[3].imshow((target - pred)[0, 0, z_slice].cpu().detach().numpy())
    # plt.tight_layout()
    # plt.pause(.1)
    # plt.savefig('_image.png')

    # Return loss
    return loss

# Define Earth Mover's Distance (EMD) loss function
def emd_loss(pred_cdf, target_cdf, dx=1):
    """
    Calulate the Earth Mover's Distance (EMD) loss between two CDFs.
    EMD loss simplifies to the L1 distance between the two CDFs.
    """
    
    # Calculate EMD
    loss = ((pred_cdf - target_cdf) * dx).abs().mean()

    # Return loss
    return loss

# Define DVH loss function
def dvh_loss(pred_dose, target_dose, structures, max_dose=None, bins=100):
    """
    Calulate the dose volume histogram (DVH) loss between the predicted dose and the target dose given the structures.
    """

    # Initialize loss
    loss = 0

    # Loop over batch
    for B in range(pred_dose.shape[0]):
        
        # Loop over structures
        for i in range(structures.shape[1]):
            structure = structures[B, i]
            if torch.sum(structure) == 0:  # Skip empty structures
                continue

            # Get dose values
            val_pred = pred_dose[B, 0, structure.bool()]
            val_target = target_dose[B, 0, structure.bool()]

            # Get max dose for binning
            max_dose_i = max_dose
            if max_dose_i is None:
                with torch.no_grad():
                    max_dose_i = max(val_pred.max().item(), val_target.max().item())

            # Get DVH
            dvh_pred = torch.histc(val_pred, bins=bins, min=0, max=max_dose_i)      # Histogram prediction
            dvh_target = torch.histc(val_target, bins=bins, min=0, max=max_dose_i)  # Histogram target
            dvh_pred = torch.cumsum(dvh_pred.flip(0), 0).flip(0) + 1e-8             # Cumulative histogram prediction
            dvh_target = torch.cumsum(dvh_target.flip(0), 0).flip(0) + 1e-8         # Cumulative histogram target
            dvh_pred = dvh_pred / dvh_pred[0]                                       # Normalize prediction DVH to 1
            dvh_target = dvh_target / dvh_target[0]                                 # Normalize target DVH to 1

            # Calculate loss
            loss += emd_loss(dvh_pred, dvh_target, dx=max_dose_i/bins)

    # Return loss
    return loss

# Define loss for dose to structures
def structure_dose_loss(pred_dose, target_dose, structures):
    """
    Calculate the dose to structures loss between the predicted dose and the target dose given the structures.
    """

    # Initialize loss
    loss = 0

    # Loop over batch
    for B in range(pred_dose.shape[0]):
        
        # Loop over structures
        for i in range(structures.shape[1]):
            structure = structures[B, i]
            if torch.sum(structure) == 0:  # Skip empty structures
                continue

            # Get dose values
            val_pred = pred_dose[B, 0, structure.bool()].mean()
            val_target = target_dose[B, 0, structure.bool()].mean()

            # Calculate loss
            loss += ((val_pred - val_target) / (val_target.abs() + 1e-5)) ** 2

    # Normalize loss
    loss = loss / math.prod(pred_dose.shape[:2])

    # Return loss
    return loss

# Define gradient loss function
def gradient_loss(pred, target):
    """
    Compute gradient difference loss between prediction and target in 3D.
    Encourages sharp edges by aligning gradients.
    """

    # Define gradient function
    def compute_gradients(x):
        dx = x[..., 1:, :, :] - x[..., :-1, :, :]
        dy = x[..., :, 1:, :] - x[..., :, :-1, :]
        dz = x[..., :, :, 1:] - x[..., :, :, :-1]
        return dx, dy, dz

    # Compute gradients
    dx_pred, dy_pred, dz_pred = compute_gradients(pred)
    dx_targ, dy_targ, dz_targ = compute_gradients(target)

    # Compute loss with MAE not MSE
    loss = (
        (dx_pred - dx_targ).abs().mean()    # Use MAE NOT MSE
        + (dy_pred - dy_targ).abs().mean()
        + (dz_pred - dz_targ).abs().mean()
    )
    
    # Return loss
    return loss


