
"""
Python file containing utility functions for the models.
A utility function is a function that is useful in multiple contexts.
"""

# Import libraries
import torch
import numpy as np
import torch.nn.functional as F


### CUSTOM FUNCTIONS ###

# D97 normalization function
def norm_d97(dose, ptvs):
    
    # Convert to numpy if tensor
    dose_np = dose.cpu().detach().numpy() if isinstance(dose, torch.Tensor) else dose
    ptvs_np = ptvs.cpu().detach().numpy() if isinstance(ptvs, torch.Tensor) else ptvs

    # Get PTV high dose and mask
    ptvhigh_value = ptvs_np.max()
    ptvhigh_mask = (ptvs_np == ptvhigh_value).any(axis=-4, keepdims=True)

    # Get norm scale using D97 of PTV_High
    norm_scale = ptvhigh_value / (np.percentile(dose_np[ptvhigh_mask], 3) + 1e-5)

    # Normalize dose
    dose *= norm_scale

    # Clip dose to 0 and 1.2 * PTV_High
    if isinstance(dose, torch.Tensor):
        dose = torch.clamp(dose, 0, ptvhigh_value * 1.2)
    else:
        dose = np.clip(dose, 0, ptvhigh_value * 1.2)

    # Return normalized dose
    return dose

# Randomly mask 3D volume
def block_mask_3d(volume, block_size=8, p=0.2):
    B, C, D, H, W = volume.shape
    mask = torch.rand(B, 1, D // block_size, H // block_size, W // block_size, device=volume.device) > p
    mask = F.interpolate(mask.float(), size=(D, H, W), mode='nearest')
    return volume * mask

# Get DVH function
def get_dvh(dose, structures, bins=100, max_dose=None):

    # If tensor, convert to numpy
    if isinstance(dose, torch.Tensor):
        dose = dose.cpu().detach().numpy()
    if isinstance(structures, torch.Tensor):
        structures = structures.cpu().detach().numpy()

    # Check inputs
    if max_dose is None:
        max_dose = np.max(dose)
    
    # Get dose range
    bins = np.linspace(0, max_dose, bins)
    dvh_bin = (bins[:-1] + bins[1:]) / 2
    dvh_bin = np.append(bins, 2*bins[-1]-bins[-2])  # Add final point to ensure 0% above max dose
    dvh_bin = np.append(0, bins)                    # Add initial point to ensure 100% at 0 dose

    # Initialize dvhs
    dvh_val = []

    # Loop over structures
    for i in range(structures.shape[1]):
        structure = structures[0, i]
        if np.sum(structure) == 0:
            continue
        
        # Get dvh
        hist, _ = np.histogram(dose[0, 0, structure.astype(bool)], bins=bins)  # Get histogram
        dvh = np.cumsum(hist[::-1])[::-1]                                      # Get cumulative histogram (reverse order)
        dvh = 100 * dvh / (dvh[0] + 1e-8)                                      # Normalize to 100%
        dvh = np.append(dvh, 0)                                                # Add final point to ensure 0% above max dose
        dvh = np.append(100, dvh)                                              # Add initial point to ensure 100% at 0 dose

        # Append to list
        dvh_val.append(dvh)

    # Convert to numpy array
    dvh_val = np.array(dvh_val)

    # Return list
    return dvh_val, dvh_bin


### DATA MANIPULATION ###

# Define function to resize 3D image
def resize_image_3d(image, target_shape, fill_value=0):
    """
    Resize a 3D image to the target shape while maintaining aspect ratio.
    Automatically handles boolean data by using nearest-neighbor interpolation.

    Args:
        image (torch.Tensor): 3D image tensor of shape (B, C, D, H, W)
        target_shape (tuple): Desired shape (D_target, H_target, W_target)

    Returns:
        torch.Tensor: Resized and padded image.
        dict: Resize and padding parameters for reversal.
    """

    # Get image shape
    _, _, D, H, W = image.shape
    D_target, H_target, W_target = target_shape

    # Determine interpolation mode based on dtype
    is_boolean = image.dtype == torch.bool
    interp_mode = 'nearest' if is_boolean else 'trilinear'
    interp_align = False if interp_mode == 'trilinear' else None

    # Compute uniform scale factor
    scale = min(D_target / D, H_target / H, W_target / W)
    new_size = (int(D * scale), int(H * scale), int(W * scale))

    # Resize the image
    resized_image = F.interpolate(
        image.float(), 
        size=new_size, 
        mode=interp_mode, 
        align_corners=interp_align
    )
    if is_boolean:
        resized_image = resized_image.bool()

    # Compute padding
    pad_d = (D_target - new_size[0]) / 2
    pad_h = (H_target - new_size[1]) / 2
    pad_w = (W_target - new_size[2]) / 2
    pad = (int(pad_w), int(pad_w + 0.5), int(pad_h), int(pad_h + 0.5), int(pad_d), int(pad_d + 0.5))

    # Apply padding with the correct default value
    padded_image = F.pad(resized_image, pad, mode='constant', value=fill_value)

    # Store parameters for inverse transformation
    transform_params = {
        "original_shape": (D, H, W),
        "scale": scale,
        "pad": pad
    }

    # Return output
    return padded_image, transform_params

# Define function to reverse resize 3D image
def reverse_resize_3d(image, transform_params):
    """
    Reverse the resize and padding operation.

    Args:
        image (torch.Tensor): Padded image tensor of shape (B, C, D_target, H_target, W_target)
        transform_params (dict): Parameters from forward transform

    Returns:
        torch.Tensor: Restored original-sized image.
    """

    # Get variables
    _, _, D_target, H_target, W_target = image.shape
    D, H, W = transform_params["original_shape"]
    pad = transform_params["pad"]

    # Remove padding
    unpadded_image = image[:, :, pad[4]:D_target-pad[5], pad[2]:H_target-pad[3], pad[0]:W_target-pad[1]]

    # Determine interpolation mode
    is_boolean = image.dtype == torch.bool
    interp_mode = 'nearest' if is_boolean else 'trilinear'
    interp_align = False if interp_mode == 'trilinear' else None

    # Resize back to original shape
    restored_image = F.interpolate(
        unpadded_image.float(), 
        size=(D, H, W), 
        mode=interp_mode, 
        align_corners=interp_align
    )
    if is_boolean:
        restored_image = restored_image.bool()

    # Return output
    return restored_image

