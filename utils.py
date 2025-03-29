
"""
Python file containing utility functions for the models.
A utility function is a function that is useful in multiple contexts.
"""

# Import libraries
import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset

# Import local
from config import *


### CUSTOM FUNCTIONS ###

# D97 normalization function
def norm_d97(dose, ptvs):
    
    # Convert to numpy if tensor
    using_torch = isinstance(dose, torch.Tensor)
    if using_torch:
        dtype = dose.dtype
        device = dose.device
        dose = dose.cpu().detach().numpy()
    if isinstance(ptvs, torch.Tensor):
        ptvs = ptvs.cpu().detach().numpy()

    # Get PTV high dose and mask
    dose_ptvhigh = ptvs.max()
    dose_ptvhigh_mask = (ptvs == dose_ptvhigh).any(axis=-4, keepdims=True)

    # Normalize using D97 of PTV_High
    norm_scale = dose_ptvhigh / (np.percentile(dose[dose_ptvhigh_mask], 3) + 1e-5)
    dose = dose * norm_scale

    # Clip dose to 0 and 1.2 * PTV_High
    dose = np.clip(dose, 0, dose_ptvhigh * 1.2)

    # Convert to tensor if using torch
    if using_torch:
        dose = torch.tensor(dose, dtype=dtype, device=device)

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

# Data augmentor
def augment_data_3d(*inputs, targets=None, affine=True, noise=True, block_mask=False):
    """
    Apply the same augmentation to all inputs and targets. Avoid noise and block mask for targets.
    """

    # Convert inputs to list
    inputs = list(inputs)
    if targets is None:
        targets = []
    elif isinstance(targets, torch.Tensor):
        targets = [targets]

    # Get constants
    device = inputs[0].device
    D, H, W = inputs[0].shape[-3:]

    # Affine transformation
    if affine:
        """Apply the same affine transformation to all inputs and targets."""

        # Sample angles, scales, and translations
        rot_max = 10 * 3.1415 / 180
        scale_max = 0.1
        shift_max = 0.05
        angles = (2 * torch.rand(3, device=device) - 1) * rot_max        # xyz angles
        scales = (2 * torch.rand(3, device=device) - 1) * scale_max + 1  # xyz scales
        shifts = (2 * torch.rand(3, device=device) - 1) * shift_max      # xyz shifts

        # Rotation matrices (Rx, Ry, Rz)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx,  cx]
        ], device=device)
        Ry = torch.tensor([
            [cy, 0, sy],
            [0,  1, 0],
            [-sy,0, cy]
        ], device=device)
        Rz = torch.tensor([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [0,   0,  1]
        ], device=device)
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)

        # Scale matrix
        S = torch.diag(scales)

        # Full affine transformation matrix and grid
        theta = torch.eye(3, 4, device=device)
        theta[:, :3] = torch.matmul(R, S)
        theta[:, 3] = shifts
        grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, D, H, W), align_corners=False)

        # Create warp function
        def warp(x):
            # Add batch dimension
            x = x.unsqueeze(0)
            # Check if input dtype
            if x.dtype in [torch.float32, torch.float64]:
                # Apply bilinear interpolation for floats
                x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
            else:
                # Apply nearest-neighbor interpolation for bools
                x = F.grid_sample(x.float(), grid, mode='nearest', align_corners=False).to(x.dtype)
            # Remove batch dimension
            x = x.squeeze(0)
            # Return warped tensor
            return x 
        
        # Apply affine transformation to all inputs and targets
        inputs = [warp(x) for x in inputs]
        targets = [warp(x) for x in targets]

    # Noise
    if noise:
        """Apply different noise to each input."""
        inputs = [
            x + torch.randn_like(x, device=device) * 0.1 * x.std()  # Add noise to tensor
            if x.dtype in [torch.float32, torch.float64] else x     # if tensor is float
            for x in inputs                                         # for each input
        ]

    # Block mask
    if block_mask:
        """Apply the same block mask to all inputs."""
        mask = torch.ones_like(inputs[0], device=device)       # Initialize mask
        mask = block_mask_3d(mask, block_size=8, p=0.2)        # Apply random block masking
        inputs = [(x * mask).astype(x.dtype) for x in inputs]  # Apply mask to inputs

    # Return augmented data
    return inputs + targets

### SAVING AND LOADING ###

# Get savename function
def get_savename(dataID, modelID, **kwargs):
    """
    Get savename for a given dataset and model.
    """

    # Initialize savename
    savename = f'model_{dataID}_{modelID}'

    # Add kwargs to savename
    kwargs_sorted = sorted(kwargs.items())
    for key, value in kwargs_sorted:
        # if value:
        #     savename += f'_{key}={value}'
        savename += f'_{key}={value}'

    # Return savename
    return savename

# Initialize dataset function
def initialize_datasets(dataID, validation_set=False):

    # Import dataset
    from dataset import GDPDataset

    # Get number of samples
    n_samples = len(GDPDataset(treatment=dataID, validation_set=validation_set))

    # Get indices for split
    test_size = int(0.2 * n_samples)
    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(42))
    indices_test = indices[:test_size]
    indices_val  = indices[test_size:2*test_size]
    indices_train = indices[2*test_size:]

    # Create subsets
    dataset_train = Subset(GDPDataset(treatment=dataID, validation_set=validation_set), indices_train)
    dataset_val = Subset(GDPDataset(treatment=dataID, validation_set=validation_set), indices_val)
    dataset_test = Subset(GDPDataset(treatment=dataID, validation_set=validation_set), indices_test)

    # dataset_val, dataset_test, dataset_train = torch.utils.data.random_split(
    #     dataset,
    #     [test_size, test_size, len(dataset) - 2*test_size],
    #     generator=torch.Generator().manual_seed(42),  # Set seed for reproducibility
    # )

    # Return dataset
    return dataset_val, dataset_test, dataset_train

# Initialize model
def initialize_model(modelID, in_channels, **kwargs):
    from model import DosePredictionModel
    model = DosePredictionModel(modelID, in_channels, **kwargs)
    return model

# Save checkpoint
def save_checkpoint(checkpoint_path, model, datasets, optimizer, metadata):
    torch.save(
        {
            'model_config': model.get_config(),
            'model_state_dict': model.state_dict(),
            'data_config': datasets[0].dataset.get_config(),
            'data_indices': {
                'train': datasets[0].indices,
                'val': datasets[1].indices,
                'test': datasets[2].indices,
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata,
        }, 
        checkpoint_path
    )

# Load checkpoint
def load_checkpoint(checkpoint_path, load_best=False):
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load model from checkpoint
    from model import DosePredictionModel
    if load_best:
        model_state_dict = checkpoint['metadata']['model_state_dict_best']
        model = DosePredictionModel.from_checkpoint(checkpoint_path, model_state_dict=model_state_dict)
    else:
        model = DosePredictionModel.from_checkpoint(checkpoint_path)

    # Load datasets from checkpoint
    from dataset import GDPDataset
    data_config = checkpoint['data_config']
    data_indices = checkpoint['data_indices']
    dataset_train = Subset(GDPDataset(**{**data_config, 'augment': True}), data_indices['train'])
    dataset_val = Subset(GDPDataset(**{**data_config, 'augment': True}), data_indices['val'])
    dataset_test = Subset(GDPDataset(**{**data_config, 'augment': False}), data_indices['test'])
    datasets = (dataset_train, dataset_val, dataset_test)

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load metadata
    metadata = checkpoint['metadata']

    # Return outputs
    return model, datasets, optimizer, metadata


### MEASURE CPU MEMORY ###

# Define function to measure CPU memory
def estimate_memory_usage(model, *inputs, print_stats=True):
    """
    Estimate total memory used by model parameters, gradients, activations, and optimizer states.
    
    WARNING:
    The CPU function should only be run once per script execution. Running it multiple times in a  
    row may produce incorrect results due to PyTorch's memory caching, delayed garbage collection, 
    and OS-level memory fragmentation. For accurate measurements, restart the process before 
    each run.
    """

    # Print status
    if print_stats:
        print("Estimating Memory consumption...")
        t_start = time.time()

    # Import libraries
    import psutil

    # Measure memory before execution
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss  # Total RAM usage before forward pass

    # Forward and backward pass
    y = model(*inputs)
    loss = y.sum()
    loss.backward()

    # Measure memory after execution
    mem_after = process.memory_info().rss  # Total RAM usage after backward pass

    # Compute memory usage for model parameters
    dtype = next(model.parameters()).dtype
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = torch.finfo(dtype).bits // 8  # 4 bytes for float32, 2 bytes for float16
    mem_params = n_params * bytes_per_param         # Memory for parameters
    mem_gradients = mem_params                      # Gradients require same amount of memory
    mem_optimizer = 2 * mem_params                  # Adam needs ~2x parameter memory

    # Total estimated memory usage
    mem_total = mem_after - mem_before
    mem_activations = mem_total - (mem_params + mem_gradients + mem_optimizer)

    # Print stats
    if print_stats:
        print(f"Estimated Memory Usage ({time.time() - t_start:.3f} seconds to compute):")
        # Loop over memory components
        for key, value in {
            "Parameters": mem_params,
            "Gradients": mem_gradients,
            "Optimizer": mem_optimizer,
            "Activations": mem_activations,
            "Total": mem_total,
        }.items():
            # Dynamically find the appropriate unit
            scale = 1
            for unit in ["bytes", "KB", "MB", "GB", "TB"]:
                if value / scale < 1024:
                    break
                scale *= 1024
            print(f"  - {key}: {value / scale:.3f} {unit}")

    # Return total memory
    return mem_total

