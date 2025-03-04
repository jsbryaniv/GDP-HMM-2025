
"""
Python file containing utility functions for the models.
A utility function is a function that is useful in multiple contexts.
"""

# Import libraries
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)
ROOT_DIR = config['PATH_OUTPUT']


### CUSTOM FUNCTIONS ###

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


### RESHAPING DATA ###

# Define function to resize 3D image
def resize_image_3d(image, target_shape, fill_value=0):
    """
    Resize a 3D image to the target shape while maintaining aspect ratio.
    Automatically handles boolean data by using nearest-neighbor interpolation.

    Args:
        image (torch.Tensor): 3D image tensor of shape (C, D, H, W)
        target_shape (tuple): Desired shape (D_target, H_target, W_target)

    Returns:
        torch.Tensor: Resized and padded image.
        dict: Resize and padding parameters for reversal.
    """

    # Get image shape
    _, D, H, W = image.shape
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
        image.float().unsqueeze(0), 
        size=new_size, 
        mode=interp_mode, 
        align_corners=interp_align
    ).squeeze(0)
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
        image (torch.Tensor): Padded image tensor of shape (C, D_target, H_target, W_target)
        transform_params (dict): Parameters from forward transform

    Returns:
        torch.Tensor: Restored original-sized image.
    """

    # Get variables
    _, D_target, H_target, W_target = image.shape
    D, H, W = transform_params["original_shape"]
    pad = transform_params["pad"]

    # Remove padding
    unpadded_image = image[:, pad[4]:D_target-pad[5], pad[2]:H_target-pad[3], pad[0]:W_target-pad[1]]

    # Determine interpolation mode
    is_boolean = image.dtype == torch.bool
    interp_mode = 'nearest' if is_boolean else 'trilinear'
    interp_align = False if interp_mode == 'trilinear' else None

    # Resize back to original shape
    restored_image = F.interpolate(
        unpadded_image.float().unsqueeze(0), 
        size=(D, H, W), 
        mode=interp_mode, 
        align_corners=interp_align
    ).squeeze(0)
    if is_boolean:
        restored_image = restored_image.bool()

    # Return output
    return restored_image


### DATA AND MODEL SAVING AND LOADING ###

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
        if value:
            savename += f'_{key}={value}'

    # Return savename
    return savename

# Initialize dataset function
def initialize_dataset(dataID, **kwargs):

    # Import dataset
    from dataset import GDPDataset

    # Set constants
    in_channels = 43  # 1 CT + 1 Beam + 3 PTVs + 37 OARs + 1 Body
    out_channels = 1
    shape = 128

    # Load dataset
    if dataID.lower() == 'all':
        # All datasets
        treatment = 'All'
    elif dataID.lower() == 'han':
        # Head and Neck dataset
        in_channels = 36  # 1 CT + 1 Beam + 3 PTVs + 30 OARs + 1 Body
        treatment = 'HaN'
    elif dataID.lower() == 'lung':
        # Head and Neck dataset
        in_channels = 16  # 1 CT + 1 Beam + 3 PTVs + 10 OARs + 1 Body
        treatment = 'Lung'
    else:
        raise ValueError(f'Dataset {dataID} not recognized.')

    # Create dataset
    dataset = GDPDataset(
        treatment=treatment,
        **{'shape': shape, **kwargs},
    )

    # Collect metadata
    metadata = {
        'dataID': dataID,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'shape': shape,
        **kwargs,
    }

    # Return dataset
    return dataset, metadata

# Initialize model
def initialize_model(modelID, in_channels, out_channels, **kwargs):

    # Identify model
    if modelID.lower() == 'unet':
        # Unet3D model
        from models.unet import Unet3D
        model = Unet3D(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs,
        )
    elif modelID.lower() == 'vit':
        # Vision Transformer model
        from models.vit import ViT3D
        model = ViT3D(
            in_channels=in_channels, 
            out_channels=out_channels,
            **{'shape': 128, **kwargs},
        )
    elif modelID.lower() == 'crossattnae':
        # Cross Attention Autoencoder model
        from models.crossattnae import CrossAttnAEModel
        model = CrossAttnAEModel(
            in_channels=4,
            out_channels=1,
            n_cross_channels_list=[1, 4, in_channels-5],  # ct,( beam, ptvs), (oars, body)
            **kwargs,
        )
    else:
        raise ValueError(f'Model {modelID} not recognized.')

    # Return model
    return model

# Load trained model and dataset
def load_model_and_datasets(savename):

    # Load metadata
    with open(os.path.join(ROOT_DIR, f'{savename}.json'), 'r') as f:
        metadata = json.load(f)

    # Extract metadata
    dataID = metadata['dataID']
    modelID = metadata['modelID']
    data_kwargs = metadata['data_kwargs']
    model_kwargs = metadata['model_kwargs']
    train_kwargs = metadata['train_kwargs']
    indices_train = metadata['indices_train']
    indices_val = metadata['indices_val']
    indices_test = metadata['indices_test']
    training_statistics = metadata['training_statistics']
    
    # Load dataset
    dataset, data_metadata = initialize_dataset(dataID, **data_kwargs)
    in_channels = data_metadata['in_channels']
    out_channels = data_metadata['out_channels']
    # Split into train, validation, and test sets
    dataset_train = Subset(dataset, indices_train)
    dataset_val = Subset(dataset, indices_val)
    dataset_test = Subset(dataset, indices_test)
    # Package into tuple
    datasets = (dataset_train, dataset_val, dataset_test)

    # Load model
    model = initialize_model(modelID, in_channels, out_channels, **model_kwargs)
    # Load weights from file
    model_state_dict = torch.load(os.path.join(ROOT_DIR, f'{savename}.pth'), weights_only=True)
    model.load_state_dict(model_state_dict)

    # Return outputs
    return model, datasets, metadata


### MEASURE CPU MEMORY ###

# Define function to measure CPU memory
def estimate_memory_usage(model, x, print_stats=True, device=None):
    """
    Estimate total memory used by model parameters, gradients, activations, and optimizer states.
    
    WARNING:
    The CPU function should only be run once per script execution. Running it multiple times in a  
    row may produce incorrect results due to PyTorch's memory caching, delayed garbage collection, 
    and OS-level memory fragmentation. For accurate measurements, restart the process before 
    each run.
    """

    # Import libraries
    import psutil
        
    # Check which method to use
    if device is None or device == 'cpu':
        """
        Calculate memory usage on CPU using psutil.
        """

        # Move to CPU
        device = torch.device("cpu")  # Ensure execution on CPU
        model = model.to(device)
        x = x.to(device)

        # Measure memory before execution
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss  # Total RAM usage before forward pass

        # Forward and backward pass
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Measure memory after execution
        mem_after = process.memory_info().rss  # Total RAM usage after backward pass

    else:
        """
        Calculate memory usage on GPU using PyTorch.
        """

        # Move to GPU
        device = torch.device("cuda")  # Ensure execution on GPU
        model = model.to(device)
        x = x.to(device)

        # Measure memory before execution
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.max_memory_allocated(device)

        # Forward and backward pass
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Measure memory after execution
        mem_after = torch.cuda.max_memory_allocated(device)

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
        print(f"Model has {n_params:,} parameters")
        print(f"Estimated Memory Usage:")
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

