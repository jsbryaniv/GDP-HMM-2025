
"""
Python file containing utility functions for the models.
A utility function is a function that is useful in multiple contexts.
"""

# Import libraries
import os
import torch
import numpy as np
import torch.nn.functional as F



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



### CUSTOM LOSS FUNCTIONS ###

# Define competition loss function
def competition_loss(prediction, ref_dose, body):

    # Convert to numpy
    prediction = prediction.cpu().detach().numpy()
    ref_dose = ref_dose.cpu().detach().numpy()
    body = body.cpu().detach().numpy()

    # the mask include the body AND the region where the dose/prediction is higher than 5Gy
    isodose_5Gy_mask = ((ref_dose > 5) | (prediction > 5)) & (body > 0) 

    diff = ref_dose - prediction

    error = np.mean(np.abs(diff)[isodose_5Gy_mask > 0])

    return error

# Define Earth Mover's Distance (EMD) loss function
def emd_loss(pred_cdf, target_cdf, structures, dx=1):
    """
    Calulate the Earth Mover's Distance (EMD) loss between two CDFs.
    EMD loss simplifies to the L1 distance between the two CDFs.
    """
    
    # Calculate EMD
    loss = ((pred_cdf - target_cdf) * dx).abs().mean(dim=-1)

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
            if torch.sum(structure) == 0:
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
            loss += emd_loss(dvh_pred, dvh_target, structures, dx=max_dose_i/bins)

    # Return loss
    return loss



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

