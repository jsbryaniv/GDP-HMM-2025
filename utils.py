
"""
Python file containing utility functions for the models.
A utility function is a function that is useful in multiple contexts.
"""

# Import libraries
import os
import torch
import psutil
# import pytorch_msssim

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



### CUSTOM LOSS FUNCTIONS ###

# # Define 3D SSIM loss
# def ssim3d_loss(pred, target):
#     """Computes 3D SSIM efficiently by treating depth slices as batch elements."""
#     B, C, D, H, W = pred.shape
#     pred_2d = pred.reshape(B * D, C, H, W)  # Flatten depth into batch
#     target_2d = target.reshape(B * D, C, H, W)
#     return pytorch_msssim.ssim(pred_2d, target_2d, data_range=1.0)

