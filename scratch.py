
# Import libraries
import torch
import torch.nn as nn
import tracemalloc


# Define function to measure CPU memory
def measure_cpu_memory(model, x):

    # Move to CPU
    device = torch.device("cpu")  # Ensure execution on CPU
    model = model.to(device)

    # Start memory tracking
    tracemalloc.start()

    # Forward pass
    output = model(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Measure peak memory usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Return peak memory usage
    return peak_mem


# Main
if __name__ == '__main__':

    # Example Model
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 32 * 32, 10)
    )

    # Example input
    x = torch.randn(1, 3, 32, 32)

    # Measure CPU memory
    measure_cpu_memory(model, x, dtype=torch.float32)

    # Done
    print('Done')


