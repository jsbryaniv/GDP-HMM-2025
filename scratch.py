
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

{"losses_train": [0.02846313427042655, 0.03269330804239494], "losses_val": [0.08249649374108564, 0.07744296893738864], "loss_val_best": 0.07744296893738864}
{"losses_train": [0.02846313427042655, 0.03269330804239494, 0.01695274965904975, 0.010123631326849012], "losses_val": [0.08249649374108564, 0.07744296893738864, 0.04097211486414859, 0.02718196270758646], "loss_val_best": 0.02718196270758646}
{"losses_train": [0.02846313427042655, 0.03269330804239494, 0.01695274965904975, 0.010123631326849012, 0.0074497248494917205, 0.006111893815921931], "losses_val": [0.08249649374108564, 0.07744296893738864, 0.04097211486414859, 0.02718196270758646, 0.026415043860151054, 0.012504906821669192], "loss_val_best": 0.012504906821669192}