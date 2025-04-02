
# Import libraries
import os
import copy
import time
import torch
import tracemalloc
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import local
from config import *
from dataset import collate_gdp
from utils import inspect_parameters, inspect_activations


# Set up training function
def train_model(
    model, datasets, optimizer=None,
    batch_size=1, learning_rate=0.001, max_grad=1, n_epochs=2, 
    epoch_start=0, loss_val_best=float('inf'), model_state_dict_best=None,
    jobname=None, print_every=100, debug=False, num_workers=0,
): 
    # Set up constants
    if jobname is None:
         jobname = ''
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_epochs = n_epochs + epoch_start  # Start counting epochs from epoch_start
    
    # Print status
    print('-'*50)
    print(f'Training model with {n_parameters} parameters on {device} for epochs {epoch_start}-{n_epochs}.')
    print('-'*50)
    if debug:
        for _ in range(5):
            print("WARNING: Be aware job is running with debug=True.")

    # Set up data loaders
    dataset_train, dataset_val = datasets
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_gdp)
    if num_workers == 0:
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_gdp)
    else:
        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_gdp,
            num_workers=num_workers, prefetch_factor=2, pin_memory=False, 
        )

    # Set up optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up training statistics
    if model_state_dict_best is None:
        model_state_dict_best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    losses_train = []
    losses_val = []
    time_stats = []
    mem_GPU_stats = []
    mem_CPU_stats = []

    # Training loop
    for epoch in range(epoch_start, n_epochs):
        if debug and epoch > epoch_start + 1:
                print('DEBUG MODE: Breaking early.')
                break

        # Status update
        t_epoch = time.time()
        print(f'████ Epoch {epoch}/{n_epochs} {jobname} ████')

        ### Training ###
        print('--Training')

        # Set model to training mode
        model.train()

        # Initialize average loss
        loss_train_avg = 0
        loss_train_counter = 0

        # Loop over training batches
        t_batch = time.time()                           # Start timer
        tracemalloc.start()                             # Start CPU memory tracker
        if device.type != "cpu":
            torch.cuda.reset_peak_memory_stats(device)  # Start GPU memory tracker
        for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_train):
            if debug and batch_idx > 2:
                print('DEBUG MODE: Breaking early.')
                break

            # Status update
            if (batch_idx % print_every == 0) or ((epoch == epoch_start) and (batch_idx < 10)):
                print(f'---- E{epoch}/{n_epochs} Batch {batch_idx}/{len(loader_train)} {jobname}')

            # Send to device
            scan, beam, ptvs, oars, body, dose = [
                x.to(device) for x in (scan, beam, ptvs, oars, body, dose)
            ]

            # Zero gradients
            optimizer.zero_grad()

            # Get loss
            loss = model.calculate_loss(scan, beam, ptvs, oars, body, dose)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()

            # Update average loss
            loss_train_avg += loss.item()
            loss_train_counter += 1

            # Inspect parameters and activations
            flag = inspect_parameters(model)
            if flag:
                inspect_activations(model, scan, beam, ptvs, oars, body)

            # Get memory usage
            mem_CPU = tracemalloc.get_traced_memory()[1] / 1024**3
            tracemalloc.stop()
            tracemalloc.start()
            if device.type != "cpu":
                mem_GPU = torch.cuda.max_memory_allocated(device) / 1024**3
                torch.cuda.reset_peak_memory_stats(device)
            else:
                mem_GPU = 0
            mem_CPU_stats.append(mem_CPU)
            mem_GPU_stats.append(mem_GPU)

            # Update time statistics
            t_elapsed = time.time() - t_batch
            t_batch = time.time()
            time_stats.append(t_elapsed)

            # Status update
            if (batch_idx % print_every == 0) or ((epoch == epoch_start) and (batch_idx < 10)):
                print(' | '.join([
                    f'------ Time: {t_elapsed:.2f} s / batch',
                    f'GPU: {mem_GPU:.2f} GB',
                    f'CPU: {mem_CPU:.2f} GB',
                    f'Loss: {loss.item():.4f}',
                ]))
        

        ### Validation ###
        print('--Validation')

        # Set model to evaluation mode
        model.eval()

        # Initialize average loss
        loss_val_avg = 0
        loss_val_counter = 0

        # Loop over validation batches
        for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_val):
            if debug and batch_idx > 2:
                    print('DEBUG MODE: Breaking early.')
                    break

            # Send to device
            scan, beam, ptvs, oars, body, dose = [
                x.to(device) for x in (scan, beam, ptvs, oars, body, dose)
            ]

            # Get loss
            with torch.no_grad():
                loss = model.calculate_loss(scan, beam, ptvs, oars, body, dose)

            # Update average loss
            loss_val_avg += loss.item()
            loss_val_counter += 1

            # Status update
            if batch_idx % print_every == 0:
                print(f'---- E{epoch}/{n_epochs} Val Batch {batch_idx}/{len(loader_val)}')


        ### Finalize training statistics ###
        print('--Finalizing training statistics')

        # Update training statistics
        loss_train_avg /= loss_train_counter
        loss_val_avg /= loss_val_counter
        losses_train.append(loss_train_avg)
        losses_val.append(loss_val_avg)
        if loss_val_avg < loss_val_best:
            loss_val_best = loss_val_avg
            model_state_dict_best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        # Status update
        print(f'-- Epoch {epoch}/{n_epochs} Summary {jobname}')
        print(f'---- Train Loss: {loss_train_avg:.4f}')
        print(f'---- Val Loss: {loss_val_avg:.4f}')
        print(f'---- Time: {time.time()-t_epoch:.2f} s / epoch')

    
    ### Training complete ###
    print(f'Training complete. Best validation loss: {loss_val_best:.4f}')
    
    # Finalize training statistics
    training_statistics = {
        'epoch': epoch,
        'mem_stats': mem_GPU_stats,
        'time_stats': time_stats,
        'losses_train': losses_train,
        'losses_val': losses_val,
        'loss_val_best': loss_val_best,
        'model_state_dict_best': model_state_dict_best,
    }

    # Return model and training statistics
    return model, optimizer, training_statistics
    


