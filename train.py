
# Import libraries
import os
import copy
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import local
from config import *
from dataset import collate_gdp


# Set up training function
def train_model(
    model, datasets, optimizer=None,
    batch_size=1, learning_rate=0.001, max_grad=1, n_epochs=5, 
    epoch_start=0, loss_val_best=float('inf'), model_state_dict_best=None,
    jobname=None, print_every=100, debug=False,
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
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_gdp,
        # pin_memory=True, n_workers=4, prefetch_factor=2,
    )

    # Set up optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up training statistics
    losses_train = []
    losses_val = []
    if model_state_dict_best is None:
        model_state_dict_best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

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

        # Loop over training batches
        t_batch = time.time()                           # Start timer
        if device.type != "cpu":
            torch.cuda.reset_peak_memory_stats(device)  # Start memory tracker
        for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_train):
            if debug and batch_idx > 10:
                    print('DEBUG MODE: Breaking early.')
                    break

            # Status update
            if batch_idx % print_every == 0:
                print(f'---- E{epoch}/{n_epochs} Batch {batch_idx}/{len(loader_train)} {jobname}')

            # Send to device
            scan = scan.to(device)
            beam = beam.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            body = body.to(device)
            dose = dose.to(device)

            # Get loss
            loss = model.calculate_loss(scan, beam, ptvs, oars, body, dose)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)  # Gradient clipping
            optimizer.step()

            # Update average loss
            loss_train_avg += loss.item() / len(loader_train)

            # Status update
            if batch_idx % print_every == 0:
                # Get time per batch
                t_elapsed = (time.time() - t_batch) / (1 if batch_idx == 0 else print_every)
                # Get memory usage 
                mem = 'n/a' if device.type == "cpu" else f'{torch.cuda.max_memory_allocated() / 1024**3:.2f}'
                # Print status
                print(
                    f'------ Time: {t_elapsed:.2f} s / batch | Mem: {mem} GB | Loss: {loss.item():.4f}'
                )
                # Reset timer and memory tracker
                t_batch = time.time()
                if device.type != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)

        ### Validation ###
        print('--Validation')

        # Set model to evaluation mode
        model.eval()

        # Initialize average loss
        loss_val_avg = 0

        # Loop over validation batches
        for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_val):
            if debug and batch_idx > 10:
                    print('DEBUG MODE: Breaking early.')
                    break

            # Send to device
            scan = scan.to(device)
            beam = beam.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            body = body.to(device)
            dose = dose.to(device)

            # Get loss
            with torch.no_grad():
                loss = model.calculate_loss(scan, beam, ptvs, oars, body, dose)

            # Update average loss
            loss_val_avg += loss.item() / len(loader_val)

            # Status update
            if batch_idx % print_every == 0:
                print(f'---- E{epoch}/{n_epochs} Val Batch {batch_idx}/{len(loader_val)}')


        ### Finalize training statistics ###
        print('--Finalizing training statistics')

        # Update training statistics
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
        'losses_train': losses_train,
        'losses_val': losses_val,
        'loss_val_best': loss_val_best,
        'model_state_dict_best': model_state_dict_best,
    }

    # Return model and training statistics
    return model, optimizer, training_statistics
    


