
# Import libraries
import os
import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Set up training function
def train_model(
    model, dataset_train, dataset_val,
    batch_size=1, learning_rate=0.01, max_grad=1, n_epochs=100,
    jobname=None, print_every=50,
): 
    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print status
    print('-'*50)
    print(f'Training model with {n_parameters} parameters on {device} for {n_epochs} epochs.')
    print('-'*50)

    # Set up data loaders
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        # pin_memory=True, n_workers=4, prefetch_factor=2,
    )

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up loss function
    def loss_fn(prediction, target, model):

        # Loss from likelihood (MSE)
        likelihood = F.mse_loss(prediction, target)

        # Loss from prior / regularization
        prior = sum(p.pow(2).sum() for p in model.parameters()) / n_parameters

        # Combine losses
        loss = likelihood + prior

        # Return loss
        return loss
    
    # Set up training statistics
    losses_train = []
    losses_val = []
    loss_val_best = float('inf')
    model_state_best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # Training loop
    for epoch in range(n_epochs):

        # Status update
        t_epoch = time.time()
        if jobname is not None:
            print(f'████ {jobname} | Epoch {epoch+1}/{n_epochs} ████')
        else:
            print(f'████ Epoch {epoch+1}/{n_epochs} ████')

        ### Training ###
        print('--Training')

        # Initialize average loss
        loss_train_avg = 0

        # Loop over batches
        t_batch = time.time()                           # Start timer
        if device.type != "cpu":
            torch.cuda.reset_peak_memory_stats(device)  # Start memory tracker
        for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_train):

            # Status update
            if batch_idx % print_every == 0:
                print(f'---- E{epoch}/{n_epochs} Batch {batch_idx}/{len(loader_train)}')

            # Send to device
            ct = ct.to(device)
            beam = beam.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            body = body.to(device)
            dose = dose.to(device)

            # Forward pass
            x = torch.cat([ct, beam, ptvs, oars, body], dim=1)
            y = model(x)
            loss = loss_fn(y, dose, model)

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
                t_batch = time.time() - t_batch
                if batch_idx > 0:
                    t_batch /= 10
                # Get memory usage 
                if device.type != "cpu":
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    mem = 0
                # Print status
                print(
                    f'------ Time: {t_batch:.2f} s / batch | Mem: {mem:.2f} GB | Loss: {loss.item():.4f}'
                )
                # Reset timer and memory tracker
                t_batch = time.time()
                if device.type != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)

            # Force garbage collection
            del ct, beam, ptvs, oars, body, dose, x, y, loss
            if device.type != "cuda":
                torch.cuda.empty_cache()

        ### Validation ###
        print('--Validation')

        # Initialize average loss
        loss_val_avg = 0

        # Loop over batches
        for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_val):

            # Send to device
            ct = ct.to(device)
            beam = beam.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            body = body.to(device)
            dose = dose.to(device)

            # Forward pass
            with torch.no_grad():
                x = torch.cat([ct, beam, ptvs, oars, body], dim=1)
                y = model(x)
                loss = loss_fn(y, dose, model)

            # Update average loss
            loss_val_avg += loss.item() / len(loader_val)

            # Status update
            if batch_idx % print_every == 0:
                print(f'---- E{epoch}/{n_epochs} Val Batch {batch_idx}/{len(loader_val)}')

            # Force garbage collection
            del ct, beam, ptvs, oars, body, dose, x, y, loss
            if device.type != "cuda":
                torch.cuda.empty_cache()

        ### Finalize training statistics ###
        print('--Finalizing training statistics')

        # Update training statistics
        losses_train.append(loss_train_avg)
        losses_val.append(loss_val_avg)
        if loss_val_avg < loss_val_best:
            loss_val_best = loss_val_avg
            model_state_best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

        # Status update
        print(f'-- Epoch {epoch}/{n_epochs} Summary:')
        print(f'---- Train Loss: {loss_train_avg:.4f}')
        print(f'---- Val Loss: {loss_val_avg:.4f}')
        print(f'---- Time: {time.time()-t_epoch:.2f} s / epoch')

    
    ### Training complete ###
    print(f'Training complete. Best validation loss: {loss_val_best:.4f}')

    # Load best model
    model.load_state_dict(model_state_best)
    
    # Finalize training statistics
    training_statistics = {
        'losses_train': losses_train,
        'losses_val': losses_val,
        'loss_val_best': loss_val_best,
    }

    # Return model and training statistics
    return model, training_statistics
    


