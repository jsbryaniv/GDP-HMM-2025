
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
    jobname=None, print_every=50, loss_type=None,
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
    def get_loss(ct, beam, ptvs, oars, body, dose, model):
        if loss_type is None or loss_type.lower() == 'mse':
            """
            Mean squared error loss
            """
            # Organize inputs
            x = torch.cat([ct, beam, ptvs, oars, body], dim=1)

            # Forward pass
            y = model(x)

            # Compute loss
            likelihood = F.mse_loss(y, dose)
            prior = sum(p.pow(2).sum() for p in model.parameters()) / n_parameters
            loss = likelihood + prior

            # Force garbage collection
            del ct, beam, ptvs, oars, body, dose, x, y
            if device.type != "cuda":
                torch.cuda.empty_cache()

        elif loss_type.lower() == 'crossae':
            """
            Cross attention autoencoder loss
            """
            # Organize inputs
            x = ptvs.clone()
            y_list = [ct, beam, ptvs, oars, body]

            # Forward pass
            z, y_list_ae = model(x, y_list)

            # Compute loss
            likelihood = F.mse_loss(z, dose)
            # recon_loss = sum(F.mse_loss(recon, ae_targets) for recon, ae_targets in zip(y_list_ae, y_list))
            ae_error_continuous = sum(F.mse_loss(recon, target) for recon, target in zip(y_list_ae[:-2], y_list[:-2]))
            ae_error_binary = sum(F.binary_cross_entropy_with_logits(recon, target) for recon, target in zip(y_list_ae[-2:], y_list[-2:]))
            recon_loss = ae_error_continuous + ae_error_binary
            prior = sum(p.pow(2).sum() for p in model.parameters()) / n_parameters
            loss = likelihood + recon_loss + prior

            # # Plot
            # fig, ax = plt.subplots(2, 3)
            # index = 64
            # ax[0, 0].imshow(ct[0,0,index,:,:].detach().cpu().numpy())
            # ax[0, 1].imshow(dose[0,0,index,:,:].detach().cpu().numpy())
            # ax[0, 2].imshow(z[0,0,index,:,:].detach().cpu().numpy())
            # ax[1, 0].imshow(y_list_ae[0][0,0,index,:,:].detach().cpu().numpy())
            # ax[1, 1].imshow(y_list_ae[1][0,0,index,:,:].detach().cpu().numpy())
            # ax[1, 2].imshow(y_list_ae[-2][0,0,index,:,:].detach().cpu().numpy())
            # plt.show()
            # plt.pause(1)
            # plt.savefig('_image.png')
            # plt.close()

            # Force garbage collection
            del ct, beam, ptvs, oars, body, dose, x, y_list, z, y_list_ae
            if device.type != "cuda":
                torch.cuda.empty_cache()

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

            # Get loss
            loss = get_loss(ct, beam, ptvs, oars, body, dose, model)

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
                t_elapsed = (time.time() - t_batch) / (10 if batch_idx > 0 else 1)
                # Get memory usage 
                mem = 0 if device.type == "cpu" else torch.cuda.max_memory_allocated() / 1024**3
                # Print status
                print(
                    f'------ Time: {t_elapsed:.2f} s / batch | Mem: {mem:.2f} GB | Loss: {loss.item():.4f}'
                )
                # Reset timer and memory tracker
                t_batch = time.time()
                if device.type != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)

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

            # Get loss
            with torch.no_grad():
                loss = get_loss(ct, beam, ptvs, oars, body, dose, model)

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
    


