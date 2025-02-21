
# Import libraries
import os
import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom libraries
from utils import dvh_loss, block_mask_3d


# Set up training function
def train_model(
    model, dataset_train, dataset_val,
    batch_size=1, loss_type=None, learning_rate=0.001, max_grad=1, n_epochs=10, epoch_start=0,
    jobname=None, print_every=100, debug=False,
): 
    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print status
    print('-'*50)
    print(f'Training model with {n_parameters} parameters on {device} for {n_epochs} epochs.')
    print('-'*50)
    if debug:
        for _ in range(10):
            print("WARNING: Be aware job is running with debug=True.")

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

        # Compute prior loss
        prior = 0
        for name, param in model.named_parameters():
            if 'bias' in name:
                prior += (param + .1).pow(2).sum()  # Bias relu threholds at -0.1 to prevent dead neurons
            else:
                prior += param.pow(2).sum()
        prior /= n_parameters

        # Compute likelihood loss
        if loss_type is None or loss_type.lower() == 'mse':
            """
            Mean squared error loss
            """
            # Organize inputs
            x = torch.cat([ct, beam, ptvs, oars, body], dim=1)

            # Forward pass
            pred = model(x)

            # Compute likelihood loss
            likelihood_pred = F.mse_loss(pred, dose)

            # Compute dvh loss
            likelihood_dvh = dvh_loss(
                 pred, dose, 
                 structures=torch.cat([(ptvs!=0), oars, body], dim=1)
            )

            # Combine losses
            likelihood = likelihood_pred + likelihood_dvh

        elif loss_type.lower() == 'crossae':
            """
            Cross attention autoencoder loss
            """
            # Organize inputs
            x = torch.cat([beam, ptvs], dim=1).clone()
            y_list = [ct, beam, ptvs, oars, body]
            
            # Corrupt context for autoencoder loss
            y_list_corrupted = [block_mask_3d(y.clone(), p=0.1) for y in y_list]

            # Forward pass
            pred = model(x, y_list)
            reconstructions = [ae(y) for ae, y in zip(model.context_autoencoders, y_list_corrupted)]

            # Compute likelihood loss
            likelihood_pred = F.mse_loss(pred, dose)

            # Compute dvh loss
            likelihood_dvh = dvh_loss(
                 pred, dose, 
                 structures=torch.cat([(ptvs!=0), oars, body], dim=1)
            )

            # Compute reconstruction loss
            likelihood_recon_continous = (
                sum([F.mse_loss(recon, y) for recon, y in zip(reconstructions[:-2], y_list[:-2])])
            )
            likelihood_recon_binary = (
                sum([F.binary_cross_entropy_with_logits(recon, y) for recon, y in zip(reconstructions[-2:], y_list[-2:])])
            )

            # Combine losses
            likelihood = (
                likelihood_pred 
                + likelihood_dvh
                + likelihood_recon_continous 
                + likelihood_recon_binary
            )

        # Compute total loss
        loss = prior + likelihood

        # # Plot
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, len(y_list)+1, figsize=(4*len(y_list)+1, 8))
        # index = x.shape[2] // 2
        # for i, (y, y_ae) in enumerate(zip([dose]+y_list, [z]+y_list_ae)):
        #     if i > 3:
        #         y_ae = torch.sigmoid(y_ae)
        #     ax[0, i].imshow(y[0,0,index,:,:].detach().cpu().numpy())
        #     ax[1, i].imshow(y_ae[0,0,index,:,:].detach().cpu().numpy())
        #     ax[0, i].set_title(f'({y.min().item():.2f}, {y.max().item():.2f})')
        #     ax[1, i].set_title(f'({y_ae.min().item():.2f}, {y_ae.max().item():.2f})')
        # plt.show()
        # plt.pause(1)
        # plt.savefig('_image.png')
        # plt.close()
        # print(loss.item())

        # Check for NaN and Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('Loss is NaN or Inf.')

        # Return loss
        return loss

    
    # Set up training statistics
    losses_train = []
    losses_val = []
    loss_val_best = float('inf')
    model_state_best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    # Training loop
    for epoch in range(epoch_start, epoch_start+n_epochs):
        if debug and epoch > epoch_start + 1:
                print('DEBUG MODE: Breaking early.')
                break

        # Status update
        t_epoch = time.time()
        if jobname is not None:
            print(f'████ {jobname} | Epoch {epoch}/{n_epochs} ████')
        else:
            print(f'████ Epoch {epoch}/{n_epochs} ████')

        ### Training ###
        print('--Training')

        # Initialize average loss
        loss_train_avg = 0

        # Loop over batches
        t_batch = time.time()                           # Start timer
        if device.type != "cpu":
            torch.cuda.reset_peak_memory_stats(device)  # Start memory tracker
        for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_train):
            if debug and batch_idx > 10:
                    print('DEBUG MODE: Breaking early.')
                    break

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
                t_elapsed = (time.time() - t_batch) / (1 if batch_idx == 0 else print_every)
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
            if debug and batch_idx > 10:
                    print('DEBUG MODE: Breaking early.')
                    break

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
            model_state_best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

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
    


