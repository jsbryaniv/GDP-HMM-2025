
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
    batch_size=1, learning_rate=0.01, num_epochs=100,
): 
    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print status
    print('-'*50)
    print(f'Training model with {n_parameters} parameters on {device} for {num_epochs} epochs.')
    print('-'*50)

    # Set up data loaders
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        # pin_memory=True, num_workers=4, prefetch_factor=2,
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
    best_loss_val = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    # Training loop
    for epoch in range(num_epochs):

        # Status update
        t_epoch = time.time()
        print(f'████ Epoch {epoch+1}/{num_epochs} ████')

        ### Training ###
        print('--Training')

        # Initialize average loss
        avg_loss_train = 0

        # Loop over batches
        t_batch = time.time()
        for batch_idx, (ct, ptvs, oars, beam, dose) in enumerate(loader_train):

            # Status update
            if batch_idx % 10 == 0:
                print(f'---- Batch {batch_idx}/{len(loader_train)}')

            # Send to device
            ct = ct.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            beam = beam.to(device)
            dose = dose.to(device)

            # Forward pass
            x = torch.cat([ct, ptvs, oars, beam], dim=1)
            y = model(x)
            loss = loss_fn(y, dose, model)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update average loss
            avg_loss_train += loss.item() / len(loader_train)

            # Status update
            if batch_idx % 10 == 0:
                t_batch = time.time() - t_batch
                if batch_idx > 0:
                    t_batch /= 10
                print(f'------ Time: {t_batch:.2f} s / batch | Loss: {loss.item():.4f}')
                t_batch = time.time()

        ### Validation ###
        print('--Validation')

        # Initialize average loss
        avg_loss_val = 0

        # Loop over batches
        for batch_idx, (ct, ptvs, oars, beam, dose) in enumerate(loader_val):

            # Send to device
            ct = ct.to(device)
            ptvs = ptvs.to(device)
            oars = oars.to(device)
            beam = beam.to(device)
            dose = dose.to(device)

            # Forward pass
            with torch.no_grad():
                x = torch.cat([ct, ptvs, oars, beam], dim=1)
                y = model(x)
                loss = loss_fn(y, dose, model)

            # Update average loss
            avg_loss_val += loss.item() / len(loader_val)

            # Status update
            if batch_idx % 10 == 0:
                print(f'---- Batch {batch_idx}/{len(loader_val)}')

        ### Finalize training statistics ###
        print('--Finalizing training statistics')

        # Update training statistics
        losses_train.append(avg_loss_train)
        losses_val.append(avg_loss_val)
        if avg_loss_val < best_loss_val:
            best_loss_val = avg_loss_val
            best_model_state = copy.deepcopy(model.state_dict())

        # Status update
        print(f'-- Summary')
        print(f'---- Train Loss: {avg_loss_train:.4f}')
        print(f'---- Val Loss: {avg_loss_val:.4f}')
        print(f'---- Time: {time.time()-t_epoch:.2f} s / epoch')

    
    ### Training complete ###
    print(f'Training complete. Best validation loss: {best_loss_val:.4f}')

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Finalize training statistics
    training_statistics = {
        'losses_train': losses_train,
        'losses_val': losses_val,
        'best_loss_val': best_loss_val,
    }

    # Return model and training statistics
    return model, training_statistics
    


