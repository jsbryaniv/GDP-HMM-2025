
# Import libraries
import os
import json
import copy
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom libaries
from losses import competition_loss

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)
ROOT_DIR = config['PATH_OUTPUT']

# Set up training function
@torch.no_grad()
def test_model(model, dataset_test, debug=False): 

    # Set up model
    model.eval()

    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set up data loader
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Initialize loss
    loss_test = 0
    n_test = 0

    # Loop over batches
    print(f'Testing model with {n_parameters} parameters on {device}.')
    for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if debug and batch_idx > 10:
            print('DEBUG MODE: Breaking early.')
            break
        
        # Status update
        if batch_idx % 100 == 0:
            print(f'-- Batch {batch_idx}/{len(loader_test)}')
        
        # Send to device
        ct = ct.to(device)
        beam = beam.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        body = body.to(device)
        dose = dose.to(device)

        # Get prediction
        if type(model).__name__.endswith('CrossAttnAEModel'):
            x = torch.cat([beam, ptvs], dim=1).clone()
            y_list = [
                ct, 
                torch.cat([beam, ptvs], dim=1), 
                torch.cat([oars, body], dim=1),
            ]
            pred = model(x, y_list)
        else:
            x = torch.cat([ct, beam, ptvs, oars, body], dim=1)
            pred = model(x)

        # Calculate loss
        loss = competition_loss(pred, dose, body)

        # Update loss
        loss_test += loss.item()
        n_test += 1

    # Normalize loss
    loss_test /= n_test

    # Print loss
    print(f'-- Average loss on test dataset: {loss_test}') 
    
    # Return total loss
    return loss_test


# Test model
if __name__ == '__main__':

    # Import custom libraries
    from utils import load_model_and_datasets

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and datasets
    savename = 'model_HaN_CrossAttnAE'
    model, datasets, metadata = load_model_and_datasets(savename)
    model.to(device)

    # Test model
    loss_test = test_model(model, datasets[-1])
    
    # Done
    print('Done.')
