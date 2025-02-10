
# Import libraries
import os
import json
import copy
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Import custom libaries
from main import load_dataset, load_model
from plotting import plot_losses

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)


# Load trained model and dataset
def load_model_and_test_dataset(savename, root_dir=None):

    # Check inputs
    if root_dir is None:
        root_dir = config['PATH_OUTPUT']

    ### EXTRACT PARAMETERS ###

    # Load json
    with open(os.path.join(root_dir, f'{savename}.json'), 'r') as f:
        training_statistics = json.load(f)
    
    # Get parameters
    dataID = savename.split('_')[1]
    modelID = savename.split('_')[2]
    model_kwargs = {}
    data_kwargs = {}
    train_kwargs = {}
    ## For future use:
    # modelID = training_statistics['modelID']
    # dataID = training_statistics['dataID']
    # model_kwargs = training_statistics['model_kwargs']
    # data_kwargs = training_statistics['data_kwargs']
    # train_kwargs = training_statistics['train_kwargs']
    indices_train = training_statistics['indices_train']
    indices_val = training_statistics['indices_val']
    indices_test = training_statistics['indices_test']



    ### LOAD DATASET ###
    
    # Load dataset
    dataset, data_metadata = load_dataset(dataID, **data_kwargs)

    # Get metadata
    in_channels = data_metadata['in_channels']
    out_channels = data_metadata['out_channels']

    # Split into train, validation, and test sets
    dataset_train = Subset(dataset, indices_train)
    dataset_val = Subset(dataset, indices_val)
    dataset_test = Subset(dataset, indices_test)

    # Package into tuple
    datasets = (dataset_train, dataset_val, dataset_test)


    ### LOAD MODEL ###

    # Initialize model
    model = load_model(modelID, **model_kwargs)

    # Load weights from file
    model_state_dict = torch.load(os.path.join(root_dir, f'{savename}.pth'))
    model.load_state_dict(model_state_dict)

    
    ### RETURN OUTPUTS ###

    # Return outputs
    return model, datasets, training_statistics



# Set up training function
def test_model(
    model, dataset_test, n_show=5,
): 
    
    # Set up model
    model.eval()

    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set up data loader
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    # Set up figure
    fig, ax = plt.subplots(n_show, 3, figsize=(15, 5*n_show))
    plt.ion()
    plt.show()

    # Print status
    print(f'Testing model with {n_parameters} parameters on {device}.')

    # Loop over batches
    for batch_idx, (ct, ptvs, oars, beam, dose) in enumerate(loader_test):
        if batch_idx >= n_show:
            break

        # Send to device
        ct = ct.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        beam = beam.to(device)
        dose = dose.to(device)
        x = torch.cat([ct, ptvs, oars, beam], dim=1)
        
        # Get prediction
        with torch.no_grad():
            y = model(x)

        # Get slice
        idx = ct.shape[-2] // 2

        # Plot
        ax[batch_idx, 0].imshow(ct[0, 0, idx].cpu().detach().numpy(), cmap='bone')
        ax[batch_idx, 1].imshow(dose[0, 0, idx].cpu().detach().numpy(), cmap='hot')
        ax[batch_idx, 2].imshow(y[0, 0, idx].cpu().detach().numpy(), cmap='hot')

    # Finalize figure
    ax[0, 0].set_title('CT')
    ax[0, 1].set_title('Dose (Ground Truth)')
    ax[0, 2].set_title('Dose (Prediction)')
    for i in range(n_show):
        ax[i, 0].set_ylabel(f'Example {i+1}')
        for j in range(3):
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.pause(.5)
    
    # Return figure
    return fig, ax
    

# Main script
if __name__ == '__main__':

    # Loop over savenames
    all_jobs = ['model_HaN_Unet', 'model_HaN_ViT', 'model_HaN_ConvFormer']
    for savename in all_jobs:

        # Load model and dataset
        model, datasets, training_statistics = load_model_and_test_dataset(savename)

        # Test model
        dataset_test = datasets[2]
        fig, ax = test_model(model, dataset_test)
        fig.savefig(f'figs/{savename}_test.png')
        plt.close()

        # Plot losses
        losses_train = training_statistics['losses_train']
        losses_val = training_statistics['losses_val']
        fig, ax = plot_losses(losses_train, losses_val)
        fig.savefig(f'figs/{savename}_losses.png')
        plt.close()
    
    # Done
    print('Done.')

