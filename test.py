
# Import libraries
import os
import json
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Import custom libaries
from main import load_dataset, load_model
from plotting import plot_losses

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)
ROOT_DIR = config['PATH_OUTPUT']


# Load trained model and dataset
def load_model_and_test_dataset(savename):

    ### EXTRACT PARAMETERS ###

    # Load json
    with open(os.path.join(ROOT_DIR, f'{savename}.json'), 'r') as f:
        metadata = json.load(f)

    # Extract metadata
    dataID = metadata['dataID']
    modelID = metadata['modelID']
    data_kwargs = metadata['data_kwargs']
    model_kwargs = metadata['model_kwargs']
    train_kwargs = metadata['train_kwargs']
    indices_train = metadata['indices_train']
    indices_val = metadata['indices_val']
    indices_test = metadata['indices_test']
    training_statistics = metadata['training_statistics']


    ### LOAD DATASET ###
    
    # Load dataset
    dataset, data_metadata = load_dataset(dataID, **data_kwargs)
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
    model = load_model(modelID, in_channels, out_channels, **model_kwargs)

    # Load weights from file
    model_state_dict = torch.load(os.path.join(ROOT_DIR, f'{savename}.pth'))
    model.load_state_dict(model_state_dict)

    
    ### RETURN OUTPUTS ###

    # Return outputs
    return model, datasets, metadata



# Set up training function
def test_model(
    model, dataset_test, metadata, n_show=5,
): 

    # Extract metadata
    dataID = metadata['dataID']
    modelID = metadata['modelID']
    data_kwargs = metadata['data_kwargs']
    model_kwargs = metadata['model_kwargs']
    train_kwargs = metadata['train_kwargs']
    indices_train = metadata['indices_train']
    indices_val = metadata['indices_val']
    indices_test = metadata['indices_test']
    training_statistics = metadata['training_statistics']


    # Set up model
    model.eval()

    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set up data loader
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Initialize lists
    plot_row = []
    plot_loss = []

    # Loop over batches
    print(f'Testing model with {n_parameters} parameters on {device}.')
    for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if batch_idx >= n_show:
            break

        # Send to device
        ct = ct.to(device)
        beam = beam.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        body = body.to(device)
        dose = dose.to(device)

        # Check loss type
        with torch.no_grad():
            if ('loss_type' not in train_kwargs) or (train_kwargs['loss_type'].lower() == 'mse'):
                """
                Mean squared error loss
                """
                # Get prediction
                x = torch.cat([ct, beam, ptvs, oars, body], dim=1)
                y = model(x)
                # Get plot data
                slice_index = ct.shape[-3] // 2
                plot_labels = ['CT', 'Dose (Ground Truth)', 'Dose (Prediction)']
                plot_col = [
                    ct[0, 0, slice_index].cpu().detach().numpy(),    # CT
                    dose[0, 0, slice_index].cpu().detach().numpy(),  # Dose Ground Truth
                    y[0, 0, slice_index].cpu().detach().numpy(),     # Dose Prediction
                ]
                # Get loss
                body_index = body.cpu().detach().numpy().astype(bool)
                loss = f'MSE={F.mse_loss(y[body_index], dose[body_index]).item():.4f}; MAE={F.l1_loss(y[body_index], dose[body_index]).item():.4f}'
                plot_loss.append(loss)

            elif train_kwargs['loss_type'].lower() == 'crossae':
                """
                Cross Attention Autoencoder loss
                """
                # Get prediction
                x = ptvs.clone()
                y_list = [ct, beam, ptvs, oars, body]
                z, y_list_ae = model(x, y_list)
                slice_index = ct.shape[-3] // 2
                plot_labels = ['CT', 'Dose (Ground Truth)', 'Dose (Prediction)']
                plot_col = [
                    ct[0, 0, slice_index].cpu().detach().numpy(),    # CT
                    dose[0, 0, slice_index].cpu().detach().numpy(),  # Dose Ground Truth
                    z[0, 0, slice_index].cpu().detach().numpy(),     # Dose Prediction
                ]
                # body_index = body.astype(bool)
                # ^^^ This doesnt work. Try this instead:
                # Get loss
                body_index = body.cpu().detach().numpy().astype(bool)
                loss = f'MSE={F.mse_loss(z[body_index], dose[body_index]).item():.4f}; MAE={F.l1_loss(z[body_index], dose[body_index]).item():.4f}'
                plot_loss.append(loss)
                # Get plot data
                # # Softmax binary outputs
                # y_list_ae = y_list_ae[:-2] + [torch.sigmoid(y) for y in y_list_ae[-2:]]
                # # Sum multichannel outputs
                # y_list = [y.sum(dim=1, keepdim=True) for y in y_list]
                # y_list_ae = [y.sum(dim=1, keepdim=True) for y in y_list_ae]
                # # Get plot data
                # slice_index = ct.shape[-3] // 2
                # plot_labels = [
                #     'CT Ground Truth', 
                #     'CT Prediction', 
                #     'PTVs Ground Truth', 
                #     'PTVs Prediction', 
                #     'OARs Ground Truth', 
                #     'OARs Prediction', 
                #     'Dose Ground Truth', 
                #     'Dose Prediction'
                # ]
                # plot_col = [
                #     y_list[0][0, 0, slice_index].cpu().detach().numpy(),     # CT Ground Truth
                #     y_list_ae[0][0, 0, slice_index].cpu().detach().numpy(),  # CT Prediction
                #     y_list[2][0, 0, slice_index].cpu().detach().numpy(),     # PTVs Ground Truth
                #     y_list_ae[2][0, 0, slice_index].cpu().detach().numpy(),  # PTVs Prediction
                #     y_list[3][0, 0, slice_index].cpu().detach().numpy(),     # OARs Ground Truth
                #     y_list_ae[3][0, 0, slice_index].cpu().detach().numpy(),  # OARs Prediction
                #     dose[0, 0, slice_index].cpu().detach().numpy(),          # Dose Ground Truth
                #     z[0, 0, slice_index].cpu().detach().numpy(),             # Dose Prediction
                # ]

        # Append to plot row
        plot_row.append(plot_col)

    # Set up figure
    n_rows = len(plot_row)
    n_cols = len(plot_row[0])
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    plt.ion()
    plt.show()
    for i in range(n_rows):
        ax[i, 0].set_ylabel(f'Example {i+1}')
        for j in range(n_cols):
            ax[i, j].set_title(f'{plot_labels[j]}\n{plot_loss[i]}')
            ax[i, j].imshow(plot_row[i][j], cmap='hot')
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.pause(1)
    
    # Return figure
    return fig, ax
    

# Main script
if __name__ == '__main__':

    # Make list of all jobs
    all_jobs = [
        'model_HaN_half_CrossAttnAE',
        'model_HaN_half_Unet',
        'model_HaN_Unet',
        'model_HaN_ViT',
    ]

    # Loop over savenames
    for savename in all_jobs:

        # Load model and dataset
        model, datasets, metadata = load_model_and_test_dataset(savename)

        # Test model
        dataset_test = datasets[2]
        fig, ax = test_model(model, dataset_test, metadata)
        fig.savefig(f'figs/{savename}_test.png')
        plt.close()

        # Plot losses
        losses_train = metadata['training_statistics']['losses_train']
        losses_val = metadata['training_statistics']['losses_val']
        fig, ax = plot_losses(losses_train, losses_val)
        fig.savefig(f'figs/{savename}_losses.png')
        plt.close()
    
    # Done
    print('Done.')

