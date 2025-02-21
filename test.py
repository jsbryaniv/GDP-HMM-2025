
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
from plotting import plot_losses, copy_axis

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)
ROOT_DIR = config['PATH_OUTPUT']


# Get DVH function
def get_dvh(dose, structures, bins=100):

    # If tensor, convert to numpy
    if isinstance(dose, torch.Tensor):
        dose = dose.cpu().detach().numpy()
    if isinstance(structures, torch.Tensor):
        structures = structures.cpu().detach().numpy()
    
    # Get dose range
    bins = np.linspace(0, np.max(dose), bins)
    dvh_bin = (bins[:-1] + bins[1:]) / 2
    bins = np.append(bins, 2*bins[-1]-bins[-2])  # Add final point to ensure 0% above max dose
    bins = np.append(0, bins)                    # Add initial point to ensure 100% at 0 dose

    # Initialize dvhs
    dvh_val = []

    # Loop over structures
    for i in range(structures.shape[1]):
        structure = structures[0, i]
        if np.sum(structure) == 0:
            continue
        
        # Get dvh
        hist, _ = np.histogram(dose[0, 0, structure.astype(bool)], bins=bins)  # Get histogram
        dvh = np.cumsum(hist[::-1])[::-1]                                      # Get cumulative histogram (reverse order)
        dvh = 100 * dvh / (dvh[0] + 1e-8)                                      # Normalize to 100%
        dvh = np.append(dvh, 0)                                                # Add final point to ensure 0% above max dose
        dvh = np.append(100, dvh)                                              # Add initial point to ensure 100% at 0 dose

        # Append to list
        dvh_val.append(dvh)

    # Convert to numpy array
    dvh_val = np.array(dvh_val)

    # Return list
    return dvh_val, dvh_bin

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
    dvh_gt_val_list = []
    dvh_gt_bin_list = []
    dvh_pred_val_list = []
    dvh_pred_bin_list = []

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
                pred = body*y
                # Get plot data
                slice_index = ct.shape[-3] // 2
                plot_labels = ['CT', 'Dose (Ground Truth)', 'Dose (Prediction)']
                plot_col = [
                    ct[0, 0, slice_index].cpu().detach().numpy(),        # CT
                    dose[0, 0, slice_index].cpu().detach().numpy(),      # Dose Ground Truth
                    pred[0, 0, slice_index].cpu().detach().numpy(),      # Dose Prediction
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
                x = torch.cat([beam, ptvs], dim=1).clone()
                y_list = [ct, beam, ptvs, oars, body]
                z, y_list_ae = model(x, y_list)
                pred = body*z
                # Get plot data
                slice_index = ct.shape[-3] // 2
                plot_labels = ['CT', 'Dose (Ground Truth)', 'Dose (Prediction)']
                plot_col = [
                    ct[0, 0, slice_index].cpu().detach().numpy(),        # CT
                    dose[0, 0, slice_index].cpu().detach().numpy(),      # Dose Ground Truth
                    pred[0, 0, slice_index].cpu().detach().numpy(),      # Dose Prediction
                ]
                # Get loss
                body_index = body.cpu().detach().numpy().astype(bool)
                loss = f'MSE={F.mse_loss(z[body_index], dose[body_index]).item():.4f}; MAE={F.l1_loss(z[body_index], dose[body_index]).item():.4f}'
                plot_loss.append(loss)

        # Append to plot row
        plot_row.append(plot_col)

        # Get DVHs
        dvh_gt_val, dvh_gt_bin = get_dvh(dose, torch.cat([ptvs, oars], dim=1))
        dvh_pred_val, dvh_pred_bin = get_dvh(pred, torch.cat([ptvs, oars], dim=1))
        dvh_gt_val_list.append(dvh_gt_val)
        dvh_gt_bin_list.append(dvh_gt_bin)
        dvh_pred_val_list.append(dvh_pred_val)
        dvh_pred_bin_list.append(dvh_pred_bin)

    # Set up figure
    n_rows = len(plot_row)
    n_cols = len(plot_row[0]) + 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    plt.ion()
    plt.show()
    for i in range(n_rows):
        ax[i, 0].set_ylabel(f'Example {i+1}')
        # Plot images
        for j in range(n_cols-2):
            ax[i, j].set_title(f'{plot_labels[j]}' + (f'\n{plot_loss[i]}' if j == 2 else ''))
            ax[i, j].imshow(plot_row[i][j], cmap=('gray' if j == 0 else 'hot'))
            ax[i, j].axis('off')
        # Plot DVH ground truth
        ax[i, n_cols-2].set_title('DVH (Ground Truth)')
        ax[i, n_cols-2].set_xlabel('Dose (Gy)')
        ax[i, n_cols-2].set_ylabel('Volume (%)')
        dvh_gt_bin = dvh_gt_bin_list[i]
        dvh_gt_val = dvh_gt_val_list[i]
        for s in range(dvh_gt_val.shape[0]):
            ax[i, n_cols-2].plot(dvh_gt_bin, dvh_gt_val[s])
        # Plot DVH prediction
        ax[i, n_cols-1].set_title('DVH (Prediction)')
        ax[i, n_cols-1].set_xlabel('Dose (Gy)')
        ax[i, n_cols-1].set_ylabel('Volume (%)')
        dvh_pred_bin = dvh_pred_bin_list[i]
        dvh_pred_val = dvh_pred_val_list[i]
        for s in range(dvh_pred_val.shape[0]):
            ax[i, n_cols-1].plot(dvh_pred_bin, dvh_pred_val[s])
    plt.tight_layout()
    plt.pause(1)
    
    # Return figure
    return fig, ax
    

# Main script
if __name__ == '__main__':

    # Loop over dataIDs
    for dataID in ['HaN', 'HalfHaN']:

        # Get all jobs
        if dataID.lower() == 'han':
            all_jobs = [
                'model_HaN_CrossAttnAE',
                'model_HaN_Unet',
                'model_HaN_ViT',
            ]
        elif dataID.lower() == 'halfhan':
            all_jobs = [
                'model_HalfHaN_CrossAttnAE',
                'model_HalfHaN_Unet',
                'model_HalfHaN_ViT_scale=2_shape=64',
            ]

        # Loop over savenames
        figs = []
        axs = []
        for savename in all_jobs:

            # Load model and dataset
            model, datasets, metadata = load_model_and_test_dataset(savename)

            # Plot losses
            losses_train = metadata['training_statistics']['losses_train']
            losses_val = metadata['training_statistics']['losses_val']
            fig, ax = plot_losses(losses_train, losses_val)
            fig.savefig(f'figs/{savename}_losses.png')
            plt.close()

            # Test model
            dataset_test = datasets[2]
            fig, ax = test_model(model, dataset_test, metadata)
            fig.savefig(f'figs/{savename}_test.png')
            figs.append(fig)
            axs.append(ax)

        # Create summary figure
        n_jobs = len(all_jobs)
        n_rows = axs[0].shape[0]
        n_cols = 3 + 2*n_jobs
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        plt.ion()
        plt.show()
        for i in range(n_rows):
            # Plot CT and ground truth Dose
            ax[i, 0] = copy_axis(axs[0][i, 0], ax[i, 0])
            ax[i, 1] = copy_axis(axs[0][i, 1], ax[i, 1])
            # Plot predicted dose for each job
            for job in range(n_jobs):
                ax[i, job+2] = copy_axis(axs[job][i, 2], ax[i, job+2])
                ax[i, job+2].set_title(
                    all_jobs[job].split('_')[2]+'\n'+axs[job][i, 2].get_title().split('\n')[-1]
                )
            # Plot ground truth DVH
            ax[i, n_jobs+2].set_title('DVH (Ground Truth)')
            ax[i, n_jobs+2] = copy_axis(axs[0][i, -2], ax[i, n_jobs+2])
            ax[i, n_jobs+2].set_xlim([0, 8])
            # Plot predicted DVH for each job
            for job in range(len(all_jobs)):
                ax[i, n_jobs+3+job] = copy_axis(axs[job][i, -1], ax[i, n_jobs+3+job])
                ax[i, n_jobs+3+job].set_title(f'DVH {all_jobs[job].split("_")[2]}')
                ax[i, n_jobs+3+job].set_xlim([0, 8])
        plt.tight_layout()
        plt.pause(1)
        fig.savefig(f'figs/summary_{dataID}.png')

        # Close figures
        plt.close(fig)
        for fig in figs:
            plt.close(fig)
    
    # Done
    print('Done.')

