
# Import libraries
import os
import json
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import local
from config import *
from test import test_model
from utils import get_dvh, get_savename, load_checkpoint
from losses import competition_loss
from plotting import plot_losses, copy_axis


# Set up training function
def plot_model_results(
    model, dataset_test, metadata, n_show=5,
): 

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
    for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if batch_idx >= n_show:
            break

        # Send to device
        scan = scan.to(device)
        beam = beam.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        body = body.to(device)
        dose = dose.to(device)

        # Forward pass
        with torch.no_grad():
            pred = model(scan, beam, ptvs, oars, body)

        # Ignore voxels outside body
        pred = body*pred

        # Get plot data
        slice_index = scan.shape[-3] // 2
        plot_labels = ['Scan', 'Dose (Ground Truth)', 'Dose (Prediction)']
        plot_col = [
            scan[0, 0, slice_index].cpu().detach().numpy(),      # Scan
            dose[0, 0, slice_index].cpu().detach().numpy(),      # Dose Ground Truth
            pred[0, 0, slice_index].cpu().detach().numpy(),      # Dose Prediction
        ]

        # Get loss
        body_index = body.cpu().detach().numpy().astype(bool)
        loss_info = '; '.join([
            f'MAE={F.l1_loss(pred[body_index], dose[body_index]).item():.4f}',
            f'GDP_MAE={competition_loss(pred, dose, body):.4f}',
        ])

        # Append to lists
        plot_loss.append(loss_info)

        # Get DVHs
        dvh_gt_val, dvh_gt_bin = get_dvh(dose, torch.cat([ptvs, oars], dim=1))
        dvh_pred_val, dvh_pred_bin = get_dvh(pred, torch.cat([ptvs, oars], dim=1))

        # Append to lists
        plot_row.append(plot_col)
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

    # Loop over rows
    for i in range(n_rows):

        # Set up axes label
        ax[i, 0].set_ylabel(f'Example {i+1}')

        # Plot images
        for j in range(n_cols-2):
            ax[i, j].set_title(f'{plot_labels[j]}' + (f'\n{plot_loss[i]}' if j == 2 else ''))
            ax[i, j].axis('off')
            if j == 0:
                ax[i, j].imshow(plot_row[i][j], cmap='gray')
            else:
                ax[i, j].imshow(plot_row[i][j], cmap='jet', vmin=0, vmax=80)

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
    
    # Finalize plot
    plt.tight_layout()
    plt.pause(1)
    
    # Return figure
    return fig, ax

# Summarize multiple jobs
def plot_results_summary(fig_ax_list):

    # Get axes
    axs = [ax for _, ax, _ in fig_ax_list]
    losses = [loss for _, _, loss in fig_ax_list]

    # Get constants
    n_jobs = len(all_jobs)
    n_rows = axs[0].shape[0]
    n_cols = 3 + 2*n_jobs

    # Initialize figure
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    plt.ion()
    plt.show()

    # Loop over rows
    for i in range(n_rows):

        # Plot scan and ground truth Dose
        ax[i, 0] = copy_axis(axs[0][i, 0], ax[i, 0])
        ax[i, 1] = copy_axis(axs[0][i, 1], ax[i, 1])

        # Plot predicted dose for each job
        for job in range(n_jobs):
            title = '\n'.join([
                all_jobs[job].split('_')[2],
                'img_loss='+axs[job][i, 2].get_title().split('=')[-1],
                'avg_loss='+f'{losses[job]:.4f}',
            ])
            ax[i, job+2] = copy_axis(axs[job][i, 2], ax[i, job+2])
            ax[i, job+2].set_title(title)

        # Plot ground truth DVH
        ax[i, n_jobs+2].set_title('DVH (Ground Truth)')
        ax[i, n_jobs+2] = copy_axis(axs[0][i, -2], ax[i, n_jobs+2])
        ax[i, n_jobs+2].set_xlim([0, 80])

        # Plot predicted DVH for each job
        for job in range(len(all_jobs)):
            ax[i, n_jobs+3+job] = copy_axis(axs[job][i, -1], ax[i, n_jobs+3+job])
            ax[i, n_jobs+3+job].set_title(f'DVH {all_jobs[job].split("_")[2]}')
            ax[i, n_jobs+3+job].set_xlim([0, 80])

    # Finalize plot
    plt.tight_layout()
    plt.pause(1)

    # Return figure
    return fig, ax
    

# Main script
if __name__ == '__main__':

    # Set constants
    dataID = 'All'
    modelID_list = [
        ('CrossAttnUnet',     {'shape': 128}),                                           # 0
        ('CrossViT',          {'shape': 128}),                                           # 1
        ('ViT',               {'shape': 128}),                                           # 2
        ('Unet',              {'shape': 256}),                                           # 3
        ('Unet',              {'shape': 128}),                                           # 4
        ('MOECrossAttnUnet',  {'shape': 128}),                                           # 5
        ('MOECrossViT',       {'shape': 128}),                                           # 6
        ('MOEViT',            {'shape': 128}),                                           # 7
        ('MOEUnet',           {'shape': 128}),                                           # 8
        ('MOEUnet',           {'shape': 256}),                                           # 9
        ('CrossAttnUnet',     {'shape': 128, 'n_features': 16}),                         # 10
        ('CrossAttnUnet',     {'shape': 256, 'n_features': 4, 'use_checkpoint': True}),  # 11
    ]
    all_jobs = [get_savename(dataID, modelID, **model_kwargs) for modelID, model_kwargs in modelID_list]
    all_jobs = [j for j in all_jobs if os.path.exists(os.path.join(PATH_OUTPUT, f'{j}.pth'))]

    # Plot each job separately
    data_list = []
    for savename in all_jobs:

        # Load checkpoint
        checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
        model, datasets, metadata = load_checkpoint(checkpoint_path)
        loss_test = metadata['train_stats']['loss_test']
        losses_train = metadata['train_stats']['losses_train']
        losses_val = metadata['train_stats']['losses_val']

        # Plot losses
        fig, ax = plot_losses(losses_train, losses_val)
        fig.suptitle(f'{savename} - Test Loss: {loss_test:.4f}')
        fig.savefig(f'figs/{savename}_losses.png')
        plt.close()  # Close figure

        # Test model
        dataset_test = datasets[2]
        fig, ax = plot_model_results(model, dataset_test, metadata)
        fig.savefig(f'figs/{savename}_test.png')
        data_list.append((fig, ax, loss_test))

    # Plot summary
    fig, ax = plot_results_summary(data_list)
    fig.savefig(f'figs/summary_{dataID}.png')

    # Close figures
    plt.close(fig)
    for fig, ax, _ in data_list:
        plt.close(fig)
    
    # Done
    print('Done.')

