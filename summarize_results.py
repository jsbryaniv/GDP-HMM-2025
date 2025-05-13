
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
from utils import get_dvh, get_savename, load_checkpoint
from losses import competition_loss
from plotting import plot_losses, plot_loss_histogram, copy_axis


# Set up training function
def plot_model_results(model, dataset_test, metadata, n_show=5,): 

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
        scan, beam, ptvs, oars, body, dose = [
            x.to(device) for x in (scan, beam, ptvs, oars, body, dose)
        ]

        # Forward pass
        with torch.no_grad():
            pred = model(scan, beam, ptvs, oars, body).detach()

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
    n_jobs = len(all_savenames)
    n_rows = axs[0].shape[0]
    n_cols = 3 + 2*n_jobs
    # n_cols = 2 + n_jobs

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
                '_'.join(all_savenames[job].split('_')[2:]),
                # all_savenames[job].split('_')[2],
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
        for job in range(len(all_savenames)):
            ax[i, n_jobs+3+job] = copy_axis(axs[job][i, -1], ax[i, n_jobs+3+job])
            # ax[i, n_jobs+3+job].set_title(f'DVH\n{"_".join(all_savenames[job].split("_")[2:])}')
            ax[i, n_jobs+3+job].set_title(f'DVH {all_savenames[job].split("_")[2]}')
            ax[i, n_jobs+3+job].set_xlim([0, 80])

        # Remove ticks
        for j in range(n_cols):
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # Finalize plot
    plt.tight_layout()
    plt.pause(1)

    # Return figure
    return fig, ax
    

# Main script
if __name__ == '__main__':
    
    # Set up all jobs
    suptitle = f""
    # dataIDs_list = ['All']
    # modelID_list = [
    #     ('diffunet',  {'shape': 128, 'max_batches': 100}),
    #     ('crossunet', {'shape': 128}),
    #     ('unet',      {'shape': 128, 'scale': 1}),
    #     ('unet',      {'shape': 128, 'scale': 2}),
    #     ('unet',      {'shape': 256, 'scale': 4}),
    # ]
    # all_jobs = []
    # for dataID in dataIDs_list:
    #     for (modelID, kwargs) in modelID_list:
    #         kwargs = {k: v for k, v in kwargs.items() if not k in ['batch_size', 'max_batches']}  # Remove training args
    #         all_jobs.append({'dataID': dataID, 'modelID': modelID, **kwargs})

    # # Convert to savenames
    # all_savenames = [get_savename(**args) for args in all_jobs]
    all_savenames = [
        "model_All_crossunet_shape=128_best",
        "model_diffunet_shape=128",
        "model_unet_scale=2_shape=128",
    ]
    all_savenames = [j for j in all_savenames if os.path.exists(os.path.join(PATH_OUTPUT, f'{j}.pth'))]

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Plot each job separately
    data_list = []
    for savename in all_savenames:
        print(savename)

        # Load checkpoint
        checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
        model, datasets, optimizer, metadata = load_checkpoint(checkpoint_path, load_best=True)
        loss_test = metadata['loss_test']
        losses_test = metadata['losses_test']
        losses_test = metadata['losses_test_d97']
        losses_train = metadata['losses_train']
        losses_val = metadata['losses_val']

        # Send model to device
        model.to(device)

        # Plot losses
        fig, ax = plot_losses(Train=losses_train, Val=losses_val)
        fig.suptitle(f'{savename} - Test Loss: {loss_test:.4f}')
        fig.savefig(f'figs/{savename}_losses.png')
        plt.close()  # Close figure

        # Histogram of losses
        fig, ax = plot_loss_histogram(LossOnTest=losses_test)
        fig.suptitle(f'{savename} - Test Loss Histogram')
        fig.savefig(f'figs/{savename}_loss_histogram.png')
        plt.close()

        # Test model
        dataset_test = datasets[2]
        fig, ax = plot_model_results(model, dataset_test, metadata)
        fig.savefig(f'figs/{savename}_test.png')
        data_list.append((fig, ax, loss_test))

    # Plot summary
    fig, ax = plot_results_summary(data_list)
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.pause(.1)
    fig.savefig(f'figs/summary.png')

    # Close figures
    plt.close(fig)
    for fig, ax, _ in data_list:
        plt.close(fig)
    
    # Done
    print('Done.')


    