
# Import libraries
import os
import json
import copy
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import local
from config import *


# Set up testing function
@torch.no_grad()
def test_model(model, dataset_test, jobname=None, print_every=100, debug=False):

    # Set up constants
    if jobname is None:
        jobname = ''
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set up model
    model.eval()

    # Set up data loader
    dataset_test = copy.deepcopy(dataset_test)  # Copy dataset
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Open competition metadata
    metadata = pd.read_csv(os.path.join(PATH_METADATA, 'meta_data.csv'))

    # Initialize loss
    losses_test = []

    # Loop over batches
    print(f'Testing model with {n_parameters} parameters on {device}.')
    for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if debug and batch_idx > 2:
            print('DEBUG MODE: Breaking early.')
            break
        
        # Configure inputs
        scan, beam, ptvs, oars, body, dose = [
            x.to(device) for x in (scan, beam, ptvs, oars, body, dose)
        ]


        # Forward pass
        pred = model(scan, beam, ptvs, oars, body)
        pred = pred.squeeze().cpu().numpy()

        # Get reference dose
        idx = dataset_test.indices[batch_idx]
        filepath = dataset_test.dataset.files[idx]
        patient_id = os.path.basename(filepath).split('+')[0]
        plan_id = os.path.basename(filepath).split('+')[1]
        patient_df = metadata.loc[(metadata['PID'] == patient_id) & (metadata['PlanID'] == plan_id)]
        PTVHighname = 'PTVHighOPT' if patient_df['site'].item() == 1 else 'PTV'
        data_npz = np.load(filepath, allow_pickle=True)
        data_dict = dict(data_npz)['arr_0'].item()
        scale_dose_Dict = json.load(open(os.path.join(PATH_METADATA, 'PTV_DICT.json')))
        ref_dose = data_dict['dose'] * data_dict['dose_scale']
        ptv_highdose =  scale_dose_Dict[patient_id]['PTV_High']['PDose']
        norm_scale = ptv_highdose / (np.percentile(ref_dose[data_dict[PTVHighname].astype('bool')], 3) + 1e-5)
        ref_dose = ref_dose * norm_scale

        # Calculate error
        isodose_5Gy_mask = ((ref_dose > 5) | (pred > 5)) & (data_dict['Body'] > 0) 
        isodose_ref_5Gy_mask = (ref_dose > 5) & (data_dict['Body'] > 0) 
        diff = ref_dose - pred
        error = np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)

        # Update loss
        losses_test.append(error)
        
        # Status update
        if batch_idx % print_every == 0:
            print(f'-- Batch {batch_idx}/{len(loader_test)} error={error:.4f}')


    # Normalize loss
    loss_test = np.mean(losses_test)

    # Print loss
    print(f'-- Average loss on test dataset: {loss_test:.4f} {jobname}') 
    
    # Return total loss
    return loss_test, losses_test


# Plot worst predictions
def show_worst(model, dataset_test, losses_test=None, n_show=5):

    # Import libraries
    import matplotlib.pyplot as plt

    # Get constants
    device = next(model.parameters()).device

    # Get worst indices
    if losses_test is None:
        _, losses_test = test_model(model, dataset_test)
    worst_indices = np.argsort(losses_test)[::-1]
    worst_indices = worst_indices[:n_show]

    # Set up figure
    fig, ax = plt.subplots(len(worst_indices), 6, figsize=(16, 20)) # 6 columns: Axial GT, Axial PRED, Coronal GT, Coronal PRED, Sagittal GT, Sagittal PRED
    plt.ion()
    plt.show()

    # Loop over worst indices
    for ax_idx, data_idx in enumerate(worst_indices):

        # Get data
        scan, beam, ptvs, oars, body, dose = dataset_test[data_idx]
        scan = scan.unsqueeze(0).to(device)
        beam = beam.unsqueeze(0).to(device)
        ptvs = ptvs.unsqueeze(0).to(device)
        oars = oars.unsqueeze(0).to(device)
        body = body.unsqueeze(0).to(device)
        dose = dose.unsqueeze(0).to(device)

        # Forward pass
        pred = model(scan, beam, ptvs, oars, body)

        # Convert to numpy
        scan = scan.squeeze().detach().cpu().numpy()
        beam = beam.squeeze().detach().cpu().numpy()
        ptvs = ptvs.squeeze().detach().cpu().numpy()
        oars = oars.squeeze().detach().cpu().numpy()
        body = body.squeeze().detach().cpu().numpy()
        dose = dose.squeeze().detach().cpu().numpy()
        pred = pred.squeeze().detach().cpu().numpy()
        
        # Calculate error
        # isodose_5Gy_mask = ((ref_dose > 5) | (pred > 5)) & (data_dict['Body'] > 0) 
        # isodose_ref_5Gy_mask = (ref_dose > 5) & (data_dict['Body'] > 0) 
        # diff = ref_dose - pred
        isodose_5Gy_mask = ((dose > 5) | (pred > 5)) & (body > 0)
        isodose_ref_5Gy_mask = (dose > 5) & (body > 0)
        diff = dose - pred
        error = np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)

        # Find worst slices
        worst_ijk = np.unravel_index(np.abs((dose - pred)*body).argmax(), dose.shape)
        vmin = min(dose.min(), pred.min())
        vmax = max(dose.max(), pred.max())
        # dose[worst_ijk] = 0
        # pred[worst_ijk] = 0

        # Plot axial
        ax[ax_idx, 0].set_title('Axial GT')
        ax[ax_idx, 1].set_title('Axial PRED')
        ax[ax_idx, 0].imshow(dose[worst_ijk[0], :, :], vmin=vmin, vmax=vmax)
        ax[ax_idx, 1].imshow(pred[worst_ijk[0], :, :], vmin=vmin, vmax=vmax)
        ax[ax_idx, 0].plot(worst_ijk[2], worst_ijk[1], 'ro')
        ax[ax_idx, 1].plot(worst_ijk[2], worst_ijk[1], 'ro')

        # Plot coronal
        ax[ax_idx, 2].set_title('Coronal GT')
        ax[ax_idx, 3].set_title('Coronal PRED')
        ax[ax_idx, 2].imshow(dose[:, worst_ijk[1], :], vmin=vmin, vmax=vmax)
        ax[ax_idx, 3].imshow(pred[:, worst_ijk[1], :], vmin=vmin, vmax=vmax)
        ax[ax_idx, 2].plot(worst_ijk[2], worst_ijk[0], 'ro')
        ax[ax_idx, 3].plot(worst_ijk[2], worst_ijk[0], 'ro')

        # Plot sagittal
        ax[ax_idx, 4].set_title('Sagittal GT')
        ax[ax_idx, 5].set_title('Sagittal PRED')
        ax[ax_idx, 4].imshow(dose[:, :, worst_ijk[2]], vmin=vmin, vmax=vmax)
        ax[ax_idx, 5].imshow(pred[:, :, worst_ijk[2]], vmin=vmin, vmax=vmax)
        ax[ax_idx, 4].plot(worst_ijk[1], worst_ijk[0], 'ro')
        ax[ax_idx, 5].plot(worst_ijk[1], worst_ijk[0], 'ro')

        # Set up y-label
        ax[ax_idx, 0].set_ylabel(f'Patient {data_idx}\nloss={losses_test[data_idx]:.4f}')

    # Finalize figure
    plt.tight_layout()
    plt.pause(.1)

    # Return fig and ax
    return fig, ax
        


# Test model
if __name__ == '__main__':

    # Import libraries
    import matplotlib.pyplot as plt
    from utils import load_checkpoint

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and dataset
    savename = 'model_All_diffunet_shape=128'
    checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
    model, datasets, _, metadata = load_checkpoint(checkpoint_path, load_best=True)
    dataset_val, dataset_test, dataset_train = datasets
    losses_test = metadata['losses_test']
    model.to(device)

    # Show worst predictions
    fig, ax = show_worst(model, dataset_test, losses_test=losses_test)
    plt.savefig('_image.png')
    plt.close()

    # # Test model
    # loss_test = test_model(model, datasets[-1], print_every=1)
    
    # Done
    print('Done.')




