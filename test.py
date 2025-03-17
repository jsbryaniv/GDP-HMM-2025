
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
    n_test = 0
    loss_test = 0

    # Loop over batches
    print(f'Testing model with {n_parameters} parameters on {device}.')
    for batch_idx, (scan, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if debug and batch_idx > 10:
            print('DEBUG MODE: Breaking early.')
            break
        
        # Configure inputs
        scan = scan.to(device)
        beam = beam.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        body = body.to(device)
        dose = dose.to(device)

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
        n_test += 1
        loss_test += error
        
        # Status update
        if batch_idx % print_every == 0:
            print(f'-- Batch {batch_idx}/{len(loader_test)} error={error:.4f}')


    # Normalize loss
    loss_test /= n_test

    # Print loss
    print(f'-- Average loss on test dataset: {loss_test:.4f} {jobname}') 
    
    # Return total loss
    return loss_test


# Test model
if __name__ == '__main__':

    # Import custom libraries
    from utils import load_checkpoint

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Load model and dataset
    savename = 'model_All_CrossAttnUnet_shape=128'
    checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
    model, datasets, metadata = load_checkpoint(checkpoint_path)
    model.to(device)

    # Test model
    loss_test = test_model(model, datasets[-1], print_every=1)
    
    # Done
    print('Done.')




