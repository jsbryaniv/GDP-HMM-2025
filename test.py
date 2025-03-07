
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

# Import custom libaries
from losses import competition_loss
from dataset import GDPDataset
from utils import resize_image_3d, reverse_resize_3d

# Get config 
with open('config.json', 'r') as f:
    config = json.load(f)
ROOT_DIR = config['PATH_OUTPUT']
PATH_METADATA = config['PATH_METADATA']

# Set up testing function
@torch.no_grad()
def test_model(model, dataset_test, debug=False):

    # Set up constants
    device = next(model.parameters()).device
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set up model
    model.eval()

    # Set up data loader
    dataset_test = copy.deepcopy(dataset_test)  # Copy dataset
    shape = dataset_test.dataset.shape          # Get shape
    dataset_test.dataset.shape = None           # Remove shape
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Open competition metadata
    metadata = pd.read_csv(os.path.join(PATH_METADATA, 'meta_data.csv'))

    # Initialize loss
    loss_test = 0
    n_test = 0
    MAE_list = []

    # Loop over batches
    print(f'Testing model with {n_parameters} parameters on {device}.')
    for batch_idx, (ct, beam, ptvs, oars, body, dose) in enumerate(loader_test):
        if debug and batch_idx > 10:
            print('DEBUG MODE: Breaking early.')
            break
        
        # Status update
        if batch_idx % 100 == 0:
            print(f'-- Batch {batch_idx}/{len(loader_test)}')
        
        # Configure inputs
        if shape is not None:
            dose0 = dose.clone()
            ct, transform_params = resize_image_3d(ct.squeeze(0), shape, fill_value=ct.min())
            beam, _ = resize_image_3d(beam.squeeze(0), shape)
            ptvs, _ = resize_image_3d(ptvs.squeeze(0), shape)
            oars, _ = resize_image_3d(oars.squeeze(0), shape)
            body, _ = resize_image_3d(body.squeeze(0), shape)
            dose, _ = resize_image_3d(dose.squeeze(0), shape)
            ct = ct.unsqueeze(0)
            beam = beam.unsqueeze(0)
            ptvs = ptvs.unsqueeze(0)
            oars = oars.unsqueeze(0)
            body = body.unsqueeze(0)
            dose = dose.unsqueeze(0)
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

        # Configure outputs
        prediction = pred.clone()                                     # Copy prediction
        prediction = prediction.squeeze(0)                            # Remove batch dimension
        prediction = reverse_resize_3d(prediction, transform_params)  # Reshape back to original size
        prediction = prediction.squeeze(0)                            # Remove channel dimension
        prediction = prediction.cpu().numpy()                         # Convert to numpy array

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
        ref_body = data_dict['Body']

        # Calculate loss
        loss = competition_loss(pred, dose, body)

        # Update loss
        loss_test += loss.item()
        n_test += 1

        # Calculate error
        isodose_5Gy_mask = ((ref_dose > 5) | (prediction > 5)) & (data_dict['Body'] > 0) 
        isodose_ref_5Gy_mask = (ref_dose > 5) & (data_dict['Body'] > 0) 
        diff = ref_dose - prediction
        error = np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
        MAE_list.append(error)

        # Print status
        print(f'Batch index: {batch_idx}, Index: {idx}')
        print(f'Loss: {loss.item()}')
        print(f'Error: {error}')
        print(f'filepath: {filepath}')

        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 5)
        plt.ion()
        plt.show()
        ax[0].set_title('ct')
        ax[0].imshow(ct[0, 0, ct.shape[2] // 2].cpu().detach().numpy())
        ax[1].set_title('dose')
        ax[1].imshow(dose[0, 0, dose.shape[2] // 2].cpu().detach().numpy())
        ax[2].set_title('pred')
        ax[2].imshow(pred[0, 0, pred.shape[2] // 2].cpu().detach().numpy())
        ax[3].set_title('ref_dose')
        ax[3].imshow(ref_dose[ref_dose.shape[0] // 2])
        ax[4].set_title('prediction')
        ax[4].imshow(prediction[prediction.shape[0] // 2])
        plt.tight_layout()
        plt.pause(.1)
        plt.savefig('_image.png')
        plt.close()


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
    device = 'cpu'

    # Load model and datasets
    savename = 'model_All_Unet'
    model, datasets, metadata = load_model_and_datasets(savename)
    model.to(device)

    # Test model
    loss_test = test_model(model, datasets[-1])
    
    # Done
    print('Done.')
