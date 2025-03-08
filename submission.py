
# Import libraries
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import custom classes
from dataset import GDPDataset
from utils import load_model_and_datasets, resize_image_3d, reverse_resize_3d

# Get config
with open('config.json', 'r') as f:
    config = json.load(f)
path_output = config['PATH_OUTPUT']


# Define dose predictor
def package_results(model, model_type=None, shape=None):

    # Set model to evaluation mode
    model.eval()
    
    # Get constants
    device = next(model.parameters()).device
    if isinstance(shape, int):
        shape = (shape, shape, shape)

    # Initialize results folder
    path_results = os.path.join(path_output, 'val_results')  # Define path
    if not os.path.exists(path_results):                     # Create path if it does not exist
        os.makedirs(path_results)
    for file in os.listdir(path_results):                    # Remove files in path
        os.remove(os.path.join(path_results, file))

    # Initialize dataset and savenames
    dataset = GDPDataset('All', validation_set=True)
    filenames = [os.path.basename(f) for f in dataset.files]

    # Loop over dataset
    print('Getting validation results...')
    for i, (ct, beam, ptvs, oars, body) in enumerate(dataset):
        print(f'--{i}/{len(dataset)}')

        # Format data
        scan0 = scan.clone()  # Copy scan for visualization
        if shape is not None:
            scan, transform_params = resize_image_3d(scan, shape)
            beam, _ = resize_image_3d(beam, shape)
            ptvs, _ = resize_image_3d(ptvs, shape)
            oars, _ = resize_image_3d(oars, shape)
            body, _ = resize_image_3d(body, shape)
        scan = scan.unsqueeze(0).to(device)
        beam = beam.unsqueeze(0).to(device)
        ptvs = ptvs.unsqueeze(0).to(device)
        oars = oars.unsqueeze(0).to(device)
        body = body.unsqueeze(0).to(device)

        # Get prediction
        if model_type is None:
            # All in one
            x = torch.cat([scan, beam, ptvs, oars, body], dim=1)
            pred = model(x)
        elif model_type == 'crossae':
            # Cross attention
            x = torch.cat([beam, ptvs], dim=1).clone()
            y_list = [
                scan, 
                torch.cat([beam, ptvs], dim=1), 
                torch.cat([oars, body], dim=1),
            ]
            pred = model(x, y_list)
        
        # Reverse formatting
        pred = pred.squeeze(0)
        if shape is not None:
            pred = reverse_resize_3d(pred, transform_params)
        pred = pred.detach().cpu().numpy()

        # Plot
        fig, ax = plt.subplots(1, 2)
        plt.ion()
        plt.show()
        ax[0].imshow(scan0[0, shape[0]//2, :, :])
        ax[1].imshow(pred[0, shape[0]//2, :, :])
        plt.tight_layout()
        plt.pause(0.1)
        plt.savefig('_image.png')
        plt.close()

        # Save prediction
        filename = filenames[i]
        np.save(os.path.join(path_results, f'{filename[:-4]}_pred.npy'), pred)

        # Open prediction and ensure it was saved correctly
        pred_new = np.load(os.path.join(path_results, f'{filename[:-4]}_pred.npy'))
        assert np.allclose(pred, pred_new), 'Prediction was not saved correctly!'

    # Zip results folder
    full_path = os.path.abspath(path_results)
    print(f'Zipping results folder at {full_path}...')
    os.system(f'cd "{full_path}" && zip -r ../results.zip ./*_pred.npy')

    # Done
    print('Done!')
    return path_results


# Run main function
if __name__ == '__main__':

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get model and metadata
    savename = 'model_All_Unet'
    model, _, metadata = load_model_and_datasets(savename)
    shape = metadata['data_metadata']['shape']
    model.to(device)

    # Get validation results
    path_results = package_results(model, shape=shape)

    # Done
    print('Done!')
    
