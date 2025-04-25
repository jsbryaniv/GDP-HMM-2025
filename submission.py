
# Import libraries
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import local
from config import *
from dataset import GDPDataset
from utils import load_checkpoint


# Define dose predictor
@torch.no_grad()
def package_results(model):

    # Set model to evaluation mode
    model.eval()
    
    # Get constants
    device = next(model.parameters()).device

    # Initialize results folder
    path_results = os.path.join(PATH_OUTPUT, 'val_results')  # Define path
    if not os.path.exists(path_results):                     # Create path if it does not exist
        os.makedirs(path_results)
    for file in os.listdir(path_results):                    # Remove files in path
        os.remove(os.path.join(path_results, file))

    # Initialize dataset and savenames
    dataset = GDPDataset('All', validation_set=True)
    filenames = [os.path.basename(f) for f in dataset.files]

    # Loop over dataset
    print('Getting validation results...')
    for i, (scan, beam, ptvs, oars, body, dose) in enumerate(dataset):
        print(f'--{i}/{len(dataset)}')

        # Format data
        scan = scan.unsqueeze(0).to(device)
        beam = beam.unsqueeze(0).to(device)
        ptvs = ptvs.unsqueeze(0).to(device)
        oars = oars.unsqueeze(0).to(device)
        body = body.unsqueeze(0).to(device)

        # Get prediction
        pred = model(scan, beam, ptvs, oars, body, d97=True)

        # Unformat prediction and data
        pred = pred.squeeze(0).cpu().numpy()
        scan = scan.squeeze(0).cpu().numpy()
        beam = beam.squeeze(0).cpu().numpy()
        ptvs = ptvs.squeeze(0).cpu().numpy()
        oars = oars.squeeze(0).cpu().numpy()
        body = body.squeeze(0).cpu().numpy()

        # Plot
        fig, ax = plt.subplots(1, 2)
        plt.ion()
        plt.show()
        fig.suptitle(f'Prediction for {filenames[i]}')
        z_slice = scan.shape[2] // 2
        ax[0].imshow(scan[0, z_slice], cmap='gray')
        ax[1].imshow(pred[0, z_slice], cmap='hot')
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
    savename = 'model_All_crossunet_shape=128'
    checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
    model, _, _, _ = load_checkpoint(checkpoint_path, load_best=True)
    model.to(device)

    # Get validation results
    path_results = package_results(model)

    # Done
    print('Done!')
    