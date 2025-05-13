
# Import libraries
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import local
from model import DosePredictionModel
from dataset import GDPTestDataset


# Define dose predictor
@torch.no_grad()
def package_results(model, path_data, path_results):

    # Set model to evaluation mode
    model.eval()
    
    # Get constants
    device = next(model.parameters()).device

    # Remove old files from results folder
    for file in os.listdir(path_results):
        os.remove(os.path.join(path_results, file))

    # Initialize dataset and savenames
    dataset = GDPTestDataset(path_data)
    filenames = [os.path.basename(f) for f in dataset.files]

    # Loop over dataset
    print('Getting test results...')
    for i, (scan, beam, ptvs, oars, body) in enumerate(dataset):

        # Get file name 
        filename = filenames[i]
        print(f'--{i}/{len(dataset)} {filenames[i]}')

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

        # Save prediction
        np.save(os.path.join(path_results, f'{filename[:-4]}_pred.npy'), pred)

        # Open prediction and ensure it was saved correctly
        pred_new = np.load(os.path.join(path_results, f'{filename[:-4]}_pred.npy'))
        assert np.allclose(pred, pred_new), 'Prediction was not saved correctly!'

    # Done
    print('Done!')
    return path_results


# Run main function
if __name__ == '__main__':

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get system arguments
    path_data = sys.argv[1] if len(sys.argv) > 1 else '../data/'
    path_results = sys.argv[2] if len(sys.argv) > 2 else '../results/'

    # Print status
    print('Generating predictions for test set')
    print(f'-- Path data:    {path_data}')
    print(f'-- Path results: {path_results}')
    print(f'-- Device:       {device}')

    # Get model
    checkpoint_path = 'model_All_crossunet_shape=128_best.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_state_dict = checkpoint['metadata']['model_state_dict_best']
    model = DosePredictionModel.from_checkpoint(checkpoint_path, model_state_dict=model_state_dict)
    model.to(device)

    # Get validation results
    path_results = package_results(model, path_data, path_results)

    # Done
    print('Done!')
    