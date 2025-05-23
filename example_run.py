
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
from dataset import GDPDataset



# Run main function
if __name__ == '__main__':

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get model
    checkpoint_path = 'outfiles/model_All_crossunet_shape=128_best.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_state_dict = checkpoint['metadata']['model_state_dict_best']
    model = DosePredictionModel.from_checkpoint(checkpoint_path, model_state_dict=model_state_dict)
    model.to(device)
    model.eval()

    # Create dataset
    dataset = GDPDataset(treatment='All', validation_set=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Set up figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    plt.show()

    # Loop over dataset
    print('Getting test results...')
    for i, (scan, beam, ptvs, oars, body, dose) in enumerate(dataloader):

        # Format data
        scan = scan.to(device)
        beam = beam.to(device)
        ptvs = ptvs.to(device)
        oars = oars.to(device)
        body = body.to(device)
        dose = dose.to(device)

        # Get prediction
        pred = model(scan, beam, ptvs, oars, body, d97=True)

        # Plot results
        z = scan.shape[2] // 2
        z = np.where(ptvs.detach().cpu().numpy() == ptvs.max().item())[2][0]
        ax[0].cla()
        ax[0].set_title('Scan')
        ax[0].imshow(scan[0, 0, z].detach().cpu().numpy(), cmap='gray')
        ax[1].cla()
        ax[1].set_title('Prediction')
        ax[1].imshow(pred[0, 0, z].detach().cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.pause(0.1)

    # Done
    print('Done!')
    