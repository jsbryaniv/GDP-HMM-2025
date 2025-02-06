
# Import libraries
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import custom classes
from train import train_model
from plotting import plot_losses, plot_prediction
from dataset import GDPDataset
from models.unet import Simple3DUnet

# Set environment
with open('config.json', 'r') as f:
    config = json.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### DATASET ###
print("Loading dataset.")

# Load dataset and split into train, validation, and test sets
dataset = GDPDataset(
    treatment='HaN',
    path_data='data/han/train',
    path_dose_dict='data/PTV_DICT.json',
    return_dose=True,
)
test_size = int(0.2 * len(dataset))
dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
    dataset,
    [len(dataset) - 2*test_size, test_size, test_size]
 )


### MODEL ###
print("Setting up model.")

# Initialize model and optimizer
model = Simple3DUnet(
    in_channels=35, out_channels=1, 
    n_features=8, n_blocks=3, n_layers_per_block=3
)
model.to(device)


### TRAINING LOOP ###
print("Starting training.")

# Train model
model, training_statistics = train_model(
    model, dataset_train, dataset_val,
    batch_size=1, learning_rate=0.001, num_epochs=20,
)

# Unpack training statistics
losses_train = training_statistics['losses_train']
losses_val = training_statistics['losses_val']

# Save model
torch.save(model.state_dict(), 'outfiles/model.pth')


### PLOTTING ###
print("Plotting results.")

# Get prediction on a single scan
ct, ptvs, oars, beam, dose = dataset_test[0]
ct = ct.unsqueeze(0).to(device)
ptvs = ptvs.unsqueeze(0).to(device)
oars = oars.unsqueeze(0).to(device)
beam = beam.unsqueeze(0).to(device)
dose = dose.unsqueeze(0).to(device)
with torch.no_grad():
    x = torch.cat([ct, ptvs, oars, beam], dim=1)
    pred = model(x)

# Plot results of a single slice
plot_prediction(ct, pred, dose)

# Plot training and validation losses
plot_losses(losses_train, losses_val)

# Done
print('Done.')

