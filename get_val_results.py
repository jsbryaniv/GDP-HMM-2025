
# Import libraries
import os
import sys
import json
import time
import torch

# Import custom classes
from test import test_model
from train import train_model
from utils import load_model_and_datasets, resize_image_3d, reverse_resize_3d

# Get config
with open('config.json', 'r') as f:
    config = json.load(f)
path_output = config['PATH_OUTPUT']


# Define dose predictor
def get_val_results(dataset, model):

    # Loop over dataset
    pass


# Run main function
if __name__ == '__main__':

    # Get model and dataset

    # Done
    print('Done!')
    
