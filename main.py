
# Import libraries
import os
import sys
import json
import time
import copy
import torch

# Import local
from config import *
from test import test_model
from train import train_model
from utils import get_savename, save_checkpoint, load_checkpoint, initialize_model, initialize_datasets


# Define main function
def main(
    dataID, modelID, model_kwargs=None,
    from_checkpoint=False, debug=False,
):
    """
    Main function to train a model on a dataset.
    """
    print(f"Running main function for model {modelID} on dataset {dataID}.")

    # Check inputs
    if model_kwargs is None:
        model_kwargs = {}
        
    # Get save info
    savename = get_savename(dataID, modelID, **model_kwargs)
    checkpoint_path = os.path.join(PATH_OUTPUT, f'{savename}.pth')
    print(f"-- savename={savename}")
    
    # If continuing training, load previous files
    if from_checkpoint:
        # Load previous dataset and model
        for _ in range(10):
            print("WARNING: Be aware job is running with from_checkpoint=True.")  # Print warning message
        print("Loading checkpoint.")
        model, datasets, optimizer, metadata = load_checkpoint(checkpoint_path)  # Load model, datasets, and metadata
        dataset_val, dataset_test, dataset_train = datasets                      # Unpack datasets
    else:
        # Initialize datasets and model
        print("Initializing datasets and model.")
        datasets = initialize_datasets(dataID)                         # Initialize datasets
        dataset_val, dataset_test, dataset_train = datasets            # Unpack datasets
        n_channels = dataset_train.dataset.n_channels                  # Get number of channels
        model = initialize_model(modelID, n_channels, **model_kwargs)  # Initialize model
        optimizer = None                                               # Initialize optimizer
        metadata = {                                                   # Initialize metadata
            'dataID': dataID,
            'modelID': modelID,
            'model_kwargs': model_kwargs,
            'train_stats': {
                'epoch': 0,
                'losses_val': [],
                'losses_train': [],
                'loss_val_best': float('inf'),
                'model_state_dict_best': None,
                'loss_test': None,
            },
        }

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        device = torch.device('cpu')
    model.to(device)


    ### TRAINING LOOP ###
    print("Starting training.")

    # Train model
    model, optimizer, train_stats_new = train_model(
        model, 
        datasets=(dataset_train, dataset_val), 
        optimizer=optimizer,
        jobname=savename, debug=debug,
        epoch_start=metadata['train_stats']['epoch'],
        loss_val_best=metadata['train_stats']['loss_val_best'],
        model_state_dict_best=metadata['train_stats']['model_state_dict_best'],
    )

    ### TESTING ###
    print("Testing model.")

    # Test model
    model_best = copy.deepcopy(model)
    model_best.load_state_dict(metadata['train_stats']['model_state_dict_best'])
    loss_test = test_model(model, dataset_test, debug=debug)


    ### SAVE RESULTS ###
    print("Saving results.")

    # Merge training statistics
    train_stats_new['loss_test'] = loss_test
    train_stats_new['losses_val'] = metadata['train_stats']['losses_val'] + train_stats_new['losses_val']
    train_stats_new['losses_train'] = metadata['train_stats']['losses_train'] + train_stats_new['losses_train']
    metadata['train_stats'] = train_stats_new

    # Save checkpoint
    save_checkpoint(
        checkpoint_path,
        model=model, 
        datasets=datasets, 
        optimizer=optimizer, 
        metadata=metadata
    )


    ### DONE ###
    print(f"Finished running job for {modelID} on dataset {dataID}.")

    # Return model and training statistics
    return model, metadata


# Run main function
if __name__ == '__main__':

    # Set up all jobs
    dataIDs_list = ['All']
    modelID_list = [
        ('CrossAttnUnet',   {'shape': 128}),                                           # 0
        ('CrossViT',        {'shape': 128}),                                           # 1
        ('ViT',             {'shape': 128}),                                           # 2
        ('Unet',            {'shape': 256}),                                           # 3
        ('Unet',            {'shape': 128}),                                           # 4
        ('MOECrossAttnUnet',   {'shape': 128}),                                        # 5
        ('MOECrossViT',        {'shape': 128}),                                        # 6
        ('MOEViT',             {'shape': 128}),                                        # 7
        ('MOEUnet',            {'shape': 128}),                                        # 8
        ('MOEUnet',            {'shape': 256}),                                        # 9
        ('CrossAttnUnet',   {'shape': 128, 'n_features': 16}),                         # 10
        ('CrossAttnUnet',   {'shape': 256, 'n_features': 4, 'use_checkpoint': True}),  # 11
    ]
    all_jobs = []
    for dataID in dataIDs_list:
        for (modelID, model_kwargs) in modelID_list:
            all_jobs.append({
                'dataID': dataID, 
                'modelID': modelID,
                'model_kwargs': model_kwargs,
            })
    
    # Get training IDs from system arguments
    ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # # DEBUGGING all files
    # for ID in range(len(all_jobs)):
    #     for ITER in [0, 1]:
    #         job_args = copy.deepcopy(all_jobs[ID])
    #         shape = job_args['model_kwargs'].get('shape', None)
    #         if shape is not None:
    #             # Make shape smaller for debugging
    #             job_args['model_kwargs']['shape'] = shape // 2
    #         model, metadata = main(**job_args, from_checkpoint=bool(ITER > 0), debug=True)
    #         print('\n'*5)

    # Run main function
    job_args = all_jobs[ID]
    model, metadata = main(**job_args, from_checkpoint=bool(ITER > 0))

    # Done
    print('Done!')
    
