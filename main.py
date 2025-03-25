
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
    dataID, modelID, batch_size=None,
    from_checkpoint=False, debug=False, device=None,
    **model_kwargs
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
        for _ in range(5):
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
        }

    # Move model and optimizer to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Get device
    model.to(device)                                                           # Move model to device
    if optimizer is not None:                                                  # Move optimizer to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    ### TRAINING LOOP ###
    print("Starting training.")

    # Get training progress
    epoch_start = metadata.get('epoch', 0)
    loss_val_best = metadata.get('loss_val_best', float('inf'))
    model_state_dict_best = metadata.get('model_state_dict_best', None)

    # Train model
    model, optimizer, train_stats = train_model(
        model, 
        datasets=(dataset_train, dataset_val), 
        optimizer=optimizer,
        batch_size=batch_size,
        jobname=savename, debug=debug,
        epoch_start=epoch_start, loss_val_best=loss_val_best, model_state_dict_best=model_state_dict_best,
    )

    # Merge training statistics
    for key, value in train_stats.items():
        if isinstance(value, list):
            value_old = metadata.get(key, [])
            train_stats[key] = value_old + value
    metadata.update(train_stats)

    ### TESTING ###
    print("Testing model.")

    # Test model
    model_best = copy.deepcopy(model)
    model_best.load_state_dict(metadata['model_state_dict_best'])
    loss_test, losses_test = test_model(model_best, dataset_test, debug=debug)
    metadata['loss_test'] = loss_test
    metadata['losses_test'] = losses_test


    ### SAVE RESULTS ###
    print("Saving results.")

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

    # Return best model and training statistics
    return model_best, metadata


# Run main function
if __name__ == '__main__':
    
    # Set up all jobs
    dataIDs_list = ['All']
    modelID_list = [
        ('unet',           {'batch_size': 2, 'shape': 128}),
        ('crossunet',      {'batch_size': 2, 'shape': 128}),
        ('crossunetlight', {'batch_size': 2, 'shape': 128}),
        ('diffunet',       {'batch_size': 3, 'shape': 128}),
        ('diffunetlight',  {'batch_size': 4, 'shape': 128}),
        # ('diffvit',        {'batch_size': 2, 'shape': 128}),
        # ('diffvitlight',   {'batch_size': 2, 'shape': 128}),
    ]
    all_jobs = []
    for dataID in dataIDs_list:
        for (modelID, kwargs) in modelID_list:
            all_jobs.append({'dataID': dataID, 'modelID': modelID, **kwargs})
    
    # Get training IDs from system arguments
    ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Check machine
    if MACHINE == 'carina@mca':
        """Carina is for debugging."""

        # Debug main function
        for ID in range(len(all_jobs)):
            for ITER in [0, 1]:
                job_args = copy.deepcopy(all_jobs[ID])
                if 'shape' in job_args:
                    job_args['shape'] = job_args['shape'] // 2  # Make shape smaller for debugging
                model, metadata = main(**job_args, from_checkpoint=bool(ITER > 0), debug=True)
                print('\n'*5)

    else:
        """Other machines are for running."""

        # Run main function
        job_args = all_jobs[ID]
        model, metadata = main(**job_args, from_checkpoint=bool(ITER > 0))

    # Done
    print('Done!')
    
