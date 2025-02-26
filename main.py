
# Import libraries
import os
import sys
import json
import time
import torch

# Import custom classes
from test import test_model
from train import train_model
from utils import get_savename, initialize_dataset, initialize_model

# Get config
with open('config.json', 'r') as f:
    config = json.load(f)
path_output = config['PATH_OUTPUT']


# Define main function
def main(
    dataID, modelID,
    data_kwargs=None, model_kwargs=None, train_kwargs=None,
    continue_training=False, debug=False,
):
    """
    Main function to train a model on a dataset.
    """
    print(f"Running main function for model {modelID} on dataset {dataID}.")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Get savename
    savename = get_savename(dataID, modelID, **data_kwargs, **model_kwargs)
    print(f"-- savename={savename}")

    # Check inputs
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}
    
    # If continuing training, load previous files
    if continue_training:
        # Print warning message
        for _ in range(10):
            print("WARNING: Be aware job is running with continue_training=True.")
        # Load old data
        old_model_state = torch.load(os.path.join(path_output, f'{savename}.pth'), weights_only=True)
        with open(os.path.join(path_output, f'{savename}.json'), 'r') as f:
            old_metadata = json.load(f)
        # Get epoch start
        epoch_start = len(old_metadata['training_statistics']['losses_train'])
        train_kwargs['epoch_start'] = epoch_start

    ### DATASET ###
    print("Loading dataset.")

    # Load dataset
    dataset, data_metadata = initialize_dataset(dataID, **data_kwargs)

    # Get metadata
    in_channels = data_metadata['in_channels']
    out_channels = data_metadata['out_channels']

    # Split into train, validation, and test sets
    if continue_training:
        # Get indices of each subset
        indices_val = old_metadata['indices_val']
        indices_test = old_metadata['indices_test']
        indices_train = old_metadata['indices_train']
        # Initialize datasets as Subset objects
        dataset_val = torch.utils.data.Subset(dataset, indices_val)
        dataset_test = torch.utils.data.Subset(dataset, indices_test)
        dataset_train = torch.utils.data.Subset(dataset, indices_train)
    else:
        # Split dataset into train, validation, and test sets
        test_size = int(0.2 * len(dataset))
        dataset_val, dataset_test, dataset_train = torch.utils.data.random_split(
            dataset,
            [test_size, test_size, len(dataset) - 2*test_size],
            generator=torch.Generator().manual_seed(42),  # Set seed for reproducibility
        )
        # Get indices of each subset
        indices_val = dataset_val.indices
        indices_test = dataset_test.indices
        indices_train = dataset_train.indices


    ### MODEL ###
    print("Setting up model.")

    # Initialize model
    model = initialize_model(modelID, in_channels, out_channels, **model_kwargs)

    # Load model if continuing training
    if continue_training:
        model.load_state_dict(old_model_state)

    # Move model to device
    model.to(device)


    ### TRAINING LOOP ###
    print("Starting training.")

    # Train model
    model, training_statistics = train_model(
        model, dataset_train, dataset_val,
        jobname=savename, debug=debug,
        **train_kwargs,
    )


    ### TESTING ###
    print("Testing model.")

    # Test model
    loss_test = test_model(model, dataset_test, debug=debug)


    ### SAVE RESULTS ###
    print("Saving results.")

    # Merge new and old training statistics
    if continue_training:
        # Get old training statistics
        old_training_statistics = old_metadata['training_statistics']
        # Find the best loss
        old_loss_val_bset = old_training_statistics['loss_val_best']
        new_loss_val_best = training_statistics['loss_val_best']
        if old_loss_val_bset < new_loss_val_best:
            # If old loss is better, keep old loss and load old model state
            training_statistics['loss_val_best'] = old_loss_val_bset
            model.load_state_dict(old_model_state)  # Load old model state
        # Merge loss lists
        for key in ['losses_train', 'losses_val']:
            training_statistics[key] = old_training_statistics[key] + training_statistics[key]

    # Collect metadata
    metadata = {
        'dataID': dataID,
        'modelID': modelID,
        'data_metadata': data_metadata,
        'data_kwargs': data_kwargs,
        'model_kwargs': model_kwargs,
        'train_kwargs': train_kwargs,
        'indices_val': indices_val,
        'indices_test': indices_test,
        'indices_train': indices_train,
        'training_statistics': training_statistics,
        'loss_test': loss_test,
    }

    # Save model
    torch.save(
        model.state_dict(), 
        os.path.join(path_output, f'{savename}.pth')
    )

    # Save training statistics
    with open(os.path.join(path_output, f'{savename}.json'), 'w') as f:
        json.dump(metadata, f)


    ### DONE ###
    print(f"Finished running job for {modelID} on dataset {dataID}.")

    # Return model and training statistics
    return model, metadata


# Run main function
if __name__ == '__main__':

    # Set job IDs
    # dataID = 'HaN'
    dataID = 'HalfHaN'  # TODO: Debugging
    all_jobs = []
    for modelID in ['CrossAttnAE', 'ViT', 'Unet']:

        # Initialize kwargs
        data_kwargs = {}
        model_kwargs = {}
        train_kwargs = {}

        # Get job specific kwargs
        if modelID == 'CrossViT':
            train_kwargs = {'loss_type': 'crossae'}
        if modelID == 'CrossAttnAE':
            train_kwargs = {'loss_type': 'crossae'}
        if ('vit' in modelID.lower()) and ('half' in dataID.lower()):
            model_kwargs = {'shape': 64, 'scale': 2}

        # Add job
        all_jobs.append({
            'dataID': dataID, 
            'modelID': modelID,
            'data_kwargs': data_kwargs,
            'model_kwargs': model_kwargs,
            'train_kwargs': train_kwargs,
        })
    
    # Get training IDs from system arguments
    ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Run main function
    job_args = all_jobs[ID]
    model, metadata = main(**job_args, continue_training=bool(ITER > 0))

    # # Debugging
    # for ID in range(len(all_jobs)//2):
    #     for ITER in range(2):
    #         job_args = all_jobs[ID]
    #         model, metadata = main(**job_args, continue_training=bool(ITER > 0), debug=True)
    #         print('\n'*5)

    # Done
    print('Done!')
    
