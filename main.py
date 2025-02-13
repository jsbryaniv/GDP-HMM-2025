
# Import libraries
import os
import sys
import json
import time
import torch

# Import custom classes
from train import train_model
from plotting import plot_losses, plot_prediction

# Set environment
with open('config.json', 'r') as f:
    config = json.load(f)


# Get savename function
def get_savename(dataID, modelID, **kwargs):
    """
    Get savename for a given dataset and model.
    """

    # Initialize savename
    savename = f'model_{dataID}_{modelID}'

    # Add kwargs to savename
    kwargs_sorted = sorted(kwargs.items())
    for key, value in kwargs_sorted:
        if value:
            savename += f'_{key}={value}'

    # Return savename
    return savename


# Load dataset function
def load_dataset(dataID, **kwargs):

    # Load dataset
    if dataID.lower() == 'han':

        # Import dataset
        from dataset import GDPDataset

        # Set constants
        in_channels = 36
        out_channels = 1
        shape = (128, 128, 128)
        scale = 1

        # Create dataset
        dataset = GDPDataset(
            treatment='HaN', 
            shape=shape,
            scale=scale,
            return_dose=True,
            **kwargs,
        )

        # Collect metadata
        metadata = {
            'dataID': dataID,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'shape': shape,
            'scale': scale,
        }

    # Return dataset
    return dataset, metadata

# Load model
def load_model(modelID, in_channels, out_channels, **kwargs):

    # Load model
    if modelID.lower() == 'unet':
        # Unet3D model
        from models.unet import Unet3D
        model = Unet3D(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs,
        )
    elif modelID.lower() == 'vit':
        # Vision Transformer model
        from models.vit import ViT3D
        model = ViT3D(
            in_channels=in_channels, 
            out_channels=out_channels,
            shape=(128, 128, 128),
            **kwargs,
        )
    elif modelID.lower() == 'convformer':
        # Convolutional Transformer model
        from models.convformer import ConvformerModel
        model = ConvformerModel(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs,
        )
    elif modelID.lower() == 'uconvtrans':
        # U-Convformer model
        from models.uconvformer import UConvformerModel
        model = UConvformerModel(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs,
        )

    # Return model
    return model


# Define main function
def main(
    dataID, modelID,
    data_kwargs=None, model_kwargs=None, train_kwargs=None,
    continue_training=False,
):
    """
    Main function to train a model on a dataset.
    """
    print(f"Running main function for model {modelID} on dataset {dataID}.")
        
    # Get savename
    savename = get_savename(dataID, modelID, **data_kwargs, **model_kwargs)
    print(f"-- savename={savename}")

    # Get constants
    path_output = config['PATH_OUTPUT']

    # Check inputs
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}
    
    # If continuing training, load previous files
    if continue_training:
        old_metadata = json.load(os.path.join(path_output, f'{savename}.json'))
        old_model_state = torch.load(os.path.join(path_output, f'{savename}.pth'))

    ### DATASET ###
    print("Loading dataset.")

    # Load dataset
    dataset, data_metadata = load_dataset(dataID, **data_kwargs)

    # Get metadata
    in_channels = data_metadata['in_channels']
    out_channels = data_metadata['out_channels']

    # Split into train, validation, and test sets
    if continue_training:
        # Initialize datasets as Subset objects
        dataset_val = torch.utils.data.Subset(dataset, old_metadata['indices_val'])
        dataset_test = torch.utils.data.Subset(dataset, old_metadata['indices_test'])
        dataset_train = torch.utils.data.Subset(dataset, old_metadata['indices_train'])
    else:
        # If continuing training, split dataset into train, validation, and test sets
        test_size = int(0.2 * len(dataset))
        dataset_val, dataset_test, dataset_train = torch.utils.data.random_split(
            dataset,
            [test_size, test_size, len(dataset) - 2*test_size],
        )
        # Get files of each subset
        indices_val = dataset_val.indices
        indices_test = dataset_test.indices
        indices_train = dataset_train.indices


    ### MODEL ###
    print("Setting up model.")

    # Initialize model
    model = load_model(modelID, in_channels, out_channels, **model_kwargs)

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
        batch_size=1, learning_rate=0.01, n_epochs=20,
        jobname=savename,
        **train_kwargs,
    )


    ### SAVE RESULTS ###
    print("Saving results.")

    # Merge new and old training statistics
    if continue_training:
        # Get old training statistics
        old_training_statistics = old_metadata['training_statistics']
        # Find the best loss
        training_statistics['loss_val_best'] = min(
            training_statistics['loss_val_best'],
            old_training_statistics['loss_val_best'],
        )
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

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set job IDs
    all_jobs = [
        {
            'dataID': 'HaN', 
            'modelID': 'UConvTrans',
            'data_kwargs': {},
            'model_kwargs': {},
            'train_kwargs': {},
        },
        {
            'dataID': 'HaN', 
            'modelID': 'ConvFormer',
            'data_kwargs': {},
            'model_kwargs': {},
            'train_kwargs': {},
        },
        {
            'dataID': 'HaN', 
            'modelID': 'ViT',
            'data_kwargs': {},
            'model_kwargs': {},
            'train_kwargs': {},
        },
        {
            'dataID': 'HaN', 
            'modelID': 'Unet',
            'data_kwargs': {},
            'model_kwargs': {},
            'train_kwargs': {},
        },
    ]
    
    # Get training IDs from system arguments
    ID = 0
    args = sys.argv
    if len(args) > 1:
        ID = int(args[1])

    # Run main function
    job_args = all_jobs[ID]
    model, metadata = main(**job_args)

    # Done
    print('Done!')
    
