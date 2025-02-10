
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


# Load dataset function
def load_dataset(dataID, **kwargs):

    # Load dataset
    if dataID.lower() == 'han':

        # Import dataset
        from dataset import GDPDataset

        # Set constants
        in_channels = 35
        out_channels = 1
        shape = (64, 64, 64)
        scale = .5

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
            shape=(64, 64, 64),
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

    # Return model
    return model


# Define main function
def main(dataID, modelID, data_kwargs=None, model_kwargs=None, train_kwargs=None):
    """
    Function to train a model on a dataset.
    """
    print(f"Running main function for model {modelID} on dataset {dataID}.")

    # Check inputs
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}

    ### DATASET ###
    print("Loading dataset.")

    # Load dataset
    dataset, data_metadata = load_dataset(dataID, **data_kwargs)

    # Get metadata
    in_channels = data_metadata['in_channels']
    out_channels = data_metadata['out_channels']

    # Split into train, validation, and test sets
    test_size = int(0.2 * len(dataset))
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        [len(dataset) - 2*test_size, test_size, test_size]
    )

    # Get files of each subset
    indices_train = dataset_train.indices
    indices_val = dataset_val.indices
    indices_test = dataset_test.indices


    ### MODEL ###
    print("Setting up model.")

    # Initialize model
    model = load_model(modelID, in_channels, out_channels, **model_kwargs)

    # Move model to device
    model.to(device)


    ### TRAINING LOOP ###
    print("Starting training.")

    # Train model
    model, training_statistics = train_model(
        model, dataset_train, dataset_val,
        batch_size=1, learning_rate=0.01, num_epochs=50,
        **train_kwargs,
    )


    ### SAVE RESULTS ###
    print("Saving results.")

    # Get savename
    savename = f'model2_{dataID}_{modelID}'

    # Update training statistics
    training_statistics['dataID'] = dataID
    training_statistics['modelID'] = modelID
    training_statistics['data_kwars'] = data_kwargs
    training_statistics['model_kwars'] = model_kwargs
    training_statistics['train_kwars'] = train_kwargs
    training_statistics['indices_train'] = indices_train
    training_statistics['indices_val'] = indices_val
    training_statistics['indices_test'] = indices_test

    # Save model
    torch.save(
        model.state_dict(), 
        os.path.join(config['PATH_OUTPUT'], f'{savename}.pth')
    )

    # Save training statistics
    with open(os.path.join(config['PATH_OUTPUT'], f'{savename}.json'), 'w') as f:
        json.dump(training_statistics, f)


    ### DONE ###
    print(f"Finished running job for {modelID} on dataset {dataID}.")

    # Return model and training statistics
    return model, training_statistics


# Run main function
if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set job IDs
    all_jobs = [
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
    main(**job_args)

    # Done
    print('Done!')
    
