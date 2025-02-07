
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define main function
def main(dataID, modelID, train_kwars=None, model_kwars=None):
    """
    Function to train a model on a dataset.
    """
    print(f"Running main function for model {modelID} on dataset {dataID}.")

    # Check inputs
    if train_kwars is None:
        train_kwars = {}
    if model_kwars is None:
        model_kwars = {}

    ### DATASET ###
    print("Loading dataset.")

    # Load dataset
    if dataID.lower() == 'han':
        from dataset import GDPDataset
        dataset = GDPDataset(
            treatment='HaN', 
            shape=(128, 128, 128),
            return_dose=True,
        )
        in_channels = 35

    # Split into train, validation, and test sets
    test_size = int(0.2 * len(dataset))
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        [len(dataset) - 2*test_size, test_size, test_size]
    )

    # Get files of each subset
    files_train = [dataset.files[i] for i in dataset_train.indices]
    files_val = [dataset.files[i] for i in dataset_val.indices]
    files_test = [dataset.files[i] for i in dataset_test.indices]


    ### MODEL ###
    print("Setting up model.")

    # Initialize model
    if modelID.lower() == 'unet':
        from models.unet import Unet3D
        model = Unet3D(
            in_channels=in_channels, 
            out_channels=1, 
            **model_kwars,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Loaded Unet3D model with {n_params} trainable parameters.')
    elif modelID.lower() == 'vit':
        from models.vit import ViT3D
        model = ViT3D(
            in_channels=in_channels,
            out_channels=1,
            **model_kwars,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Loaded ViT3D model with {n_params} trainable parameters.')
    elif modelID.lower() == 'convformer':
        from models.convformer import Convformer
        model = Convformer(
            in_channels=in_channels,
            out_channels=1,
            **model_kwars,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Loaded Convformer model with {n_params} trainable parameters.')

    # Move model to device
    model.to(device)


    ### TRAINING LOOP ###
    print("Starting training.")

    # Train model
    model, training_statistics = train_model(
        model, dataset_train, dataset_val,
        batch_size=1, learning_rate=0.001, num_epochs=2,
    )


    ### SAVE RESULTS ###
    print("Saving results.")

    # Update training statistics
    training_statistics['files_train'] = files_train
    training_statistics['files_val'] = files_val
    training_statistics['files_test'] = files_test

    # Save model and statistics
    savename = f'model_{dataID}_{modelID}'
    torch.save(
        model.state_dict(), 
        os.path.join(config['PATH_OUTPUT'], f'{savename}.pth')
    )
    with open(os.path.join(config['PATH_OUTPUT'], f'{savename}.json'), 'w') as f:
        json.dump(training_statistics, f)


    # ### PLOTTING ###
    # print("Plotting results.")

    # # Plot prediction on a single scan
    # ct, ptvs, oars, beam, dose = dataset_test[0]
    # ct = ct.unsqueeze(0).to(device)
    # ptvs = ptvs.unsqueeze(0).to(device)
    # oars = oars.unsqueeze(0).to(device)
    # beam = beam.unsqueeze(0).to(device)
    # dose = dose.unsqueeze(0).to(device)
    # with torch.no_grad():
    #     x = torch.cat([ct, ptvs, oars, beam], dim=1)
    #     pred = model(x)
    # plot_prediction(ct, pred, dose)

    # # Plot training and validation losses
    # losses_train = training_statistics['losses_train']
    # losses_val = training_statistics['losses_val']
    # plot_losses(losses_train, losses_val)

    # Print status
    print(f"Finished running job for {modelID} on dataset {dataID}.")

    # Return model and training statistics
    return model, training_statistics


# Run main function
if __name__ == '__main__':

    # Set job IDs
    all_jobs = [
        ('HaN', 'ConvFormer'),
        ('HaN', 'ViT'),
        ('HaN', 'Unet'),
    ]
    
    # Get training IDs from system arguments
    print('sysargs=',sys.argv)
    ID = 0
    args = sys.argv
    if len(args) > 1:
        ID = int(args[1])

    print('ID=',ID)

    # Run main function
    dataID, modelID = all_jobs[ID]
    main(dataID, modelID)

    # Done
    print('Done!')
    
