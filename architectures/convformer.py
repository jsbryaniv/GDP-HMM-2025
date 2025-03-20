
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from architectures.blocks import ConvformerEncoder3d, VolumeContract3d, VolumeExpand3d


# Define full convolutional transformer model
class ConvformerModel(nn.Module):
    """Full Convolutional Transformer model"""
    def __init__(self,
        in_channels, out_channels,
        n_features=32, n_layers=16, n_heads=4, kernel_size=5, expansion=1,
        scale=4,
    ):
        super(ConvformerModel, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.scale = scale

        # Define input block
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=3, padding=1),
            # Shrink volume
            VolumeContract3d(n_features=n_features, scale=scale),

        )

        # Define convformer encoder
        self.convformer = ConvformerEncoder3d(
            n_features=n_features,
            n_layers=n_layers,
            kernel_size=kernel_size,
            n_heads=n_heads,
            expansion=expansion,
        )

        # Define output block
        self.output_block = nn.Sequential(
            # Expand volume
            VolumeExpand3d(n_features=n_features, scale=scale),
            # Merge features to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_features': self.n_features,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'kernel_size': self.kernel_size,
            'expansion': self.expansion,
            'scale': self.scale,
        }

    def forward(self, x):
        
        # Input block
        x = self.input_block(x)

        # Convformer layers
        x = self.convformer(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    shape = (64, 64, 64)
    in_channels = 36
    out_channels = 1

    # Create data
    x = torch.randn(1, in_channels, *shape)

    # Create a model
    model = ConvformerModel(
        in_channels, 
        out_channels,
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        y = model(x)

    # Estimate memory usage
    estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

