
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from architectures.blocks import ConvBlock3d, ConvBlock3d_v1, ConvBlock3d_v2, ConvBlock3d_v3, ConvBlock3d_v4


# Define Unet encoder
class UnetEncoder3d(nn.Module):
    def __init__(self, in_channels, n_features=16, n_blocks=5, n_layers_per_block=4, scale=1, version=0):
        super(UnetEncoder3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.scale = scale

        # Get convblock class
        if version == 0:
            convblock = ConvBlock3d
        elif version == 1:
            convblock = ConvBlock3d_v1
        elif version == 2:
            convblock = ConvBlock3d_v2
        elif version == 3:
            convblock = ConvBlock3d_v3
        elif version == 4:
            convblock = ConvBlock3d_v4

        # Define input block
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Shrink volume
            convblock(n_features, n_features, scale=1/scale),
            # Additional convolutional layers
            *(convblock(n_features, n_features) for _ in range(n_layers_per_block - 1))
        )

        # Define downsample blocks
        self.down_blocks = nn.ModuleList()
        for depth in range(n_blocks):
            n_in = (depth+1) * n_features
            n_out = (depth+2) * n_features
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    convblock(n_in, n_out, groups=n_features, scale=1/2),
                    # Additional convolutional layers
                    *[convblock(n_out, n_out, groups=n_features) for _ in range(n_layers_per_block - 1)]
                )
            )
        
    def forward(self, x):

        # Initialize features list
        feats = []

        # Input block
        x = self.input_block(x)
        feats.append(x)

        # Downsample blocks
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            feats.append(x)

        # Return the features
        return feats

# Define Unet decoder
class UnetDecoder3d(nn.Module):
    def __init__(self, out_channels, n_features=16, n_blocks=5, n_layers_per_block=4, scale=1, version=0):
        super(UnetDecoder3d, self).__init__()
        
        # Set attributes
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.scale = scale

        # Get convblock class
        if version == 0:
            convblock = ConvBlock3d
        elif version == 1:
            convblock = ConvBlock3d_v1
        elif version == 2:
            convblock = ConvBlock3d_v2
        elif version == 3:
            convblock = ConvBlock3d_v3
        elif version == 4:
            convblock = ConvBlock3d_v4

        # Define upsample blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            depth = self.n_blocks - 1 - i
            n_in = (depth+2) * n_features
            n_out = (depth+1) * n_features
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample layer
                    convblock(n_in, n_out, scale=2, groups=n_features),
                    # Additional convolutional layers
                    *[convblock(n_out, n_out, groups=n_features) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Convolutional layers
            *[convblock(n_features, n_features) for _ in range(n_layers_per_block - 1)],
            # Expand volume
            convblock(n_features, n_features, scale=scale),
            # Merge features to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
    def forward(self, feats):

        # Get x
        x = feats.pop()

        # Upsample blocks
        for i, block in enumerate(self.up_blocks):
            # Upsample
            x = block(x)
            # Merge with skip
            x_skip = feats.pop()
            x = x + x_skip

        # Output block
        x = self.output_block(x)

        # Return the output
        return x

# Define simple 3D Unet model
class Unet3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=16, n_blocks=5, n_layers_per_block=4, scale=1, version=0):
        super(Unet3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.scale = scale
        self._version = version

        # Define encoder
        self.encoder = UnetEncoder3d(
            in_channels=in_channels,
            n_features=n_features,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale,
            version=version,
        )

        # Define decoder
        self.decoder = UnetDecoder3d(
            out_channels=out_channels,
            n_features=n_features,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale,
            version=version,
        )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
            'scale': self.scale,
            'version': self._version,
        }
        
    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats)
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
    model = Unet3d(
        in_channels=in_channels, 
        out_channels=out_channels,
        version=3,
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

