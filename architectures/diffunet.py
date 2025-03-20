
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import custom libraries
from architectures.unet import Unet3d, UnetEncoder3d
from architectures.blocks import ConvformerDecoder3d, VolumeContract3d, VolumeExpand3d


# Define Diffusion Model Unet
class DiffUnet3d(nn.Module): # TODO: Make this a wrapper
    def __init__(self, 
        in_channels, n_cross_channels_list,
        n_features=8, n_blocks=4, 
        n_layers_per_block=2, n_mixing_blocks=2,
        dt=1, kT_max=10, n_steps=8,
        scale=2,
    ):
        super(DiffUnet3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_mixing_blocks = n_mixing_blocks
        self.dt = dt
        self.kT_max = kT_max
        self.n_steps = n_steps
        self.scale = scale

        # Get constants
        n_context = len(n_cross_channels_list)
        n_features_per_depth = [n_features * (i+1) for i in range(n_blocks+1)]
        kT_schedule = torch.linspace(0, kT_max, n_steps).flip(0)
        self.n_context = n_context
        self.n_features_per_depth = n_features_per_depth
        self.kT_schedule = kT_schedule

        # Create main autoencoder
        self.autoencoder = Unet3d(
            in_channels, in_channels, 
            n_features=n_features, n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale,
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_encoders.append(
                # Unet3d(
                #     n_features, n_features, 
                #     n_features=n_features, n_blocks=n_blocks,
                #     n_layers_per_block=n_layers_per_block,
                # )
                UnetEncoder3d(
                    n_channels,
                    n_features=n_features, n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block,
                    scale=scale,
                )
            )

        # Create cross attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.cross_attn_blocks.append(
                ConvformerDecoder3d(
                    n_features_per_depth[depth], 
                    n_heads=depth+1,
                    n_layers=n_mixing_blocks,
                )
            )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
            'n_mixing_blocks': self.n_mixing_blocks,
            'dt': self.dt,
            'kT_max': self.kT_max,
            'n_steps': self.n_steps,
            'scale': self.scale,
        }
    
    def force(self, x, f_context):

        # Encode x
        feats = self.autoencoder.encoder(x)
        x = feats.pop()

        # Apply context
        depth = self.n_blocks
        fcon = f_context[-1]
        x = self.cross_attn_blocks[depth](x, fcon)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            # Upsample
            x = self.autoencoder.decoder.up_blocks[i](x)
            # Merge with skip
            x_skip = feats[depth]
            x = x + x_skip
            # Apply cross attention
            fcon = f_context[depth]
            x = self.cross_attn_blocks[depth](x, fcon)

        # Output block
        x = self.autoencoder.decoder.output_block(x)
        
        # Return
        return x

    def forward(self, x, *y_list):
        """
        x is the input tensor
        y_list is a list of context tensors
        """
        
        # Encode features
        feats_context = [block(y) for block, y in zip(self.context_encoders, y_list)]
        feats_context = [sum([f for f in row]) for row in zip(*feats_context)]
        
        # Loop over temperature schedule
        for kT in self.kT_schedule:

            # Add noise
            x = x + kT * torch.randn_like(x, device=x.device)
            
            # Calculate force
            # F = self.force(x, feats_context)
            F = checkpoint(self.force, x.clone().requires_grad_(True), feats_context, use_reentrant=False)

            # Update position
            x = x + self.dt * F
        
        # Return
        return x
        

# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    shape = (128, 128, 128)
    in_channels = 1
    n_cross_channels_list = [1, 4, 36]

    # Create data
    x = torch.randn(1, in_channels, *shape)
    y_list = [torch.randn(1, c, *shape) for c in n_cross_channels_list]

    # Create a model
    model = DiffUnet3d(
        in_channels, n_cross_channels_list,
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        y = model(x, *y_list)

    # Estimate memory usage
    estimate_memory_usage(model, x, *y_list, print_stats=True)

    # Done
    print('Done!')


