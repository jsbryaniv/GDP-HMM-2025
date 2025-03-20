
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
from architectures.vit import ViT3d, ViTEncoder3d
from architectures.blocks import ConvBlock3d, CrossTransformerBlock


# Define Diffusion Model ViT
class DiffViT3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        shape=128, scale=1, shape_patch_ratio=8, n_features=128, n_heads=4, 
        n_layers=8, n_layers_mixing=4, n_layers_input=4,
        dt=1, kT_max=10, n_steps=8, langevin=False,
    ):
        super(DiffViT3d, self).__init__()

        # Check inputs
        if isinstance(shape, int):
            shape = (shape, shape, shape)
        if isinstance(n_cross_channels_list, int):
            n_cross_channels_list = [n_cross_channels_list]
        
        # Set attributes
        self.in_channels = in_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.shape = shape
        self.scale = scale
        self.shape_patch_ratio = shape_patch_ratio
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_mixing = n_layers_mixing
        self.n_layers_input = n_layers_input
        self.dt = dt
        self.kT_max = kT_max
        self.n_steps = n_steps
        self.langevin = langevin
        
        # Get constants
        n_context = len(n_cross_channels_list)
        shape_scaled = tuple(s // scale for s in shape)
        kT_schedule = torch.linspace(0, kT_max, n_steps).flip(0)
        self.n_context = n_context
        self.shape_scaled = shape_scaled
        self.kT_schedule = kT_schedule

        # Define input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Shrink volume
            ConvBlock3d(n_features, n_features, scale=1/scale, groups=n_features),
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1))
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    nn.Conv3d(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    ConvBlock3d(n_features, n_features, scale=1/scale, groups=n_features),
                    # Additional convolutional layers
                    *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1))
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Convolutional layers
            *[ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1)],
            # Expand volume
            ConvBlock3d(n_features, n_features, scale=scale, groups=n_features),
            # Merge features to output channels
            nn.Conv3d(n_features, in_channels, kernel_size=1),
        )

        # Create main autoencoder
        self.autoencoder = ViT3d(
            in_channels=n_features, out_channels=n_features,
            shape=shape_scaled, scale=1, shape_patch_ratio=shape_patch_ratio,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers, 
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for _ in range(len(n_cross_channels_list)):
            self.context_encoders.append(
                ViTEncoder3d(
                    in_channels=n_features,
                    shape=shape_scaled, scale=1, shape_patch_ratio=shape_patch_ratio,
                    n_features=n_features, n_heads=n_heads, n_layers=n_layers//2, 
                )
            )

        # Create mixing block
        self.mixing_blocks = nn.ModuleList()
        for _ in range(n_layers_mixing):
            self.mixing_blocks.append(
                CrossTransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                )
            )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'shape': self.shape,
            'scale': self.scale,
            'shape_patch_ratio': self.shape_patch_ratio,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_layers_mixing': self.n_layers_mixing,
            'n_layers_input': self.n_layers_input,
            'dt': self.dt,
            'kT_max': self.kT_max,
            'n_steps': self.n_steps,
            'langevin': self.langevin,
        }
    
    def force(self, x, f_context, pos_embedding=None):

        # Encode input
        x = self.autoencoder.encoder(x)

        # Add positional embeddings
        if pos_embedding is None:
            pos_embedding = (
                self.autoencoder.encoder.pos_embedding_0 
                + self.autoencoder.encoder.pos_embedding_1 
                + self.autoencoder.encoder.pos_embedding_2
            )
            pos_embedding = pos_embedding.flatten(2).transpose(1, 2).expand(x.shape[0], -1, -1)
        x = x + pos_embedding 

        # Mixing block
        for block in self.mixing_blocks:
            x = block(x, f_context)

        # Decode
        x = self.autoencoder.decoder(x)
        
        # Return
        return x

    def forward(self, x, *y_list):
        """
        x is the input tensor
        y_list is a list of context tensors
        """

        # Input blocks
        x = self.input_block(x)
        y_list = [block(y) for block, y in zip(self.context_input_blocks, y_list)]
        
        # Encode features
        f_context = sum(block(y) for block, y in zip(self.context_encoders, y_list))

        # Add positional embeddings to context
        pos_embedding = (
            self.autoencoder.encoder.pos_embedding_0 
            + self.autoencoder.encoder.pos_embedding_1 
            + self.autoencoder.encoder.pos_embedding_2
        )
        pos_embedding = pos_embedding.flatten(2).transpose(1, 2).expand(x.shape[0], -1, -1)
        f_context = f_context + pos_embedding
        
        # Loop over temperature schedule
        for kT in self.kT_schedule:

            # Add noise
            x = x + kT * torch.randn_like(x, device=x.device)
            
            # Calculate force
            F = checkpoint(
                self.force, 
                # x.clone().requires_grad_(True), 
                # f_context.clone().requires_grad_(True), 
                x,
                f_context,
                pos_embedding.clone().requires_grad_(True), 
                use_reentrant=False,
            )

            # Update position
            if self.langevin:
                x = x + self.dt * F
            else:
                x = F

        # Output block
        x = self.output_block(x)

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
    model = DiffViT3d(
        in_channels, n_cross_channels_list,
        shape=shape,
        scale=2,
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


