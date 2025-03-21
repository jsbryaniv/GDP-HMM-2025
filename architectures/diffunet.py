
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
from architectures.blocks import ConvBlock3d, ConvformerDecoder3d


# Define Diffusion Model Unet
class DiffUnet3d(nn.Module): # TODO: Make this a wrapper
    def __init__(self, 
        in_channels, n_cross_channels_list,
        scale=1, n_features=16, n_blocks=4, 
        n_layers_per_block=2, n_mixing_blocks=2,
        beta_min=1e-4, beta_max=.02, n_steps=16,
    ):
        super(DiffUnet3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.scale = scale
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_mixing_blocks = n_mixing_blocks
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_steps

        # Get constants
        n_context = len(n_cross_channels_list)
        n_features_per_depth = [n_features * (i+1) for i in range(n_blocks+1)]
        self.n_context = n_context
        self.n_features_per_depth = n_features_per_depth

        # Get noise schedule
        beta_schedule = torch.linspace(beta_min, beta_max, n_steps).flip(0)
        alpha_schedule = 1 - beta_schedule
        alpha_cumprod = alpha_schedule.cumprod(dim=0)
        self.beta_schedule = beta_schedule
        self.alpha_schedule = alpha_schedule
        self.alpha_cumprod = alpha_cumprod

        # Define time embedding layers
        self.time_embedding = nn.ModuleList()
        for f in n_features_per_depth:
            self.time_embedding.append(
                nn.Sequential(
                    nn.Linear(1, f),
                    nn.ReLU(), 
                    nn.Linear(f, f),
                    nn.Unflatten(1, (f, 1, 1, 1)),
                )
            )

        # Define input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Shrink volume
            ConvBlock3d(n_features, n_features, scale=1/scale),
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1))
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    nn.Conv3d(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    ConvBlock3d(n_features, n_features, scale=1/scale),
                    # Additional convolutional layers
                    *(ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1))
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Convolutional layers
            *[ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1)],
            # Expand volume
            ConvBlock3d(n_features, n_features, scale=scale),
            # Merge features to output channels
            nn.Conv3d(n_features, in_channels, kernel_size=1),
        )

        # Create main autoencoder
        self.autoencoder = Unet3d(
            n_features, n_features, 
            n_features=n_features, n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=1,
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for _ in range(len(n_cross_channels_list)):
            self.context_encoders.append(
                UnetEncoder3d(
                    n_features,
                    n_features=n_features, n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block,
                    scale=1,
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
            'scale': self.scale,
            'dt': self.dt,
            'kT_max': self.kT_max,
            'n_steps': self.n_steps,
            'langevin': self.langevin,
        }
    
    def step(self, x, feats_context, t):

        # Check inputs 
        if isinstance(t, int):
            t = torch.tensor(t, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        
        # Encode x
        feats = self.autoencoder.encoder(x)

        # Add time embedding
        feats = [f + self.time_embedding[i](t) for i, f in enumerate(feats)]

        # Apply context
        feats = [block(fx, fy) for block, fx, fy in zip(self.cross_attn_blocks, feats, feats_context)]

        # Decode features
        noise_pred = self.autoencoder.decoder(feats)
        
        # Return noise prediction
        return noise_pred
    
    def forward(self, *context):

        # Input blocks
        context = [block(y) for block, y in zip(self.context_input_blocks, context)]

        # Initialize x
        x = torch.randn_like(context[0], device=context[0].device)
        
        # Encode features
        feats_context = [block(y) for block, y in zip(self.context_encoders, context)]
        feats_context = [sum([f for f in row]) for row in zip(*feats_context)]

        # Diffusion steps
        for t in reversed(range(self.n_steps)):

            # Predict noise
            noise_pred = self.step(x, feats_context, t)

            # Update position
            x = (x - torch.sqrt(1 - self.alpha_cumprod[t]) * noise_pred) / torch.sqrt(self.alpha_cumprod[t])

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
    shape = (64, 64, 64)
    batch_size = 3
    in_channels = 1
    n_cross_channels_list = [1, 4, 36]

    # Create data
    y_list = [torch.randn(batch_size, c, *shape) for c in n_cross_channels_list]

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
        y = model(*y_list)

    # Estimate memory usage
    estimate_memory_usage(model, *y_list, print_stats=True)

    # Done
    print('Done!')


