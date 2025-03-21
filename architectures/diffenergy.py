
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
from architectures.blocks import ConvBlock3d, ConvformerDecoder3d, VolumeContractSparse3d, VolumeExpandSparse3d


# Define SDM Volume Encoder
class SDMVolumeEncoder(nn.Module):
    """SDM Volume Encoder module."""
    def __init__(self, in_channels, n_features=4, n_blocks=5, n_layers_per_block=4):
        super(SDMVolumeEncoder, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Define input block
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1))
        )

        # Define downsample blocks
        self.down_blocks = nn.ModuleList()
        for depth in range(n_blocks):
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    ConvBlock3d(n_features, n_features, downsample=True),
                    # Additional convolutional layers
                    *[ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1)]
                )
            )

    def forward(self, x):
        
        # Initialize list of features
        feats = []

        # Input block
        x = self.input_block(x)
        feats.append(x.clone())

        # Downsample blocks
        for block in self.down_blocks:
            x = block(x)
            feats.append(x.clone())

        # Return features
        return feats


# Define Scalar Diffusion Model
class SDM3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        n_features=16, n_blocks=5, 
        n_layers_per_block=2, n_mixing_blocks=2,
        dt=1, kT_max=10, n_steps=16
    ):
        super(SDM3d, self).__init__()
        
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

        # Create temperature schedule
        kT_schedule = torch.linspace(0, kT_max, n_steps).flip(0)
        self.kT_schedule = kT_schedule

        # Create input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Shrink volume
            VolumeContractSparse3d(n_features=n_features, scale=4),
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    nn.Conv3d(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    VolumeContractSparse3d(n_features=n_features, scale=4),
                )
            )

        # Create output block
        self.output_block = nn.Sequential(
            # Expand volume
            VolumeExpandSparse3d(n_features=n_features, scale=4),
            # Merge output channels to in_channels
            nn.Conv3d(n_features, in_channels, kernel_size=1),
        )

        # Define volume encoder
        self.volume_encoder = SDMVolumeEncoder(
            in_channels=n_features,
            n_features=n_features,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
        )

        # Define context encoders
        self.context_encoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_encoders.append(
                SDMVolumeEncoder(
                    in_channels=n_features,
                    n_features=n_features,
                    n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block,
                )
            )

        # Define mixing and linear blocks
        self.mixing_blocks = nn.ModuleList()
        self.linear_blocks = nn.ModuleList()
        for _ in range(n_blocks+1):
            self.mixing_blocks.append(
                ConvformerDecoder3d(n_features=n_features, n_layers=n_mixing_blocks)
            )
            self.linear_blocks.append(
                nn.Conv3d(n_features, n_features, kernel_size=1)
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
        }
    
    def energy(self, x, feats_context):
        """Calculate energy of a volume."""

        # Initialize energy
        U_terms = []
        
        # Encode inputs
        feats = self.volume_encoder(x)

        # Calculate energy
        for i, (fx, fy) in enumerate(zip(feats, feats_context)):
            fx = self.mixing_blocks[i](fx, fy)  # Mix features
            fx = self.linear_blocks[i](fx)      # Linear layer
            u = - (fx**2).mean(dim=(1,2,3,4))   # Calculate energy
            U_terms.append(u)

        # Sum energy terms
        U = sum(U_terms)

        # Return energy
        return U
    
    def force(self, x, feats_context):
        """Calculate the force (-grad(U)) of a volume."""

        # Track gradients
        x = x.clone().requires_grad_(True)
        feats_context = [f.clone().requires_grad_(True) for f in feats_context]

        # Calculate energy
        U = checkpoint(self.energy, x, feats_context, use_reentrant=False)

        # Calculate force
        F = - torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), create_graph=True, retain_graph=True)[0]

        # Return force
        return F
        
    def forward(self, x, *y_list):
        """
        x is the input tensor
        y_list is a list of context tensors
        """

        # Shrink input and context
        x = self.input_block(x)
        y_list = [block(y) for block, y in zip(self.context_input_blocks, y_list)]
        
        # Encode features
        feats_context = [encoder(y) for encoder, y in zip(self.context_encoders, y_list)]
        feats_context = [sum([f for f in row]) for row in zip(*feats_context)]
        
        # Loop over temperature schedule
        for kT in self.kT_schedule:

            # Add noise
            x = x + kT * torch.randn_like(x, device=x.device)
            
            # Calculate force
            F = self.force(x, feats_context)

            # Update position
            x = x + self.dt * F
        
        # Return
        return x
        



# Test the model
if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)  # (Optional) Helps debug gradient issues
    torch.set_num_threads(1)                 # Forces computations to use one thread
    torch.set_num_interop_threads(1)         # Disables inter-op parallelism

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
    model = SDM3d(
        in_channels, n_cross_channels_list,
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    y = model(x, *y_list)

    # Estimate memory usage
    estimate_memory_usage(model, x, *y_list, print_stats=True)

    # Done
    print('Done!')
