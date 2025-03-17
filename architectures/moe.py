
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import custom libraries
from architectures.blocks import ConvBlock, VoxelNorm3d


# Define MOE Gating
class MOEGating3d(nn.Module):
    """Multi-expert gating mechanism."""
    def __init__(self, in_channels, n_experts, n_features=16, n_blocks=3, n_layers_per_block=2):
        super(MOEGating3d, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.n_experts = n_experts
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Set up layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv3d(in_channels, n_features, kernel_size=1))
        for i in range(n_blocks):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(n_features, n_features, downsample=True),
                    *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block-1)],
                )
            )

        # Set up norm
        self.norm = VoxelNorm3d(n_features)
        
        # Set up linear layer
        self.linear = nn.Linear(n_features, n_experts)

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_experts': self.n_experts,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
        }

    def forward(self, x):
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        # Average across spatial dimensions
        x = x.mean(dim=(2, 3, 4))

        # Get expert weights
        weights = F.softmax(self.linear(x), dim=1)

        # Return output
        return weights


# Define MOE Wrapper
class MOEWrapper3d(nn.Module):
    """Wrapper for a multi-expert model."""
    def __init__(self, model, in_channels, n_experts=4, gating_config=None, expert_config=None, **kwargs):
        super(MOEWrapper3d, self).__init__()

        # Set up configs

        # Set up attributes
        self.in_channels = in_channels
        self.n_experts = n_experts
            
        # Set up gating
        if gating_config is None:
            gating_config = {}
        gating_config = {
            'in_channels': in_channels,
            'n_experts': n_experts,
            **gating_config,
        }
        self.gating = MOEGating3d(**gating_config)

        # Set up experts
        if expert_config is None:
            expert_config = {}
        expert_config = {
            'in_channels': in_channels,
            **expert_config, 
            **kwargs
        }
        self.experts = nn.ModuleList([model(**expert_config) for _ in range(n_experts)])

    def get_config(self):
        """Get configuration."""
        return {
            'in_channels': self.experts[0].in_channels,
            'n_experts': self.n_experts,
            'gating_config': self.gating.get_config(),
            'expert_config': self.experts[0].get_config(),
        }
    
    def call_gating(self, dummy, *inputs):
        """
        Call the gating mechanism. Used as a workaround for PyTorch's checkpointing,
        which requires a dummy variable with a gradient in order to track grads.
        """
        inputs = (x + dummy for x in inputs)
        return self.gating(*inputs)

    def call_expert(self, dummy, expert, *inputs):
        """Same as call_gating but for the expert."""
        inputs = (x + dummy for x in inputs)
        return expert(*inputs)

    def forward(self, *inputs):

        # Initialize dummy
        device = next(self.parameters()).device
        dummy = torch.zeros(1, requires_grad=True).to(device)

        # Get expert weights
        weights = checkpoint(self.call_gating, dummy, *inputs)

        # Get expert outputs
        x = torch.stack(
            [checkpoint(self.call_expert, dummy, expert, *inputs) for expert in self.experts], 
            dim=1
        )

        # Weight the expert outputs
        x = torch.einsum('be,becxyz->bcxyz', weights, x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage
    from architectures.unet import Unet3d

    # Set constants
    shape = (64, 64, 64)
    in_channels = 36
    out_channels = 1

    # Create data
    x = torch.randn(1, in_channels, *shape)

    # Create a model
    model = MOEWrapper3d(
        model=Unet3d,
        in_channels=in_channels, 
        out_channels=out_channels,
    )

    # Print model parameter info
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        y = model(x)

    # Done
    print('Done!')

