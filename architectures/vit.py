
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from architectures.blocks import TransformerBlock, VolumeExpand3d, VolumeContract3d


# Create Vision Transformer 3D Encoder
class ViTEncoder3d(nn.Module):
    def __init__(self, 
        in_channels,
        shape=(64, 64, 64), scale=1, patch_size=None, shape_patch_ratio=16,
        n_features=128, n_heads=4, n_layers=8,
    ):
        super(ViTEncoder3d, self).__init__()

        ### Check inputs ###
        # Check number of heads
        if n_features % n_heads != 0:
            # n_features must be divisible by n_heads
            raise ValueError('Number of features must be divisible by number of heads!')
        # Check shape
        if not isinstance(shape, tuple):
            # Convert shape to tuple
            shape = (shape, shape, shape)
        # Check patch size
        if patch_size is None:
            # Set default patch size (1/shape_patch_ratio of shape)
            patch_size = tuple(shape[i] // shape_patch_ratio for i in range(3))
        elif not isinstance(patch_size, tuple):
            # Convert patch size to tuple
            patch_size = (patch_size, patch_size, patch_size)
        # Check shape and patch size compatibility
        for i in range(3):
            if shape[i] % patch_size[i] != 0:
                # Check if shape is divisible by patch size
                raise ValueError('Shape must be divisible by patch size!')
        
        # Set attributes
        self.in_channels = in_channels
        self.shape = shape
        self.scale = scale
        self.patch_size = patch_size
        self.patch_size_ratio = shape_patch_ratio
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        shape_patchgrid = (
            shape[0] // (scale * patch_size[0]),
            shape[1] // (scale * patch_size[1]),
            shape[2] // (scale * patch_size[2]),
        )
        n_patches = shape_patchgrid[0] * shape_patchgrid[1] * shape_patchgrid[2]
        self.shape_patchgrid = shape_patchgrid
        self.n_patches = n_patches
        if n_patches > 10000:
            # Check it n_patches is too large
            raise ValueError('Number of patches is too large! Descrease size or increase patch size and stride.')
        
        # Positional Encoding
        self.pos_embedding_0 = nn.Parameter(.1*torch.randn(1, n_features, shape_patchgrid[0], 1, 1))
        self.pos_embedding_1 = nn.Parameter(.1*torch.randn(1, n_features, 1, shape_patchgrid[1], 1))
        self.pos_embedding_2 = nn.Parameter(.1*torch.randn(1, n_features, 1, 1, shape_patchgrid[2]))

        # Create input and output blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Rescale volume
            VolumeContract3d(n_features=n_features, scale=scale),
            # Contract volume to patches
            VolumeContract3d(n_features=n_features, scale=patch_size[0]),
        )

        # Transformer Encoders
        self.layers = nn.ModuleList()
        for _ in range(n_layers//2):
            self.layers.append(
                TransformerBlock(n_features=n_features, n_heads=n_heads)
            )

    def forward(self, x):

        # Input block
        x = self.input_block(x)
        x = x.flatten(2).transpose(1, 2)

        # Add positional encoding
        pos_embedding = self.pos_embedding_0 + self.pos_embedding_1 + self.pos_embedding_2
        pos_embedding = pos_embedding.flatten(2).transpose(1, 2).expand(x.shape[0], -1, -1)
        x = x + pos_embedding

        # Transformer Encoding
        for layer in self.layers:
            x = layer(x)

        # Return encoded features
        return x

# Create Vision Transformer 3D Decoder
class ViTDecoder3d(nn.Module):
    def __init__(self, 
        out_channels,
        shape=(64, 64, 64), scale=1, patch_size=None, shape_patch_ratio=16,
        n_features=128, n_heads=4, n_layers=8,
    ):
        super(ViTDecoder3d, self).__init__()

        ### Check inputs ###
        # Check number of heads
        if n_features % n_heads != 0:
            # n_features must be divisible by n_heads
            raise ValueError('Number of features must be divisible by number of heads!')
        # Check shape
        if not isinstance(shape, tuple):
            # Convert shape to tuple
            shape = (shape, shape, shape)
        # Check patch size
        if patch_size is None:
            # Set default patch size (1/shape_patch_ratio of shape)
            patch_size = tuple(shape[i] // shape_patch_ratio for i in range(3))
        elif not isinstance(patch_size, tuple):
            # Convert patch size to tuple
            patch_size = (patch_size, patch_size, patch_size)
        # Check shape and patch size compatibility
        for i in range(3):
            if shape[i] % patch_size[i] != 0:
                # Check if shape is divisible by patch size
                raise ValueError('Shape must be divisible by patch size!')
        
        # Set attributes
        self.out_channels = out_channels
        self.shape = shape
        self.scale = scale
        self.patch_size = patch_size
        self.patch_size_ratio = shape_patch_ratio
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        shape_patchgrid = (
            shape[0] // (scale * patch_size[0]),
            shape[1] // (scale * patch_size[1]),
            shape[2] // (scale * patch_size[2]),
        )
        self.shape_patchgrid = shape_patchgrid

        # Create output blocks
        self.output_block = nn.Sequential(
            # Expand patches to volume
            VolumeExpand3d(n_features=n_features, scale=patch_size[0]),
            # Rescale volume
            VolumeExpand3d(n_features=n_features, scale=scale),
            # Project to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1), 
        )

        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers//2):
            self.layers.append(
                TransformerBlock(n_features=n_features, n_heads=n_heads)
            )

    def forward(self, x):

        # Transformer Decoding
        for layer in self.layers:
            x = layer(x)

        # Output block
        x = x.transpose(1, 2).reshape(-1, self.n_features, *self.shape_patchgrid)
        x = self.output_block(x)

        # Return output
        return x

# Create class
class ViT3d(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(64, 64, 64), scale=1, patch_size=None, shape_patch_ratio=16,
        n_features=128, n_heads=4, n_layers=16,
    ):
        super(ViT3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.scale = scale
        self.patch_size = patch_size
        self.patch_size_ratio = shape_patch_ratio
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Create encoder and decoder
        self.encoder = ViTEncoder3d(
            in_channels=in_channels,
            shape=shape, scale=scale, patch_size=patch_size,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers//2,
        )
        self.decoder = ViTDecoder3d(
            out_channels=out_channels,
            shape=shape, scale=scale, patch_size=patch_size,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers//2,
        )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'shape': self.shape,
            'scale': self.scale,
            'patch_size': self.patch_size,
            'shape_patch_ratio': self.patch_size_ratio,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
        }

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)

    # Set constants
    shape = (128, 128, 128)
    in_channels = 36
    out_channels = 1

    # Create data
    x = torch.randn(1, in_channels, *shape)

    # Create a model
    model = ViT3d(
        in_channels=in_channels,
        out_channels=out_channels,
        shape=shape, 
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

