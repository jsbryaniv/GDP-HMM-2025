
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from architectures.blocks import TransformerBlock, VolumeExpandSparse3d, VolumeContractSparse3d


# Create class
class ViT3D(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(64, 64, 64), patch_size=None, patch_buffer=None,
        n_features=128, n_heads=4, n_layers=16,
    ):
        super(ViT3D, self).__init__()

        ### Check inputs ###
        # Check input channels
        if n_layers % 2 != 0:
            # Number of layers must be even
            raise ValueError('Number of layers must be even!')
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
            # Set default patch size (1/16 of shape)
            patch_size = (shape[0] // 16, shape[1] // 16, shape[2] // 16)
        elif not isinstance(patch_size, tuple):
            # Convert patch size to tuple
            patch_size = (patch_size, patch_size, patch_size)
        # Check patch buffer
        if patch_buffer is None:
            # Set default patch buffer (1/4 of patch size)
            patch_buffer = (patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4)
        elif not isinstance(patch_buffer, tuple):
            # Convert patch buffer to tuple
            patch_buffer = (patch_buffer, patch_buffer, patch_buffer)
        # Check shape, patch size, and patch buffer compatibility
        for i in range(3):
            if shape[i] % patch_size[i] != 0:
                # Check if shape is divisible by patch size
                raise ValueError('Shape must be divisible by patch size!')
            if (patch_buffer[i] > 0) and (patch_size[i] % patch_buffer[i] != 0):
                # Check if patch size is divisible by patch buffer
                raise ValueError('Patch size must be divisible by patch buffer!')
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.patch_size = patch_size
        self.patch_buffer = patch_buffer
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        patch_size_effective = (
            patch_size[0] + 2*patch_buffer[0],
            patch_size[1] + 2*patch_buffer[1],
            patch_size[2] + 2*patch_buffer[2],
        )
        shape_patchgrid = (
            shape[0] // patch_size[0],
            shape[1] // patch_size[1],
            shape[2] // patch_size[2],
        )
        n_patches = shape_patchgrid[0] * shape_patchgrid[1] * shape_patchgrid[2]
        self.patch_size_effective = patch_size_effective
        self.shape_patchgrid = shape_patchgrid
        self.n_patches = n_patches
        # Check it n_patches is too large
        if n_patches > 10000:
            raise ValueError('Number of patches is too large! Descrease size or increase patch size and stride.')

        # Create input and output blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
        )
        self.output_block = nn.Sequential(
            # Smooth output
            nn.Conv3d(
                n_features, n_features, 
                kernel_size=3, padding=1,
                groups=n_features//2,
            ),
            # Project to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
        # 3D Patch Embedding and Unembedding Layers
        self.patch_embed = VolumeContractSparse3d(
            n_features=n_features,
            scale=patch_size[0],
            buffer=patch_size[0]//2,
        )
        self.patch_unembed = VolumeExpandSparse3d(
            n_features=n_features, 
            scale=patch_size[0],
            buffer=patch_size[0]//2,
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.n_patches, n_features))

        # Transformer Encoders
        self.transformer_encoder = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transformer_encoder.append(
                TransformerBlock(n_features=n_features, n_heads=n_heads)
            )

        # Transformer Decoders
        self.transformer_decoder = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transformer_decoder.append(
                TransformerBlock(n_features=n_features, n_heads=n_heads)
            )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'shape': self.shape,
            'patch_size': self.patch_size,
            'patch_buffer': self.patch_buffer,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
        }

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encoder(self, x):

        # Input block
        x = self.input_block(x)

        # Patch embedding
        x = self.patch_embed(x)           # Shape: [B, n_features, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, n_patches, n_features]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        for layer in self.transformer_encoder:
            x = layer(x)
        # x = self.transformer_encoder(x)

        # Return encoded features
        return x
    
    def decoder(self, x):

        # Transformer Decoding
        for layer in self.transformer_decoder:
            x = layer(x)
        # x = self.transformer_decoder(x)

        # Patch unembedding
        x = x.transpose(1, 2).reshape(-1, self.n_features, *self.shape_patchgrid)
        x = self.patch_unembed(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)

    # Set constants
    shape = (64, 64, 64)
    in_channels = 36
    out_channels = 1

    # Create data
    x = torch.randn(1, in_channels, *shape)

    # Create a model
    model = ViT3D(
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

    # Estimate memory usage
    estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

