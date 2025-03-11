
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from architectures.blocks import TransformerBlock


# Create class
class ViT3D(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(64, 64, 64), patch_size=8, patch_stride=None,
        n_features=128, n_heads=4, n_layers=8,
    ):
        super(ViT3D, self).__init__()

        # Check inputs
        if n_layers % 2 != 0:
            # Number of layers must be even
            raise ValueError('Number of layers must be even!')
        if not isinstance(shape, tuple):
            # Convert shape to tuple
            shape = (shape, shape, shape)
        if not isinstance(patch_size, tuple):
            # Convert patch size to tuple
            patch_size = (patch_size, patch_size, patch_size)
        if patch_stride is None:
            # Set default patch stride (1/2 of patch size)
            patch_stride = (patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        elif not isinstance(patch_stride, tuple):
            # Convert patch stride to tuple
            patch_stride = (patch_stride, patch_stride, patch_stride)
        for i in range(3):
            # Check if shape is divisible by patch size
            if shape[i] % patch_size[i] != 0:
                raise ValueError('Shape must be divisible by patch size!')
            # Check if patch size is divisible by patch stride
            if patch_size[i] % patch_stride[i] != 0:
                raise ValueError('Patch size must be divisible by patch stride!')
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.patch_size = patch_size
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        shape_patchgrid = (
            (shape[0] - patch_size[0]) // patch_stride[0] + 1,
            (shape[1] - patch_size[1]) // patch_stride[1] + 1,
            (shape[2] - patch_size[2]) // patch_stride[2] + 1,
        )
        n_patches = shape_patchgrid[0] * shape_patchgrid[1] * shape_patchgrid[2]
        self.path_stride = patch_stride
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
                groups=n_features,
            ),
            # Project to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
        # 3D Patch Embedding and Unembedding Layers
        self.patch_embed = nn.Sequential(
            nn.Conv3d(
                n_features, 
                n_features,
                kernel_size=patch_size, 
                stride=patch_stride,
                groups=n_features  # Channel-wise patching
            )
        )
        self.patch_unembed = nn.Sequential(
            nn.ConvTranspose3d(
                n_features, 
                n_features, 
                kernel_size=patch_size, 
                stride=patch_stride,
                groups=n_features  # Channel-wise unpatching
            )
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.n_patches, n_features))

        # Transformer Encoders
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                n_features, n_heads, 
                dim_feedforward=n_features,
                batch_first=True
            ),
            num_layers=n_layers//2
        )

        # Transformer Decoders
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                n_features, n_heads, 
                dim_feedforward=n_features,
                batch_first=True
            ),
            num_layers=n_layers//2
        )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'shape': self.shape,
            'patch_size': self.patch_size,
            'n_features': self.n_features,
            'n_features_top': self.n_features_top,
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
        x = self.transformer_encoder(x)

        # Return encoded features
        return x
    
    def decoder(self, x):

        # Transformer Decoding
        x = self.transformer_decoder(x)

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
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    in_channels = 36
    out_channels = 1
    shape = (256, 256, 256)

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
    estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

