
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
        shape=(128, 128, 128), scale=2, patch_size=(8, 8, 8),
        n_features=64, n_heads=8, n_layers=8,
    ):
        super(ViT3D, self).__init__()

        # Check inputs
        if n_layers % 2 != 0:
            raise ValueError('Number of layers must be even!')
        if not isinstance(shape, tuple):
            shape = (shape, shape, shape)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size, patch_size)
        for i in range(3):
            if shape[i] % patch_size[i] != 0:
                raise ValueError('Shape must be divisible by patch size!')
        
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.patch_size = patch_size
        self.downscaling_factor = scale
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        patch_stride = (patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        shape_downscaled = (
            shape[0] // scale,
            shape[1] // scale,
            shape[2] // scale,
        )
        shape_patchgrid = (
            (shape_downscaled[0] - patch_size[0]) // patch_stride[0] + 1,
            (shape_downscaled[1] - patch_size[1]) // patch_stride[1] + 1,
            (shape_downscaled[2] - patch_size[2]) // patch_stride[2] + 1,
        )
        n_patches = shape_patchgrid[0] * shape_patchgrid[1] * shape_patchgrid[2]
        self.path_stride = patch_stride
        self.shape_downscaled = shape_downscaled
        self.shape_patchgrid = shape_patchgrid
        self.n_patches = n_patches

        # Create input and output blocks
        self.input_block = nn.Sequential(
            # # Normalize
            # nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
        )
        self.output_block = nn.Sequential(
            # Mix channels
            nn.Conv3d(n_features, n_features, kernel_size=1),
            # Smooth output
            nn.Conv3d(
                n_features, n_features, 
                kernel_size=max(patch_size), padding=max(patch_size)//2,
                groups=n_features//n_heads,
            ),
            # Project to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )

        # Create downscaling and upscaling layers
        self.downscale = nn.Sequential(
            nn.Conv3d(
                n_features, 
                n_features, 
                kernel_size=scale, 
                stride=scale,
                groups=n_features  # Channel-wise downscaling
            )
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose3d(
                n_features,
                n_features,
                kernel_size=scale,
                stride=scale,
                groups=n_features  # Channel-wise upscaling
            )
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
        self.transformer_encoders = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transformer_encoders.append(
                TransformerBlock(n_features, n_heads)
            )

        # Transformer Decoders
        self.transformer_decoders = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transformer_decoders.append(
                TransformerBlock(n_features, n_heads)
            )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'shape': self.shape,
            'scale': self.downscaling_factor,
            'patch_size': self.patch_size,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
        }

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):

        # Input block
        x = self.input_block(x)

        # Patch embedding
        x = self.downscale(x)
        x = self.patch_embed(x)           # Shape: [B, n_features, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, n_patches, n_features]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        for transformer in self.transformer_encoders:
            x = transformer(x)

        # Return encoded features
        return x
    
    def decode(self, x):

        # Transformer Decoding
        for transformer in self.transformer_decoders:
            x = transformer(x)

        # Patch unembedding
        x = x.transpose(1, 2).reshape(-1, self.n_features, *self.shape_patchgrid)
        x = self.patch_unembed(x)
        x = self.upscale(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage

    # Create a model
    model = ViT3D(
        36, 1, 
        shape=(128, 128, 128), 
    )
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')

    # Create data
    x = torch.randn(1, 36, 128, 128, 128)

    # Forward pass
    y = model(x)

    # Backward pass
    loss = y.sum()
    loss.backward()

    # Estimate memory usage
    estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

