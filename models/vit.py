
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from models.blocks import TransformerBlock


# Create class
class ViT3D(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(128, 128, 128), scale=2, patch_size=(4, 4, 4),
        n_features=64, n_heads=8, n_layers=16,
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
        self.shape_downscaled = (
            shape[0] // scale,
            shape[1] // scale,
            shape[2] // scale,
        )
        self.shape_patchgrid = (
            self.shape_downscaled[0] // patch_size[0],
            self.shape_downscaled[1] // patch_size[1],
            self.shape_downscaled[2] // patch_size[2],
        )
        self.n_patches = self.shape_patchgrid[0] * self.shape_patchgrid[1] * self.shape_patchgrid[2]

        # Create input block
        self.input_block = nn.Sequential(
            # # Normalize
            # nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
        )

        # Create output block
        self.output_block = nn.Sequential(
            # Mix channels
            nn.Conv3d(n_features, n_features, kernel_size=1),
            # Smooth output
            nn.Conv3d(
                n_features, n_features, 
                kernel_size=3, padding=1,
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
                stride=patch_size,
                groups=n_features  # Channel-wise patching
            )
        )
        self.patch_unembed = nn.Sequential(
            nn.ConvTranspose3d(
                n_features, 
                n_features, 
                kernel_size=patch_size, 
                stride=patch_size,
                groups=n_features  # Channel-wise unpatching
            )
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.n_patches, n_features))

        # Transformer Encoders
        self.transfomer_encoders = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transfomer_encoders.append(
                TransformerBlock(n_features, n_heads)
            )

        # Transformer Decoders
        self.transfomer_decoders = nn.ModuleList()
        for _ in range(n_layers//2):
            self.transfomer_decoders.append(
                TransformerBlock(n_features, n_heads)
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):

        # Input block
        x = self.input_block(x)

        # Patch embedding
        x = self.downscale(x)
        x = self.patch_embed(x)  # Shape: [B, n_features, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, n_patches, n_features]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        for transformer in self.transfomer_encoders:
            x = transformer(x)

        # Return encoded features
        return x
    
    def decode(self, x):

        # Transformer Decoding
        for transformer in self.transfomer_decoders:
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
        shape=(128, 128, 128), patch_size=(8, 8, 8), 
        scale=1,
    )

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

