
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
        shape=(64, 64, 64), scale=2, patch_size=(4, 4, 4),
        n_features=64, n_heads=4, n_layers=8,
    ):
        super(ViT3D, self).__init__()

        # Check inputs
        if not isinstance(shape, tuple):
            shape = (shape, shape, shape)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size, patch_size)
        
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

        # Create downscaling and upscaling layers
        self.downscale = nn.Sequential(
            # # Normalize
            # nn.GroupNorm(in_channels, in_channels),
            # Downscale along channels
            nn.Conv3d(
                in_channels, 
                in_channels*scale, 
                kernel_size=scale, 
                stride=scale,
                groups=in_channels  # Grouped convolutions for channel-wise downscaling
            ),
            # Mix channels
            nn.Conv3d(
                in_channels*scale, 
                in_channels*scale, 
                kernel_size=1
            )
        )
        self.upscale = nn.Sequential(
            # Upscale
            nn.ConvTranspose3d(
                out_channels*scale,
                out_channels*scale,
                kernel_size=scale,
                stride=scale
            ),
            # Smooth
            nn.Conv3d(
                out_channels*scale, 
                out_channels*scale, 
                kernel_size=3, 
                padding=1
            ),
            # Project to output channels
            nn.Conv3d(
                out_channels*scale, 
                out_channels, 
                kernel_size=3, 
                padding=1
            ), 
        )
        
        # 3D Patch Embedding and Unembedding Layers
        self.patch_embed = nn.Sequential(
            # Transform channels to n_features
            nn.Conv3d(in_channels*scale, n_features, kernel_size=1),
            # Embed volume into patches
            nn.Conv3d(
                n_features, 
                n_features,
                kernel_size=patch_size, 
                stride=patch_size,
                groups=n_features  # Grouped convolutions for channel-wise patching
            )
        )
        self.patch_unembed = nn.ConvTranspose3d(
            n_features, 
            out_channels*scale, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.n_patches, n_features))

        # Transformer Encoders
        self.transformers = nn.ModuleList()
        for _ in range(n_layers):
            self.transformers.append(
                TransformerBlock(n_features, n_heads)
            )

    def forward(self, x):

        # Downscale input
        x = self.downscale(x)

        # Patch embedding
        x = self.patch_embed(x)  # Shape: [B, n_features, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, n_patches, n_features]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        for transformer in self.transformers:
            x = x + transformer(x)

        # Patch unembedding
        x = x.transpose(1, 2).reshape(-1, self.n_features, *self.shape_patchgrid)
        x = self.patch_unembed(x)

        # Upscale output
        x = self.upscale(x)

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
        scale=2,
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

