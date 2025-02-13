
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
        shape=(64, 64, 64), patch_size=(4, 4, 4), downscaling_factor=2,
        embed_dim=64, n_heads=2, n_layers=6,
    ):
        super(ViT3D, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.patch_size = patch_size
        self.downscaling_factor = downscaling_factor
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate constants
        self.shape_downscaled = (
            shape[0] // downscaling_factor,
            shape[1] // downscaling_factor,
            shape[2] // downscaling_factor,
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
                in_channels*downscaling_factor, 
                kernel_size=downscaling_factor, 
                stride=downscaling_factor,
                groups=in_channels  # Grouped convolutions for channel-wise downscaling
            ),
            # Mix channels
            nn.Conv3d(
                in_channels*downscaling_factor, 
                in_channels*downscaling_factor, 
                kernel_size=1
            )
        )
        self.upscale = nn.Sequential(
            # Upscale
            nn.ConvTranspose3d(
                out_channels*downscaling_factor,
                out_channels*downscaling_factor,
                kernel_size=downscaling_factor,
                stride=downscaling_factor
            ),
            # Smooth
            nn.Conv3d(
                out_channels*downscaling_factor, 
                out_channels*downscaling_factor, 
                kernel_size=3, 
                padding=1
            ),
            # Project to output channels
            nn.Conv3d(
                out_channels*downscaling_factor, 
                out_channels, 
                kernel_size=3, 
                padding=1
            ), 
        )
        
        # 3D Patch Embedding and Unembedding Layers
        self.patch_embed = nn.Sequential(
            # Transform channels to embed_dim
            nn.Conv3d(in_channels*downscaling_factor, embed_dim, kernel_size=1),
            # Embed volume into patches
            nn.Conv3d(
                embed_dim, 
                embed_dim,
                kernel_size=patch_size, 
                stride=patch_size,
                groups=embed_dim  # Grouped convolutions for channel-wise patching
            )
        )
        self.patch_unembed = nn.ConvTranspose3d(
            embed_dim, 
            out_channels*downscaling_factor, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.n_patches, embed_dim))

        # Transformer Encoders
        self.transformers = nn.ModuleList()
        for _ in range(n_layers):
            self.transformers.append(
                TransformerBlock(embed_dim, n_heads)
            )

    def forward(self, x):

        # Downscale input
        x = self.downscale(x)

        # Patch embedding
        x = self.patch_embed(x)  # Shape: [B, embed_dim, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, n_patches, embed_dim]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        for transformer in self.transformers:
            x = x + transformer(x)

        # Patch unembedding
        x = x.transpose(1, 2).reshape(-1, self.embed_dim, *self.shape_patchgrid)
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
        downscaling_factor=2,
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

