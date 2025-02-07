
# Import libraries
import torch
import torch.nn as nn


# Create class
class ViT3D(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(128, 128, 128), patch_size=(8, 8, 8), downscaling_factor=2,
        embed_dim=64, num_heads=8, num_layers=6,
    ):
        super(ViT3D, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = shape
        self.patch_size = patch_size
        self.downscaling_factor = downscaling_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

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
        self.num_patches = self.shape_patchgrid[0] * self.shape_patchgrid[1] * self.shape_patchgrid[2]

        # Create downscaling and upscaling layers
        self.downscale = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                in_channels*downscaling_factor, 
                kernel_size=downscaling_factor, 
                stride=downscaling_factor
            ),
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose3d(
                out_channels*downscaling_factor,
                out_channels*downscaling_factor,
                kernel_size=downscaling_factor,
                stride=downscaling_factor
            ),
            nn.Conv3d(out_channels*downscaling_factor, out_channels, kernel_size=3, padding=1),  # Smooth output
        )
        
        # 3D Patch Embedding and Unembedding Layers
        self.patch_embed = nn.Conv3d(
            in_channels*downscaling_factor, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.patch_unembed = nn.ConvTranspose3d(
            embed_dim, 
            out_channels*downscaling_factor, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim),
            num_layers=num_layers
        )

        # Create upscaling layer

    def forward(self, x):

        # Downscale input
        x = self.downscale(x)

        # Patch embeddin
        x = self.patch_embed(x)  # Shape: [B, embed_dim, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, embed_dim]

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)

        # Transformer Encoding
        x = self.transformer(x)

        # Patch unembedding
        x = x.transpose(1, 2).reshape(-1, self.embed_dim, *self.shape_patchgrid)
        x = self.patch_unembed(x)

        # Upscale output
        x = self.upscale(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Create a model
    model = ViT3D(30, 1, shape=(128, 128, 128), patch_size=(8, 8, 8), downscaling_factor=2)

    # Create a random input
    x = torch.randn(1, 30, 128, 128, 128)
    y = model(x)

    # Done
    print('Done!')
