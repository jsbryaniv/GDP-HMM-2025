
# Import libraries
import torch
import torch.nn as nn

# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=4, expansion=2):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.ReLU(),
            nn.Linear(n_features_inner, n_features),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):

        # Apply self-attention
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        return x

# Create class
class ViT3D(nn.Module):
    def __init__(self, 
        in_channels, out_channels,
        shape=(128, 128, 128), patch_size=(4, 4, 4), downscaling_factor=4,
        embed_dim=64, num_heads=2, num_layers=6,
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
            # Normalize
            nn.GroupNorm(in_channels, in_channels),
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
            # Smooth
            nn.Conv3d(
                out_channels*downscaling_factor, 
                out_channels*downscaling_factor, 
                kernel_size=3, 
                padding=1
            ),
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
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, self.num_patches, embed_dim))

        # Transformer Encoders
        self.transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(
                TransformerBlock(embed_dim, num_heads)
            )

    def forward(self, x):

        # Downscale input
        x = self.downscale(x)

        # Patch embedding
        print(x.shape)
        print(self.embed_dim)

        x = self.patch_embed(x)  # Shape: [B, embed_dim, D//pD, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, embed_dim]

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

    # Create a model
    model = ViT3D(30, 1, shape=(128, 128, 128), patch_size=(8, 8, 8), downscaling_factor=2)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} parameters')

    # Create data
    x = torch.randn(1, 30, 128, 128, 128)

    # Forward pass
    y = model(x)

    # Done
    print('Done!')

