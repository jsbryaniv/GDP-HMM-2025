
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from models.blocks import ConvBlock, ConvAttn3d, ConvformerBlock3d


# Define full convolutional transformer model
class UConvformerModel(nn.Module):
    """Full Convolutional Transformer model"""
    def __init__(self,
        in_channels, out_channels,
        n_features=8, n_blocks=3, n_layers_per_block=3
    ):
        super(UConvformerModel, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Define input block
        self.input_block = nn.Sequential(
            # Normalize
            nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Downsample
            ConvBlock(n_features, n_features, downsample=True),
        )

        # Define downsample blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    ConvBlock(n_features, n_features, downsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define bottleneck block
        self.bottleneck = ConvformerBlock3d(n_features)

        # Define upsample, qkv, attention, and mlp blocks
        self.up_blocks = nn.ModuleList()
        self.q_proj_blocks = nn.ModuleList()
        self.kv_proj_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.mlp_blocks = nn.ModuleList()
        for i in range(n_blocks):
            # Upsample block
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample layer
                    ConvBlock(n_features, n_features, upsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1)]
                )
            )
            # Q projection block
            self.q_proj_blocks.append(
                nn.Sequential(
                    nn.InstanceNorm3d(n_features),
                    nn.Conv3d(n_features, n_features, kernel_size=1)
                )
            )
            # KV projection block
            self.kv_proj_blocks.append(
                nn.Sequential(
                    nn.InstanceNorm3d(n_features),
                    nn.Conv3d(n_features, 2*n_features, kernel_size=1)
                )
            )
            # Attention block
            self.attn_blocks.append(
                ConvAttn3d(n_features)
            )
            # MLP block
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Conv3d(n_features, n_features, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(n_features, n_features, kernel_size=1),
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Upsample
            ConvBlock(n_features, n_features, upsample=True),
            # Smooth
            nn.Conv3d(n_features, n_features, kernel_size=3, padding=1),
            # Project to output channels
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):

        # Initialize skip connections
        skips = []

        # Input block
        x = self.input_block(x)
        skips.append(x)

        # Downsample blocks
        for i in range(self.n_blocks):
            x = self.down_blocks[i](x)
            skips.append(x)

        # Bottleneck block
        x = skips.pop()
        x = self.bottleneck(x)

        # Upsample blocks
        for i in range(self.n_blocks):
            # Upsample
            x = self.up_blocks[i](x)
            # Get skip
            x_skip = skips.pop()
            # Get query from x
            Q = self.q_proj_blocks[i](x)
            # Get key and value from skip
            K, V = self.kv_proj_blocks[i](x_skip).chunk(2, dim=1)
            # Apply attention
            x = x + self.attn_blocks[i](Q, K, V)
            # Apply MLP
            x = x + self.mlp_blocks[i](x)

        # Output block
        x = self.output_block(x)

        # Return the output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils import estimate_memory_usage

    # Create a model
    model = UConvformerModel(30, 1)

    # Create data
    x = torch.randn(1, 30, 128, 128, 128)

    # Forward pass
    y = model(x)

    # Estimate memory usage
    estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

