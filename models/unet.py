
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from models.blocks import ConvBlock


# Define simple 3D Unet model
class Unet3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=8, n_blocks=4, n_layers_per_block=4):
        super(Unet3D, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Get n_features per depth
        n_features_per_depth = [n_features * (i+1) for i in range(n_blocks+1)]
        self.n_features_per_depth = n_features_per_depth

        # Define input block
        self.input_block = nn.Sequential(
            # # Normalize
            # nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Additional convolutional layers
            *(ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1))
        )

        # Define downsample blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_blocks):
            n_in = n_features_per_depth[i]
            n_out = n_features_per_depth[i+1]
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    ConvBlock(n_in, n_out, downsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_out, n_out) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define bottleneck block
        n_in = n_features_per_depth[-1]
        n_out = n_features_per_depth[-1]
        self.bottleneck = ConvBlock(n_in, n_out)

        # Define upsample blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks, 0, -1):
            n_in = n_features_per_depth[i] * (1 if i == n_blocks else 2)
            n_out = n_features_per_depth[i-1]
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample layer
                    ConvBlock(n_in, n_out, upsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_out, n_out) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            ConvBlock(2*n_features, n_features),
            *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1)],
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
    def forward(self, x):

        # Encode
        feats = self.encoder(x)

        # Decode
        x = self.decoder(feats)

        # Return the output
        return x
        
    def encoder(self, x):

        # Initialize features list
        feats = []

        # Input block
        x = self.input_block(x)
        feats.append(x)

        # Downsample blocks
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            feats.append(x)

        # Bottleneck block
        x = feats.pop()
        x = self.bottleneck(x)
        feats.append(x)

        # Return the features
        return feats
    
    def decoder(self, feats):

        # Get x
        x = feats.pop()

        # Upsample blocks
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            x_skip = feats.pop()
            x = torch.cat([x, x_skip], dim=1)

        # Output block
        x = self.output_block(x)

        # Return the output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage

    # Create a model
    model = Unet3D(36, 1)

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

