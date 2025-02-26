
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
    def __init__(self, 
        in_channels, out_channels, 
        n_features=16, n_groups=4, n_blocks=4, n_layers_per_block=4, 
    ):
        super(Unet3D, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_groups = n_groups
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
        for depth in range(n_blocks):
            n_in = n_features_per_depth[depth]
            n_out = n_features_per_depth[depth+1]
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    ConvBlock(n_in, n_out, groups=n_groups, downsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_out, n_out, groups=n_groups) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define bottleneck block
        n_in = n_features_per_depth[-1]
        n_out = n_features_per_depth[-1]
        self.bottleneck = ConvBlock(n_in, n_out, groups=n_groups)

        # Define upsample and merge blocks
        self.up_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        for i in range(n_blocks):
            depth = self.n_blocks - 1 - i
            n_in = n_features_per_depth[depth+1]
            n_out = n_features_per_depth[depth]
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample layer
                    ConvBlock(n_in, n_out, groups=n_groups, upsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_out, n_out, groups=n_groups) for _ in range(n_layers_per_block - 1)]
                )
            )
            self.merge_blocks.append(
                nn.Sequential(
                    ConvBlock(2*n_out, n_out, kernel_size=1),
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block)],
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats)
        return x
        
    def encoder(self, x):

        # Initialize features list
        feats = []

        # Input block
        x = self.input_block(x)
        feats.append(x)

        # Downsample blocks
        for depth, block in enumerate(self.down_blocks):
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
            # Upsample
            x = block(x)
            # Merge with skip
            x_skip = feats.pop()
            x = torch.cat([x, x_skip], dim=1)
            x = self.merge_blocks[i](x)

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

