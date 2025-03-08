
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from architectures.blocks import ConvformerBlock3d


# Define full convolutional transformer model
class ConvformerModel(nn.Module):
    """Full Convolutional Transformer model"""
    def __init__(self,
        in_channels, out_channels,
        n_features=8, n_layers=4, n_heads=2, kernel_size=3
    ):
        super(ConvformerModel, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        # Define input block
        self.input_block = nn.Sequential(
            # # Normalize
            # nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    ConvformerBlock3d(n_features, kernel_size=kernel_size, n_heads=n_heads)
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):
        
        # Input block
        x = self.input_block(x)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage

    # Create a model
    model = ConvformerModel(36, 1)

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

