
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False, beta=.5):
        super(ConvBlock, self).__init__()

        # Check inputs
        if upsample and downsample:
            raise ValueError('Cannot upsample and downsample at the same time.')

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.beta = beta

        # Define residual layer
        if upsample:
            self.residual = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif downsample:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        elif in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        # Define convolutional layers
        if upsample:
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )
        elif downsample:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        
        # Residual connection
        x0 = self.residual(x)

        # Convolutional block
        x = self.conv(x)

        # Combine with residual
        x = self.beta * x + (1 - self.beta) * x0

        # Return the output
        return x


# Define simple 3D Unet model
class Simple3DUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=8, n_blocks=3, n_layers_per_block=2):
        super(Simple3DUnet, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Define input block
        block = [

        ]
        self.input_block = nn.Sequential(
            # Merge input channels to n_features for each voxel
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            nn.InstanceNorm3d(n_features),
            nn.ReLU(inplace=True),
            # Additional convolutional layers
            *(ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1))
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
        self.bottleneck = ConvBlock(n_features, 2*n_features)

        # Define upsample blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.up_blocks.append(
                nn.Sequential(
                    # Upsample layer
                    ConvBlock(2*n_features, n_features, upsample=True),
                    # Additional convolutional layers
                    *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1)]
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            ConvBlock(2*n_features, n_features),
            *[ConvBlock(n_features, n_features) for _ in range(n_layers_per_block - 1)],
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )
        
    def forward(self, x):

        # Initialize skip connections
        skips = []

        # Input block
        x = self.input_block(x)
        skips.append(x)

        # Downsample blocks
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            skips.append(x)

        # Bottleneck block
        x = skips.pop()
        x = self.bottleneck(x)

        # Upsample blocks
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            x_skip = skips.pop()
            x = torch.cat([x, x_skip], dim=1)

        # Output block
        x = self.output_block(x)

        # Return the output
        return x




# Test the model
if __name__ == '__main__':

    # Create a model
    model = Simple3DUnet(30, 1)

    # Create a random input
    x = torch.randn(1, 30, 128, 128, 128)
    y = model(x)
    print(y.shape)

    # Done
    print('Done!')

