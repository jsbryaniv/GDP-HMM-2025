
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from architectures.vit import ViT3d, ViTEncoder3d
from architectures.blocks import TransformerBlock, CrossTransformerBlock


# Define cross attention vistion transformer model
class CrossViT3d(nn.Module):
    """Cross attention vistion transformer model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        shape=128, scale=1, ratio_shape_patch=8, n_features=16, n_heads=4, 
        n_layers=8, n_layers_mixing=8,
    ):
        super(CrossViT3d, self).__init__()
            
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.shape = shape
        self.scale = scale
        self.ratio_shape_patch = ratio_shape_patch
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_mixing = n_layers_mixing

        # Create main vit
        self.main_vit = ViT3d(
            in_channels, out_channels,
            shape=shape, scale=scale, ratio_shape_patch=ratio_shape_patch,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers,
        )

        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_encoders.append(
                ViTEncoder3d(
                    n_channels,
                    shape=shape, scale=scale, ratio_shape_patch=ratio_shape_patch,
                    n_features=n_features, n_heads=n_heads, n_layers=n_layers,
                )
            )

        # Create mixing block
        self.mixing_blocks = nn.ModuleList()
        for _ in range(n_layers_mixing):
            self.mixing_blocks.append(
                CrossTransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                )
            )
    
    def get_config(self):
        """Get configuration."""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'shape': self.shape,
            'scale': self.scale,
            'ratio_shape_patch': self.ratio_shape_patch,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_layers_mixing': self.n_layers_mixing,
        }

    def forward(self, x, *y_list):
        """
        x is the input tensor
        y_list is a list of context tensors.
        """

        # Encode input
        x = self.main_vit.encoder(x)
        context = sum(block(y) for block, y in zip(self.context_encoders, y_list))

        # Add positional embeddings
        pos_embedding = (
            self.main_vit.encoder.pos_embedding_0 
            + self.main_vit.encoder.pos_embedding_1 
            + self.main_vit.encoder.pos_embedding_2
        )
        pos_embedding = pos_embedding.flatten(2).transpose(1, 2).expand(x.shape[0], -1, -1)
        x = x + pos_embedding 
        context = context + pos_embedding

        # Mixing block
        for block in self.mixing_blocks:
            x = block(x, context)

        # Decode
        x = self.main_vit.decoder(x)
        
        # Return
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    shape = (64, 64, 64)
    in_channels = 4
    out_channels = 1
    n_channels_context = [1, 4, 30]

    # Create data
    x = torch.randn(1, in_channels, *shape)
    y_list = [torch.randn(1, n, *shape) for n in n_channels_context]

    # Create a model
    model = CrossViT3d(
        in_channels, 
        out_channels, 
        n_channels_context,
        shape=shape
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        y = model(x, *y_list)

    # Estimate memory usage
    estimate_memory_usage(model, x, *y_list, print_stats=True)

    # Done
    print('Done!')

