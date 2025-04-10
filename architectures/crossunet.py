
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Import custom libraries
from architectures.unet import Unet3d, UnetEncoder3d
from architectures.blocks import ConvformerDecoder3d


# Define cross attention unet model
class CrossUnetModel(nn.Module):
    """Cross attention Unet model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        n_features=8, n_blocks=5, n_layers_per_block=2,
        n_attn_repeats=2, attn_kernel_size=3,
        scale=1, conv_block_type=None, use_dropout=False,
        feature_scale=None,
    ):
        super(CrossUnetModel, self).__init__()

        # Set default values
        if conv_block_type is None:
            conv_block_type = 'ConvBlock3d'
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_attn_repeats = n_attn_repeats
        self.attn_kernel_size = attn_kernel_size
        self.scale = scale
        self.use_dropout = use_dropout
        self.conv_block_type = conv_block_type
        self.feature_scale = feature_scale

        # Get constants
        n_context = len(n_cross_channels_list)
        self.n_context = n_context

        # Create main unet
        self.main_unet = Unet3d(
            in_channels, out_channels, 
            n_features=n_features, n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale, conv_block_type=conv_block_type, use_dropout=use_dropout,
            feature_scale=feature_scale,
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_encoders.append(
                UnetEncoder3d(
                    n_channels, 
                    n_features=n_features, n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block,
                    scale=scale, use_dropout=use_dropout,
                    feature_scale=feature_scale,
                )
            )

        # Get features per depth
        self.n_features_per_depth = self.main_unet.n_features_per_depth

        # Create cross attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.cross_attn_blocks.append(
                ConvformerDecoder3d(
                    self.n_features_per_depth[depth], 
                    kernel_size=attn_kernel_size,
                    n_layers=n_attn_repeats+depth,  # +depth to increase number of layers
                    n_heads=max(1, min(self.n_features_per_depth[depth] // 8, 4)),
                    dropout=.2 if use_dropout else 0,
                )
            )
    
    def get_config(self):
        """Get configuration."""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
            'n_attn_repeats': self.n_attn_repeats,
            'attn_kernel_size': self.attn_kernel_size,
            'scale': self.scale,
            'use_dropout': self.use_dropout,
            'conv_block_type': self.conv_block_type,
            'feature_scale': self.feature_scale,
        }

    def encode_context(self, *y_list):
        y_list = [y.requires_grad_() for y in y_list]
        return checkpoint(self._encode_context, *y_list, use_reentrant=False)
    
    def _encode_context(self, *y_list):
        feats_context = [block(y.float()) for block, y in zip(self.context_encoders, y_list)]
        feats_context = [sum([f for f in row]) / len(row) for row in zip(*feats_context)]
        return feats_context

    def forward(self, x, *y_list, feats_context=None):
        """
        x is the input tensor
        y_list is a list of context tensors
        feats_context is the context features (optional for pre-computed context)
        """

        # Encode x
        feats = self.main_unet.encoder(x)
        x = feats.pop()

        # Encode y_list and sum features at each depth
        if feats_context is None:
            feats_context = self.encode_context(*y_list)

        # Apply context
        depth = self.n_blocks
        fcon = feats_context.pop()
        x = self.cross_attn_blocks[depth](x, fcon)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            upblock = self.main_unet.decoder.up_blocks[i]
            catblock = self.main_unet.decoder.cat_blocks[i]
            # Upsample
            x = upblock(x)
            # Merge with skip
            x_skip = feats.pop()               # Get skip connection
            x = torch.cat([x, x_skip], dim=1)  # Concatenate features
            x = catblock(x)                    # Apply convolutional layers
            # Apply cross attention
            fcon = feats_context.pop()
            x = self.cross_attn_blocks[depth](x, fcon)

        # Output block
        x = self.main_unet.decoder.output_block(x)

        # Return the output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    shape = (64, 64, 64)
    batch_size = 3
    in_channels = 3
    out_channels = 1
    n_cross_channels_list = [1, 1, 2, 8]

    # Create data
    x = torch.randn(batch_size, in_channels, *shape)
    y_list = [torch.randn(batch_size, c, *shape) for c in n_cross_channels_list]

    # Create a model
    model = CrossUnetModel(
        in_channels, out_channels, n_cross_channels_list,
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

