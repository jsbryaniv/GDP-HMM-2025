
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from models.unet import Unet3D
from models.blocks import ConvBlock, ConvformerBlock3d, ConvformerCrossBlock3d


# Define cross attention autoencoder model
class CrossAttnAEModel(nn.Module):
    """Cross attention autoencoder model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        n_features=8, n_blocks=4, 
        n_layers_per_block=3, n_layers_per_block_context=2,
        n_attn_repeats=1, n_attn_heads=2,
    ):
        super(CrossAttnAEModel, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_layers_per_block_context = n_layers_per_block_context
        self.n_attn_repeats = n_attn_repeats
        self.n_attn_heads = n_attn_heads

        # Get constants
        n_context = len(n_cross_channels_list)
        n_features_per_depth = [n_features * (i+1) for i in range(n_blocks+1)]
        self.n_context = n_context
        self.n_features_per_depth = n_features_per_depth


        ### AUTOENCODERS ###

        # Create main autoencoder
        self.autoencoder = Unet3D(
            in_channels, out_channels, 
            n_features=n_features, n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block
        )
        
        # Create context autoencoders
        self.context_autoencoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_autoencoders.append(
                Unet3D(
                    n_channels, n_channels, 
                    n_features=n_features, n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block_context
                )
            )
        
        # Create context feature dropout layers
        self.context_dropout = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.context_dropout.append(nn.Dropout(p=1-.2**(n_blocks-depth)))

        
        ### CONVOLUTIONAL TRANSFORMERS ###

        # Create self convformer blocks
        self.self_convformer_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.self_convformer_blocks.append(
                ConvformerBlock3d(
                    n_features_per_depth[depth], 
                    kernel_size=3, n_heads=n_attn_heads,
                )
            )

        # Create cross convformer blocks
        self.cross_convformer_blocks = nn.ModuleDict()
        for depth in range(n_blocks+1):
            for j in range(n_context):
                name = f'depth{depth}_context{j}'
                self.cross_convformer_blocks[name] = ConvformerCrossBlock3d(
                    n_features_per_depth[depth], 
                    kernel_size=1, n_heads=n_attn_heads,
                )

    def apply_context(self, x, fcon, depth):
        """
        Apply context features to the input tensor at depth i.
        """
        for _ in range(self.n_attn_repeats*(depth+1)):
            # Apply context features
            for j in range(self.n_context):
                x = self.cross_convformer_blocks[f'depth{depth}_context{j}'](x, fcon[j])
            # Apply self convformer
            x = self.self_convformer_blocks[depth](x)
        # Return the output
        return x

    def forward(self, x, y_list):
        """
        x is the input tensor
        y is a list of context tensors.
        """

        # Encode y_list, copying transpose
        f_con_blk = [  # f[context][block]
            self.context_autoencoders[c].encoder(y_list[c]) for c in range(self.n_context)
        ]
        f_blk_con = [  # f[block][context]
            [row[i].clone() for row in f_con_blk] for i in range(self.n_blocks+1)
        ]

        # Apply dropout to context features and decode
        f_con_blk = [  # f[context][block]
            [self.context_dropout[i](f_con_blk[c][i]) for i in range(self.n_blocks+1)] for c in range(self.n_context)
        ]
        y_list = [  # y[context]
            self.context_autoencoders[i].decoder(f_con_blk[i]) for i in range(self.n_context)
        ]

        # Encode x
        feats = self.autoencoder.encoder(x)
        x = feats.pop()

        # Apply context
        fcon = f_blk_con.pop()
        x = self.apply_context(x, fcon, self.n_blocks)

        # Upsample blocks
        for i in range(self.n_blocks):
            # Upsample
            x = self.autoencoder.up_blocks[i](x)
            # Apply context
            fcon = f_blk_con.pop()
            x = self.apply_context(x, fcon, self.n_blocks-i-1)
            # Merge with skip
            x_skip = feats.pop()
            x = torch.cat([x, x_skip], dim=1)

        # Output block
        x = self.autoencoder.output_block(x)

        # Return the output and autoencoded ys
        return x, y_list


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from utils import estimate_memory_usage

    # Create a model
    model = CrossAttnAEModel(3, 1, (1, 1, 2, 8))

    # Create data
    x = torch.randn(1, 3, 128, 128, 128)
    context_list = [torch.randn(1, c, 128, 128, 128) for c in (1, 1, 2, 8)]

    # Forward pass
    y, context_list_ae = model(x, context_list)

    # Backward pass
    loss = y.sum() + sum([c.sum() for c in context_list_ae])
    loss.backward()

    # # Estimate memory usage
    # estimate_memory_usage(model, x, print_stats=True)

    # Done
    print('Done!')

