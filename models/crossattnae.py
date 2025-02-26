
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
from models.blocks import ConvBlock, VolCrossTransformer3d


# Define cross attention autoencoder model
class CrossAttnAEModel(nn.Module):
    """Cross attention autoencoder model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        n_features=8, n_blocks=4, 
        n_layers_per_block=4, n_layers_per_block_context=2,
        n_heads=4, n_attn_repeats=2,
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
        self.n_heads = n_heads

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
                    n_layers_per_block=n_layers_per_block_context,
                )
            )
        
        # Create context feature dropout layers
        self.context_dropout = nn.ModuleList()
        for depth in range(n_blocks+1):
            p = (1-(1-1/n_features)**(n_blocks-depth))  # No dropouts at final layer; roughly half at first
            self.context_dropout.append(nn.Dropout(p=p))

        
        ### LATENT MIXING BLOCKS ###

        # Create self convformer blocks
        self.self_mixing_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            n_in = n_features_per_depth[depth]
            n_out = n_features_per_depth[depth]
            self.self_mixing_blocks.append(
                nn.Sequential(
                    *[ConvBlock(n_in, n_out, groups=n_heads) for _ in range(n_layers_per_block)]
                )
            )

        # Create cross convformer blocks
        self.cross_mixing_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.cross_mixing_blocks.append(
                VolCrossTransformer3d(
                    n_features_per_depth[depth], n_context,
                    n_heads=n_heads
                )
            )

    def forward(self, x, y_list):
        """
        x is the input tensor
        y is a list of context tensors.
        """

        # Encode y_list, copying transpose
        f_con_blk = [autoencoder.encoder(y) for autoencoder, y in zip(self.context_autoencoders, y_list)]
        f_blk_con = [[f.clone() for f in row] for row in zip(*f_con_blk)]

        # Apply dropout to context features and decode
        f_con_blk = [[dropout(f) for dropout, f in zip(self.context_dropout, f_con_blk[c])] for c in range(self.n_context)]
        # y_list = [autoencoder.decoder(fs) for autoencoder, fs in zip(self.context_autoencoders, f_con_blk)]

        # Encode x
        feats = self.autoencoder.encoder(x)
        x = feats.pop()

        # Apply context
        fcon = f_blk_con[-1]
        depth = self.n_blocks
        for _ in range((depth+1)*self.n_attn_repeats):
            x = self.cross_mixing_blocks[depth](x, fcon)
            x = self.self_mixing_blocks[depth](x)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            # Upsample
            x = self.autoencoder.up_blocks[i](x)
            # Merge with skip
            x_skip = feats[depth]
            x = torch.cat([x, x_skip], dim=1)
            x = self.autoencoder.merge_blocks[i](x)
            # Apply context
            fcon = f_blk_con[depth]
            for _ in range((depth+1)*self.n_attn_repeats):
                x = self.cross_mixing_blocks[depth](x, fcon)
                x = self.self_mixing_blocks[depth](x)

        # Output block
        x = self.autoencoder.output_block(x)

        # Return the output
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    import psutil
    from utils import estimate_memory_usage

    # Create a model
    model = CrossAttnAEModel(3, 1, (1, 1, 2, 8))

    # Create data
    x = torch.randn(1, 3, 128, 128, 128)
    context_list = [torch.randn(1, c, 128, 128, 128) for c in (1, 1, 2, 8)]

    # Measure memory before execution
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss  # Total RAM usage before forward pass

    # Forward pass
    y = model(x, context_list)

    # Backward pass
    loss = y.sum()
    loss.backward()

    # Measure memory after execution
    mem_after = process.memory_info().rss  # Total RAM usage after backward pass
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Memory usage: {(mem_after - mem_before) / 1024**3:.2f} GB")

    # Done
    print('Done!')

