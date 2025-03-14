
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Import custom libraries
from architectures.unet import Unet3D
from architectures.blocks import ConvformerDecoder3d


# Define cross attention unet model
class CrossAttnUnetModel(nn.Module):
    """Cross attention Unet model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        n_features=8, n_blocks=5, 
        n_layers_per_block=4, n_layers_per_block_context=4,
        n_attn_repeats=2, attn_kernel_size=5,
        use_checkpoint=False,
    ):
        super(CrossAttnUnetModel, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_layers_per_block_context = n_layers_per_block_context
        self.n_attn_repeats = n_attn_repeats
        self.attn_kernel_size = attn_kernel_size
        self.use_checkpoint = use_checkpoint

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

        # Create cross attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.cross_attn_blocks.append(
                ConvformerDecoder3d(
                    n_features_per_depth[depth], 
                    kernel_size=attn_kernel_size,
                    n_heads=depth+1,
                    n_layers=n_attn_repeats,
                )
            )
        
        # Create context feature dropout layers
        self.context_dropout = nn.ModuleList()
        for depth in range(n_blocks+1):
            p = (1-(1-2/n_features)**(n_blocks-depth))  # No dropouts at last layer; slightly over half at first
            self.context_dropout.append(nn.Dropout(p=p))
    
    def get_config(self):
        """Get configuration."""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
            'n_layers_per_block_context': self.n_layers_per_block_context,
            'n_attn_repeats': self.n_attn_repeats,
            'attn_kernel_size': self.attn_kernel_size,
            'use_checkpoint': self.use_checkpoint,
        }

    def forward(self, x, y_list):
        """
        x is the input tensor
        y is a list of context tensors.
        """

        # Encode x
        if self.use_checkpoint:
            device = next(self.parameters()).device
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            feats = checkpoint(lambda *args: self.autoencoder.encoder(*args[1:]), dummy, x)
        else:
            feats = self.autoencoder.encoder(x)
        x = feats.pop()

        # Encode y_list and sum features at each depth
        if self.use_checkpoint:
            device = next(self.parameters()).device
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            f_context = [
                # checkpoint(ae.encoder, y.float()+dummy)
                checkpoint(lambda *args: ae.encoder(*args[1:]), dummy, y.float())
                for ae, y in zip(self.context_autoencoders, y_list)
            ]
        else:
            f_context = [ae.encoder(y.float()) for ae, y in zip(self.context_autoencoders, y_list)]
        f_context = [sum([f for f in row]) for row in zip(*f_context)]

        # Apply context
        depth = self.n_blocks
        fcon = f_context[-1]
        x = self.cross_attn_blocks[depth](x, fcon)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            # Upsample
            x = self.autoencoder.up_blocks[i](x)
            # Merge with skip
            x_skip = feats[depth]
            x = x + x_skip
            # Apply cross attention
            fcon = f_context[depth]
            x = self.cross_attn_blocks[depth](x, fcon)

        # Output block
        x = self.autoencoder.output_block(x)

        # Return the output
        return x
    
    def autoencode_context(self, y_list):
        """Autoencode context."""

        # Encode y_list
        y_list = [ae.encoder(y.float()) for ae, y in zip(self.context_autoencoders, y_list)]

        # Apply dropout to context features
        y_list = [
            [dropout(f) for dropout, f in zip(self.context_dropout, y_list[c])] for c in range(self.n_context)
        ]

        # Decode y_list
        y_list = [ae.decoder(fs) for ae, fs in zip(self.context_autoencoders, y_list)]

        # Return the output
        return y_list


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    import psutil

    # Set constants
    shape = (128, 128, 128)
    in_channels = 3
    out_channels = 1
    n_cross_channels_list = [1, 1, 2, 8]

    # Create data
    x = torch.randn(1, in_channels, *shape)
    context_list = [torch.randn(1, c, *shape) for c in n_cross_channels_list]

    # Create a model
    model = CrossAttnUnetModel(
        in_channels, out_channels, n_cross_channels_list,
    )

    # Print model parameter info
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        y = model(x, context_list)


    #### Estimate memory usage ####

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
    print(f"Memory usage: {(mem_after - mem_before) / 1024**3:.2f} GB")

    # Done
    print('Done!')

