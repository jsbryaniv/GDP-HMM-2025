
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn

# Import custom libraries
from architectures.vit import ViT3D
from architectures.blocks import TransformerBlock, CrossTransformerBlock


# Define cross attention vistion transformer model
class CrossViT3d(nn.Module):
    """Cross attention vistion transformer model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        shape=128, n_features=128, n_heads=4, 
        n_layers=8, n_layers_context=8, n_layers_mixing=8,
    ):
        super(CrossViT3d, self).__init__()
            
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.shape = shape
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_context = n_layers_context
        self.n_layers_mixing = n_layers_mixing

        # Create main autoencoder
        self.autoencoder = ViT3D(
            in_channels, out_channels,
            shape=shape, n_features=n_features, 
            n_heads=n_heads, n_layers=n_layers,
        )

        # Create context autoencoders
        self.context_autoencoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_autoencoders.append(
                ViT3D(
                    n_channels, n_channels,
                    shape=shape, n_features=n_features, 
                    n_heads=n_heads, n_layers=n_layers_context,
                )
            )

        # Create mixing block
        self.mixing_block = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_features, 
                nhead=n_heads,
                dim_feedforward=n_features,
                batch_first=True,
            ),
            num_layers=n_layers_mixing,
        )

        # Create positional and context embeddings
        n_patches = self.autoencoder.n_patches
        n_context = len(n_cross_channels_list)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, n_features))
        self.context_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_patches, n_features)) for _ in range(n_context)
        ])
    
    def get_config(self):
        """Get configuration."""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'shape': self.shape,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_layers_context': self.n_layers_context,
            'n_layers_mixing': self.n_layers_mixing,
        }

    def forward(self, x, y_list):
        """
        x is the input tensor
        y_list is a list of context tensors.
        """

        # Encode input
        x = self.autoencoder.encoder(x)
        context = [ae.encoder(y) for ae, y in zip(self.context_autoencoders, y_list)]

        # Add positional and context embeddings
        x = x + self.pos_embedding 
        context = [y + self.pos_embedding for y in context] 
        context = [y + self.context_embeddings[i] for i, y in enumerate(context)]
        
        # Cat context tensors
        context = torch.cat(context, dim=1)

        # Mixing block
        x = self.mixing_block(x, context)

        # Decode
        x = self.autoencoder.decoder(x)
        
        # Return
        return x
    
    def autoencode_context(self, y_list):
        """Autoencode context."""

        # Encode y_list
        y_list = [ae.encoder(y.float()) for ae, y in zip(self.context_autoencoders, y_list)]

        # Decode y_list
        y_list = [ae.decoder(fs) for ae, fs in zip(self.context_autoencoders, y_list)]

        # Return the output
        return y_list


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    import psutil
    from utils import estimate_memory_usage

    # Set constants
    shape = (128, 128, 128)
    in_channels = 4
    out_channels = 1
    n_channels_context = [1, 4, 30]

    # Create data
    x = torch.randn(1, in_channels, *shape)
    context_list = [torch.randn(1, n, *shape) for n in n_channels_context]

    # Create a model
    model = CrossViT3d(
        in_channels, 
        out_channels, 
        n_channels_context,
        shape=shape
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
    pred = model(x, context_list)
    ae_list = [ae(y) for ae, y in zip(model.context_autoencoders, context_list)]

    # Backward pass
    loss = pred.sum() + sum([ae.sum() for ae in ae_list])
    loss.backward()

    # Measure memory after execution
    mem_after = process.memory_info().rss  # Total RAM usage after backward pass
    print(f"Memory usage: {(mem_after - mem_before) / 1024**3:.2f} GB")

    # Done
    print('Done!')

