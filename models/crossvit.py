
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from models.vit import ViT3D
from models.blocks import TransformerBlock, CrossTransformerBlock


# Define cross attention vistion transformer model
class CrossViT3d(nn.Module):
    """Cross attention vistion transformer model."""
    def __init__(self,
        in_channels, out_channels, n_cross_channels_list,
        shape=(128, 128, 128), scale=2, patch_size=(4, 4, 4),
        n_features=64, n_heads=4, n_layers=8, n_layers_context=8, n_mixing_blocks=4,
    ):
        super(CrossViT3d, self).__init__()

        # Check inputs
        if n_layers % 2 != 0:
            raise ValueError('Number of layers must be even!')
        if not isinstance(shape, tuple):
            shape = (shape, shape, shape)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size, patch_size)
        for i in range(3):
            if shape[i] % patch_size[i] != 0:
                raise ValueError('Shape must be divisible by patch size!')
        
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.shape = shape
        self.scale = scale
        self.patch_size = patch_size
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_context = n_layers_context

        # Get constants
        n_context = len(n_cross_channels_list)
        shape_downscaled = (
            shape[0] // scale,
            shape[1] // scale,
            shape[2] // scale,
        )
        shape_patchgrid = (
            shape_downscaled[0] // patch_size[0],
            shape_downscaled[1] // patch_size[1],
            shape_downscaled[2] // patch_size[2],
        )
        n_patches = shape_patchgrid[0] * shape_patchgrid[1] * shape_patchgrid[2]
        self.n_context = n_context
        self.n_patches = n_patches
        self.shape_downscaled = shape_downscaled
        self.shape_patchgrid = shape_patchgrid
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(.1*torch.randn(1, n_patches, n_features))
        self.pos_embedding_context = nn.Parameter(.1*torch.randn(1, n_patches*n_context, n_features))

        # Create main autoencoder
        self.autoencoder = ViT3D(
            in_channels, out_channels,
            shape=shape, scale=scale, patch_size=patch_size,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers,
        )

        # Create context autoencoders
        self.context_autoencoders = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_autoencoders.append(
                ViT3D(
                    n_channels, n_channels,
                    shape=shape, scale=scale, patch_size=patch_size,
                    n_features=n_features, n_heads=n_heads, n_layers=n_layers_context,
                )
            )

        # Create mixing blocks
        self.self_mixing_blocks = nn.ModuleList()   # Self attention
        self.cross_mixing_blocks = nn.ModuleList()  # Cross attention
        for i in range(n_mixing_blocks):
            self.self_mixing_blocks.append(
                TransformerBlock(
                    n_features, n_heads=n_heads,
                )
            )
            self.cross_mixing_blocks.append(
                CrossTransformerBlock(
                    n_features, n_heads=n_heads,
                )
            )

    def forward(self, x, y_list):
        """
        x is the input tensor
        y_list is a list of context tensors.
        """

        # Encode input
        x = self.autoencoder.encode(x)
        y_list = [ae.encode(y) for ae, y in zip(self.context_autoencoders, y_list)]
        
        # Cat context tensors
        context = torch.cat(y_list, dim=1)

        # Add positional encoding
        x = x + self.pos_embedding.expand(x.shape[0], -1, -1)
        context = context + self.pos_embedding_context.expand(context.shape[0], -1, -1)

        # Mix self and cross attention
        for transformer, cross_transformer in zip(self.self_mixing_blocks, self.cross_mixing_blocks):
            x = cross_transformer(x, context)
            x = transformer(x)
        
        # Decode
        x = self.autoencoder.decode(x)
        
        # Return
        return x


# Test the model
if __name__ == '__main__':

    # Import custom libraries
    import psutil
    from utils import estimate_memory_usage

    # Set constants
    shape = (64, 64, 64)
    n_channels = 3
    n_channels_context = [1, 1, 3, 30, 1]

    # Create a model
    model = CrossViT3d(
        n_channels, 1, n_channels_context,
        shape=shape
    )

    # Create data
    x = torch.randn(1, n_channels, *shape)
    context_list = [torch.randn(1, n, *shape) for n in n_channels_context]

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
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Memory usage: {(mem_after - mem_before) / 1024**3:.2f} GB")

    # Done
    print('Done!')

