
# Add the project root to sys.path if running as __main__
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import libraries
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import custom libraries
from architectures.vit import ViT3d, ViTEncoder3d
from architectures.blocks import ConvBlock3d, CrossTransformerBlock, FiLM


# Define Diffusion Model ViT
class DiffViT3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        shape=(64, 64, 64), shape_patch_ratio=8,
        n_features=64, n_heads=4, n_layers=8, n_layers_input=4, n_mixing_blocks=4,
        scale=4, n_steps=16, eta=.1,
        use_checkpoint=True,
    ):
        super(DiffViT3d, self).__init__()

        # Check inputs
        if isinstance(shape, int):
            shape = (shape, shape, shape)
        
        # Set attributes
        self.in_channels = in_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.shape = shape
        self.shape_patch_ratio = shape_patch_ratio
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_input = n_layers_input
        self.n_mixing_blocks = n_mixing_blocks
        self.scale = scale
        self.n_steps = n_steps
        self.eta = eta
        self.use_checkpoint = use_checkpoint

        # Get constants
        n_context = len(n_cross_channels_list)
        shape_latent = tuple(s//scale for s in shape)
        self.n_context = n_context
        self.shape_latent = shape_latent

        # Get noise schedule
        a_max = .9
        a_min = .01
        alpha = (a_min/a_max)**(1/n_steps)
        alpha_cumprod = torch.tensor([a_max*alpha**i for i in range(n_steps)], dtype=torch.float32)
        self.alpha = alpha
        self.register_buffer('alpha_cumprod', alpha_cumprod)

        # Define time embedding layers
        self.time_embedding_block = FiLM(n_features)

        # Define input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Shrink volume
            ConvBlock3d(n_features, n_features, groups=n_features, scale=1/scale),
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1))
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    nn.Conv3d(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    ConvBlock3d(n_features, n_features, groups=n_features, scale=1/scale),
                    # Additional convolutional layers
                    *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1))
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Convolutional layers
            *[ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_input - 1)],
            # Expand volume
            ConvBlock3d(n_features, n_features, groups=n_features, scale=scale),
            # Merge features to output channels
            nn.Conv3d(n_features, in_channels, kernel_size=1),
        )

        # Create main autoencoder
        self.autoencoder = ViT3d(
            n_features, n_features,
            shape=shape_latent, scale=1, shape_patch_ratio=shape_patch_ratio,
            n_features=n_features, n_heads=n_heads, n_layers=n_layers,
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for _ in range(len(n_cross_channels_list)):
            self.context_encoders.append(
                ViTEncoder3d(
                    n_features,
                    shape=shape_latent, scale=1, shape_patch_ratio=shape_patch_ratio,
                    n_features=n_features, n_heads=n_heads, n_layers=n_layers,
                )
            )

        # Create mixing block
        self.mixing_blocks = nn.ModuleList()
        for _ in range(n_mixing_blocks):
            self.mixing_blocks.append(
                CrossTransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                )
            )

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'shape': self.shape,
            'shape_patch_ratio': self.shape_patch_ratio,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_layers_input': self.n_layers_input,
            'n_mixing_blocks': self.n_mixing_blocks,
            'scale': self.scale,
            'n_steps': self.n_steps,
            'eta': self.eta,
            'use_checkpoint': self.use_checkpoint,
        }
    
    def step(self, t, x, feats_context):
        if self.use_checkpoint:
            x = x.requires_grad_(True)
            feats_context = feats_context.requires_grad_(True)
            return checkpoint(self._step, t, x, feats_context, use_reentrant=False)
        else:
            # Regular step
            return self._step(t, x, feats_context)
    
    def _step(self, t, x, feats_context):
        
        # Encode x
        feats = self.autoencoder.encoder(x)

        # Apply time embedding
        feats = self.time_embedding_block(feats, t.float())

        # Apply context
        for block in self.mixing_blocks:
            feats = block(feats, feats_context)

        # Decode features
        noise_pred = self.autoencoder.decoder(feats)
        
        # Return noise prediction
        return noise_pred
    
    def forward(self, *context):

        # Input blocks
        latent_context = [block(y) for block, y in zip(self.context_input_blocks, context)]
        
        # Encode features
        feats_context = sum(block(y) for block, y in zip(self.context_encoders, latent_context))

        # Initialize x
        x = torch.randn_like(latent_context[0], device=context[0].device)

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Predict noise
            t_step = t * torch.ones(x.shape[0], device=x.device, dtype=torch.long)
            noise_pred = self.step(t_step, x, feats_context)

            # Get constants 
            a_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            a_t1 = self.alpha_cumprod[t-1].view(-1, 1, 1, 1, 1)
            sigma = self.eta * torch.sqrt( (1 - a_t/a_t1) * (1 - a_t) / (1 - a_t1) )

            # Update position
            x = (
                torch.sqrt(a_t1/a_t) * (x - torch.sqrt(1 - a_t) * noise_pred)
                + torch.sqrt(1 - a_t1 - sigma**2) * noise_pred 
                + sigma * torch.randn_like(x, device=x.device)
            )

            # Check for nans and infs
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError('NaNs or Infs detected in the diffusion model.')


        # Output block
        x = self.output_block(x)

        # Return
        return x
    
    def calculate_diffusion_loss(self, target, *context, n_samples=4):

        # Input blocks
        latent_target = self.input_block(target)
        latent_context = [block(y) for block, y in zip(self.context_input_blocks, context)]
        
        # Encode features
        feats_context = sum(block(y) for block, y in zip(self.context_encoders, latent_context))

        # Calculate reconstruction loss
        target_reconstructed = self.output_block(latent_target)
        loss = F.mse_loss(target_reconstructed, target)

        # Loop over samples
        for _ in range(n_samples):

            # Sample noise
            noise = torch.randn_like(latent_target, device=target.device)

            # Sample time step and corrupted target
            t = torch.randint(0, self.n_steps, (target.shape[0],), device=target.device)
            alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            latent_target_corrupted = torch.sqrt(alpha_t) * latent_target + torch.sqrt(1 - alpha_t) * noise

            # Step forward
            noise_pred = self.step(t, latent_target_corrupted, feats_context)

            # Calculate loss
            loss += F.mse_loss(noise_pred, noise) / n_samples

        # Return
        return loss
        

# Test the model
if __name__ == '__main__':

    # Import custom libraries
    from config import *  # Import config to restrict memory usage (resource restriction script in config.py)
    from utils import estimate_memory_usage

    # Set constants
    shape = (64, 64, 64)
    batch_size = 3
    in_channels = 1
    n_cross_channels_list = [36]

    # Create data
    x = torch.randn(batch_size, in_channels, *shape)
    y_list = [torch.randn(batch_size, c, *shape) for c in n_cross_channels_list]

    # Create a model
    model = DiffViT3d(
        in_channels, n_cross_channels_list,
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        loss = model.calculate_diffusion_loss(x, *y_list)
        pred = model(*y_list)

    # Estimate memory usage
    estimate_memory_usage(model, *y_list, print_stats=True)

    # Done
    print('Done!')


