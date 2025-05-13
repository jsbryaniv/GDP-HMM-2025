
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
        shape=(64, 64, 64), scale=1, ratio_shape_patch=8,
        n_features=64, n_heads=4, n_layers=8, n_mixing_blocks=4,
        n_steps=16, eta=.1,
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
        self.scale = scale
        self.ratio_shape_patch = ratio_shape_patch
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_mixing_blocks = n_mixing_blocks
        self.n_steps = n_steps
        self.eta = eta
        self.use_checkpoint = use_checkpoint

        # Get constants
        n_context = len(n_cross_channels_list)
        self.n_context = n_context

        # Get noise schedule
        a_max = .9
        a_min = .01
        alpha = (a_min/a_max)**(1/n_steps)
        alpha_cumprod = torch.tensor([a_max*alpha**i for i in range(n_steps)], dtype=torch.float32)
        self.alpha = alpha
        self.register_buffer('alpha_cumprod', alpha_cumprod)

        # Define time embedding layers
        self.time_embedding_block = FiLM(n_features)

        # Create main vit
        self.main_vit = ViT3d(
            in_channels, in_channels,
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
        for _ in range(n_mixing_blocks):
            self.mixing_blocks.append(
                CrossTransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                )
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        
        # All convolutional layers to small scale
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.5  # reduce scale

        # Done
        return

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'shape': self.shape,
            'scale': self.scale,
            'ratio_shape_patch': self.ratio_shape_patch,
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_mixing_blocks': self.n_mixing_blocks,
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

        # Apply time embedding
        x = self.time_embedding_block(x, t.float())

        # Apply context
        for block in self.mixing_blocks:
            x = block(x, feats_context)
        
        # Return noise prediction
        return x
    
    def forward(self, *context):
        
        # Encode features
        feats_context = sum(block(y) for block, y in zip(self.context_encoders, context))

        # Initialize x
        x = torch.randn_like(feats_context, device=context[0].device)

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Predict noise
            t_step = t * torch.ones(x.shape[0], device=x.device, dtype=torch.long)
            noise_pred = self.step(t_step, x, feats_context)

            # Get constants 
            a_t = self.alpha_cumprod[t].view(-1, 1, 1)
            a_t1 = self.alpha_cumprod[t-1].view(-1, 1, 1)
            sigma = self.eta * torch.sqrt( (1 - a_t/a_t1) * (1 - a_t) / (1 - a_t1) )

            # Update position
            x = (
                torch.sqrt(a_t1/a_t) * (x - torch.sqrt(1 - a_t) * noise_pred)
                + torch.sqrt(1 - a_t1 - sigma**2) * noise_pred 
                + sigma * torch.randn_like(x, device=x.device)
            )


        # Output block
        x = self.main_vit.decoder(x)

        # Return
        return x
    
    def calculate_diffusion_loss(self, target, *context, n_samples=4):

        # Input blocks
        latent_target = self.main_vit.encoder(target)
        
        # Encode features
        feats_context = sum(block(y) for block, y in zip(self.context_encoders, context))

        # Calculate reconstruction loss
        target_reconstructed = self.main_vit.decoder(latent_target)
        loss = F.mse_loss(target_reconstructed, target)

        # Loop over samples
        for _ in range(n_samples):

            # Sample noise
            noise = torch.randn_like(latent_target, device=target.device)

            # Sample time step and corrupted target
            t = torch.randint(0, self.n_steps, (target.shape[0],), device=target.device)
            alpha_t = self.alpha_cumprod[t].view(-1, 1, 1)
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
    shape = (128, 128, 128)
    batch_size = 3
    in_channels = 1
    n_cross_channels_list = [36]

    # Create data
    x = torch.randn(batch_size, in_channels, *shape)
    y_list = [torch.randn(batch_size, c, *shape) for c in n_cross_channels_list]

    # Create a model
    model = DiffViT3d(
        in_channels, n_cross_channels_list,
        shape=shape,
    )

    # Print model structure
    print(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')
    print('Number of parameters in blocks:')
    for name, block in model.named_children():
        print(f'--{name}: {sum(p.numel() for p in block.parameters()):,}')

    # Forward pass
    with torch.no_grad():
        pred = model(*y_list)
        loss = model.calculate_diffusion_loss(x, *y_list)

    # Estimate memory usage
    estimate_memory_usage(model, *y_list, print_stats=True)

    # Done
    print('Done!')


