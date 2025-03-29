
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
from architectures.unet import Unet3d, UnetEncoder3d
from architectures.blocks import ConvBlock3d, ConvBlockFiLM3d, ConvformerDecoder3d, FiLM3d, DyTanh3d


# Make time aware Unet
class TimeAwareUnet3d(Unet3d):
    def __init__(self, in_channels, out_channels, n_features=16, n_blocks=5, n_layers_per_block=4):
        super().__init__(
            in_channels, out_channels, 
            n_features=n_features, 
            n_blocks=n_blocks, 
            n_layers_per_block=n_layers_per_block, 
            scale=1,                                # No scaling in autoencoder
            use_dropout=False,                      # No dropout in diffusion model
            conv_block=ConvBlockFiLM3d,             # Use FiLM block for autoencoder
        )
        self.encoder.input_block[0] = ConvBlockFiLM3d(in_channels, n_features)
        self.decoder.output_block[-1] = ConvBlockFiLM3d(n_features, out_channels)

    def forward(self, x, t):  # We only use the encoder and decoder
        pass


# Define Diffusion Model Unet
class DiffUnet3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        n_features=8, n_blocks=5, 
        n_layers_per_block=2, n_mixing_blocks=2,
        scale=2, n_steps=16, eta=.1,
        use_self_conditioning=True,
        use_checkpoint=True,
    ):
        super(DiffUnet3d, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.n_cross_channels_list = n_cross_channels_list
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_mixing_blocks = n_mixing_blocks
        self.scale = scale
        self.n_steps = n_steps
        self.eta = eta
        self.use_self_conditioning = use_self_conditioning
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

        # Define input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1) if in_channels != n_features else nn.Identity(),
            # Shrink volume
            ConvBlock3d(n_features, n_features, scale=1/scale),  # Dense (not depthwise, groups=1) convolution for scaling
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1))
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    nn.Conv3d(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    ConvBlock3d(n_features, n_features, scale=1/scale),  # Dense (not depthwise, groups=1) convolution for scaling
                    # Additional convolutional layers
                    *(ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1))
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Convolutional layers
            *[ConvBlock3d(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1)],
            # Expand volume
            ConvBlock3d(n_features, n_features, scale=scale),  # Dense (not depthwise, groups=1) convolution for scaling
            # Merge features to output channels
            nn.Conv3d(n_features, in_channels, kernel_size=1) if in_channels != n_features else nn.Identity(),
        )

        # Create main autoencoder
        n_in = 2 * n_features if use_self_conditioning else n_features
        n_out = n_features
        self.autoencoder = TimeAwareUnet3d(
            n_in, n_out,
            n_features=n_features, 
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
        )
        
        # Create context encoders
        self.context_encoders = nn.ModuleList()
        for _ in range(len(n_cross_channels_list)):
            self.context_encoders.append(
                UnetEncoder3d(
                    n_features,
                    n_features=n_features, n_blocks=n_blocks,
                    n_layers_per_block=n_layers_per_block,
                    scale=1,
                    use_dropout=False,  # No dropout in diffusion model
                )
            )

        # Get features per depth
        n_features_per_depth = self.autoencoder.n_features_per_depth

        # Create cross attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        for depth in range(n_blocks+1):
            self.cross_attn_blocks.append(
                ConvformerDecoder3d(
                    n_features_per_depth[depth],
                    n_layers=n_mixing_blocks,
                    n_heads=max(1, min(n_features_per_depth[depth] // 8, 4)),
                    dropout=0,  # No dropout in diffusion model
                )
            )

        # Define self conditioning block regularization
        if use_self_conditioning:
            self.self_conditioning_reg = DyTanh3d(n_features=n_features, init_alpha=1.0)

        # Define time embedding layers
        self.time_embedding_blocks = nn.ModuleList()
        for f in n_features_per_depth:
            self.time_embedding_blocks.append(FiLM3d(f))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        
        # All convolutional layers to small scale
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.5  # reduce scale

        # Zero the final conv that predicts noise
        final_layer = self.autoencoder.decoder.output_block[-1].conv
        nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

        # Done
        return

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'n_cross_channels_list': self.n_cross_channels_list,
            'n_features': self.n_features,
            'n_blocks': self.n_blocks,
            'n_layers_per_block': self.n_layers_per_block,
            'n_mixing_blocks': self.n_mixing_blocks,
            'scale': self.scale,
            'n_steps': self.n_steps,
            'eta': self.eta,
            'use_self_conditioning': self.use_self_conditioning,
            'use_checkpoint': self.use_checkpoint,
        }
    
    def encode_context(self, *context):
        if self.use_checkpoint:
            context = [c.requires_grad_() for c in context]
            return checkpoint(self._encode_context, *context, use_reentrant=False)
        else:
            return self._encode_context(*context)
        
    def _encode_context(self, *context):
        
        # Input blocks
        latent_context = [block(y) for block, y in zip(self.context_input_blocks, context)]
        
        # Encode features
        feats_context = [block(y) for block, y in zip(self.context_encoders, latent_context)]
        feats_context = [sum([f for f in row]) / len(row) for row in zip(*feats_context)]

        # Return
        return latent_context, feats_context
    
    def step(self, t, x, feats_context, x0=None):
        if self.use_checkpoint:
            x = x.requires_grad_()
            feats_context = [f.requires_grad_() for f in feats_context]
            return checkpoint(self._step, t, x, feats_context, x0=x0, use_reentrant=False)
        else:
            # Regular step
            return self._step(t, x, feats_context, x0=x0)
    
    def _step(self, t, x, feats_context, x0=None):

        # Merge x and x0
        if self.use_self_conditioning:
            x0 = self.self_conditioning_reg(x0)  # Regularize self conditioning block
            x = torch.cat([x, x0], dim=1)        # Concatenate along channel dimension

        # Encode x
        feats = self.autoencoder.encoder((x, t))

        # Apply time embedding to context
        feats_context = [block(f, t) for block, f in zip(self.time_embedding_blocks, feats_context)]

        # Apply context
        depth = self.n_blocks
        fcon = feats_context[-1]
        x, _ = feats.pop()
        x = self.cross_attn_blocks[depth](x, fcon)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            # Upsample
            x, _ = self.autoencoder.decoder.up_blocks[i]((x, t))
            # Merge with skip
            x_skip, _ = feats[depth]
            x = x + x_skip
            # Apply cross attention
            fcon = feats_context[depth]
            x = self.cross_attn_blocks[depth](x, fcon)

        # Output block
        noise_pred, _ = self.autoencoder.decoder.output_block((x, t))
        
        # Return noise prediction
        return noise_pred
    
    def forward(self, *context):

        # Encode context
        latent_context, feats_context = self.encode_context(*context)

        # Initialize x
        x = torch.randn_like(latent_context[0], device=context[0].device)

        # Initialize x0
        if self.use_self_conditioning:
            x0 = torch.zeros_like(x, device=x.device)
        else:
            x0 = None

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Randomly drop self-conditioning
            if self.use_self_conditioning:
                if self.training and random.random() < 0.5:
                    x0 = torch.zeros_like(x0, device=x.device)

            # Predict noise 
            t_step = t * torch.ones(x.shape[0], device=x.device, dtype=torch.long)
            noise_pred = self.step(t_step, x, feats_context, x0=x0)

            # Get constants 
            a_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            a_t1 = self.alpha_cumprod[t-1].view(-1, 1, 1, 1, 1)
            sigma = self.eta * torch.sqrt( (1 - a_t/a_t1) * (1 - a_t) / (1 - a_t1) )

            # Update self-conditioning with predicted x0
            if self.use_self_conditioning:
                x0 = ((x - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)).detach()  # Detach is important

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
    
    def calculate_diffusion_loss(self, target, *context, n_samples=2):

        # Input blocks
        latent_target = self.input_block(target)
        latent_context = [block(y) for block, y in zip(self.context_input_blocks, context)]
        
        # Encode features
        feats_context = [block(y) for block, y in zip(self.context_encoders, latent_context)]
        feats_context = [sum([f for f in row]) / len(row) for row in zip(*feats_context)]

        # Calculate reconstruction loss
        target_reconstructed = self.output_block(latent_target)
        loss = F.mse_loss(target_reconstructed, target)

        # Loop over samples
        for _ in range(n_samples):

            # Sample noise
            noise = torch.randn_like(latent_target, device=target.device)

            # Sample time step and corrupted target
            t = torch.randint(0, self.n_steps, (target.shape[0],), device=target.device)
            a_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            x = torch.sqrt(a_t) * latent_target + torch.sqrt(1 - a_t) * noise

            # Initialize x0
            x0 = torch.zeros_like(x, device=x.device) if self.use_self_conditioning else None

            # Predict noise without self-conditioning
            noise_pred = self.step(t, x, feats_context, x0=x0)
            loss += F.mse_loss(noise_pred, noise) / n_samples

            # Predict noise with self-conditioning
            if self.use_self_conditioning:
                x0 = ((x - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)).detach()
                noise_pred = self.step(t, x, feats_context, x0=x0)
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
    model = DiffUnet3d(
        in_channels, n_cross_channels_list,
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


