
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
from architectures.crossunet import CrossUnetModel
from architectures.blocks import conv_block_selector, FiLM3d, DyTanh3d


# Make time aware Unet
class TimeAwareUnet3d(CrossUnetModel):
    def __init__(self, 
        in_channels, out_channels, n_cross_channels_list, 
        n_features=16, n_blocks=5, n_layers_per_block=4,
        feature_scale=None,
    ):
        super().__init__(
            in_channels, out_channels, n_cross_channels_list,
            n_features=n_features, 
            n_blocks=n_blocks, 
            n_layers_per_block=n_layers_per_block, 
            feature_scale=feature_scale,
            scale=1,
            use_dropout=False,                      # No dropout in diffusion model
            conv_block_type='ConvBlockFiLM3d',           # Use FiLM block
        )

        # Input regularization
        self.input_reg = DyTanh3d(in_channels, init_alpha=1.0)

        # Define time embedding layers
        self.context_time_embedding_blocks = nn.ModuleList()
        for f in self.n_features_per_depth:
            self.context_time_embedding_blocks.append(FiLM3d(f))

    def forward(self, t, x, feats_context):
        x = x.requires_grad_()
        feats_context = [f.requires_grad_() for f in feats_context]
        return checkpoint(self._forward, t, x, feats_context, use_reentrant=False)
    
    def _forward(self, t, x, feats_context):

        # Regularize input
        x = self.input_reg(x)  # Regularize input block

        # Encode x
        feats = self.main_unet.encoder((x, t))

        # Apply time embedding to context
        feats_context = [block(f, t) for block, f in zip(self.context_time_embedding_blocks, feats_context)]

        # Apply context
        depth = self.n_blocks
        fcon = feats_context[-1]
        x, _ = feats.pop()
        x = self.cross_attn_blocks[depth](x, fcon)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            # Upsample
            x, _ = self.main_unet.decoder.up_blocks[i]((x, t))
            # Merge with skip
            x_skip, _ = feats[depth]
            x = x + x_skip
            # Apply cross attention
            fcon = feats_context[depth]
            x = self.cross_attn_blocks[depth](x, fcon)

        # Output block
        noise_pred, _ = self.main_unet.decoder.output_block((x, t))
        
        # Return noise prediction
        return noise_pred


# Define Diffusion Model Unet
class DiffUnet3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        n_features=8, n_blocks=5, 
        n_layers_per_block=4 , n_mixing_blocks=4,
        scale=2, n_steps=16, eta=.1,
        reuse_prediction=True,
        conv_block_type=None, feature_scale=None,
    ):
        super(DiffUnet3d, self).__init__()

        # Set default values
        if conv_block_type is None:
            conv_block_type = 'ConvBlock3d'
        
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
        self.reuse_prediction = reuse_prediction
        self.conv_block_type = conv_block_type
        self.feature_scale = feature_scale

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

        # Set up convolutional blocks
        conv_block = conv_block_selector(conv_block_type)

        # Define input blocks
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            conv_block(in_channels, n_features, kernel_size=1),
            # Shrink volume
            conv_block(n_features, n_features, scale=1/scale),  # Dense (not depthwise, groups=1) convolution for scaling
            # Additional convolutional layers
            *(conv_block(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1))
        )
        self.context_input_blocks = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            self.context_input_blocks.append(
                nn.Sequential(
                    # Merge input channels to n_features
                    conv_block(n_channels, n_features, kernel_size=1),
                    # Shrink volume
                    conv_block(n_features, n_features, scale=1/scale),  # Dense (not depthwise, groups=1) convolution for scaling
                    # Additional convolutional layers
                    *(conv_block(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1))
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            # Expand volume
            conv_block(n_features, n_features, scale=scale),  # Dense (not depthwise, groups=1) convolution for scaling
            # Convolutional layers
            *[conv_block(n_features, n_features, groups=n_features) for _ in range(n_layers_per_block - 1)],
            # Merge features to output channels
            conv_block(n_features, in_channels, kernel_size=1),
        )

        # Create main unet
        n_in = 2 * n_features if reuse_prediction else n_features
        n_out = n_features
        self.main_unet = TimeAwareUnet3d(
            n_in, n_out, [n_features]*n_context,
            n_features=n_features, 
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            feature_scale=feature_scale,
        )

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
            'reuse_prediction': self.reuse_prediction,
            'conv_block_type': self.conv_block_type,
            'feature_scale': self.feature_scale,
        }
    
    def encode_context(self, *context):
        context = [c.requires_grad_() for c in context]
        return checkpoint(self._encode_context, *context, use_reentrant=False)
        
    def _encode_context(self, *context):
        
        # Input blocks
        latent_context = [block(y) for block, y in zip(self.context_input_blocks, context)]
        
        # Encode features
        feats_context = [block(y) for block, y in zip(self.main_unet.context_encoders, latent_context)]
        feats_context = [sum([f for f in row]) / len(row) for row in zip(*feats_context)]

        # Return
        return latent_context, feats_context
    
    def forward(self, *context, target=None, return_loss=False):

        # Check argmuents
        if return_loss and target is None:
            raise ValueError("If return_loss is True, target must be provided.")

        # If returning loss, embed target and initialize loss
        if return_loss:
            latent_target = self.input_block(target)                 # Embed target
            target_reconstructed = self.output_block(latent_target)  # Reconstruct target
            loss = F.mse_loss(target_reconstructed, target)          # Calculate loss
            loss = loss * 5  # Weight dose reconstruction loss, since it is very important

        # Encode context
        latent_context, feats_context = self.encode_context(*context)

        # Initialize x
        x = torch.randn_like(latent_context[0], device=context[0].device)

        # Initialize x0_guess
        if self.reuse_prediction:
            x0_guess = torch.zeros_like(x, device=x.device)
        else:
            x0_guess = None

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Get constants 
            a_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            a_t1 = self.alpha_cumprod[t-1].view(-1, 1, 1, 1, 1)
            sigma = self.eta * torch.sqrt( (1 - a_t/a_t1) * (1 - a_t) / (1 - a_t1) )

            # Merge with self-conditioning
            if self.reuse_prediction:
                x = torch.cat([x, x0_guess], dim=1)  # Concatenate along channel dimension

            # Predict noise 
            t_step = t * torch.ones(x.shape[0], device=x.device, dtype=torch.long)
            noise_pred = self.main_unet(t_step, x, feats_context)

            # Update self-conditioning with predicted x0_guess
            if self.reuse_prediction:
                x = x[:, :self.n_features]  # Remove self-conditioning
                x0_guess = ((x - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)).detach()  # Detach is important

            # Calculate loss
            if return_loss:
                # w_t = (self.n_steps - t) / self.n_steps
                w_t = 1 / (self.n_steps - 1)  # Evenly weight all steps
                noise = ((x - torch.sqrt(a_t) * latent_target) / torch.sqrt(1 - a_t)).detach()  # Detach is important
                loss += w_t * F.mse_loss(noise_pred, noise)

            # Update position 
            x = (
                torch.sqrt(a_t1/a_t) * (x - torch.sqrt(1 - a_t) * noise_pred)
                + torch.sqrt(1 - a_t1 - sigma**2) * noise_pred 
                + sigma * torch.randn_like(x, device=x.device)
            )

        # Output block
        x = self.output_block(x)

        # Return output
        if return_loss:
            return x, loss
        else:
            return x
    


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


