
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
from architectures.blocks import FiLM3d, DyTanh3d


# Make time aware Unet
class TimeAwareUnet3d(CrossUnetModel):
    def __init__(self, 
        in_channels, out_channels, n_cross_channels_list, scale=2, 
        n_features=16, n_blocks=5, n_layers_per_block=4,
        feature_scale=None, bidirectional=False, use_catblock=False,
    ):
        super().__init__(
            in_channels, out_channels, n_cross_channels_list,
            scale=scale,
            n_features=n_features, 
            n_blocks=n_blocks, 
            n_layers_per_block=n_layers_per_block, 
            conv_block_type='ConvBlockFiLM3d',      # Use FiLM block
            feature_scale=feature_scale,
            bidirectional=bidirectional,
            use_dropout=False,                      # No dropout in diffusion model
            use_catblock=use_catblock,
        )

        # Input regularization
        self.input_reg = DyTanh3d(in_channels, init_alpha=0.1)

        # Output scaling
        self.beta = nn.Parameter(torch.zeros(1))

        # Define time embedding layers
        self.context_time_embedding_blocks = nn.ModuleList()
        for f in self.n_features_per_depth:
            self.context_time_embedding_blocks.append(FiLM3d(f))

    def forward(self, t, x, x_pred, feats_context):
        x = x.requires_grad_()
        x_pred = x_pred.requires_grad_()
        feats_context = [f.requires_grad_() for f in feats_context]
        return checkpoint(self._forward, t, x, x_pred, feats_context, use_reentrant=False)
    
    def _forward(self, t, x, x_pred, feats_context):

        # Process input
        x_pred = x_pred.detach()           # Detach x_pred to avoid gradient flow through time levels
        x = torch.cat([x, x_pred], dim=1)  # Concatenate input and prediction
        x = self.input_reg(x)              # Regularize input

        # Encode x
        feats = self.main_unet.encoder((x, t))

        # Apply time embedding to context
        feats_context = [block(f, t) for block, f in zip(self.context_time_embedding_blocks, feats_context)]

        # Apply context
        depth = self.n_blocks
        fcon = feats_context.pop()
        x, _ = feats.pop()
        x = self.cross_attn_blocks[depth](x, fcon)
        if self.bidirectional:
            fcon = self.context_attn_blocks[depth](fcon, x)

        # Upsample blocks
        for i in range(self.n_blocks):
            depth = self.n_blocks - 1 - i
            upblock = self.main_unet.decoder.up_blocks[i]
            # Upsample
            x, _= upblock((x, t))
            # Merge with skip
            x_skip, _ = feats.pop()
            if self.use_catblock:
                x = torch.cat([x, x_skip], dim=1)                    # Concatenate features
                x, _ = self.main_unet.decoder.cat_blocks[i]((x, t))  # Apply convolutional layers
            else:
                x = x + x_skip
            # Apply cross attention
            if not self.bidirectional:
                fcon = feats_context.pop()
                x = self.cross_attn_blocks[depth](x, fcon)
            else:
                fcon = self.context_decoder.up_blocks[i](fcon)
                fcon_skip = feats_context.pop()
                if self.use_catblock:
                    fcon = torch.cat([fcon, fcon_skip], dim=1)
                    fcon = self.context_decoder.cat_blocks[i](fcon)
                else:
                    fcon = fcon + fcon_skip
                x = self.cross_attn_blocks[depth](x, fcon)
                if depth > 0:
                    fcon = self.context_attn_blocks[depth](fcon, x)

        # Output block
        x, _ = self.main_unet.decoder.output_block((x, t))
        x = x * self.beta
        
        # Return outputs
        return x


# Define Diffusion Model Unet
class DiffUnet3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list, scale=2, 
        n_features=16, n_blocks=5, n_layers_per_block=4, n_mixing_blocks=4,
        n_steps=8, eta=.1,
        bidirectional=False, use_catblock=False,
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
        self.bidirectional = bidirectional
        self.use_catblock = use_catblock

        # Get constants
        n_context = len(n_cross_channels_list)
        self.n_context = n_context

        # Get noise schedule
        a_max = .9
        a_min = .1
        alpha = (a_min/a_max)**(1/n_steps)
        alpha_cumprod = torch.tensor([a_max*alpha**i for i in range(n_steps)], dtype=torch.float32)
        self.alpha = alpha
        self.register_buffer('alpha_cumprod', alpha_cumprod)

        # Set up latent encoding blocks
        n_in = 2 * in_channels
        n_out = in_channels
        self.main_unet = TimeAwareUnet3d(
            n_in, n_out, n_cross_channels_list,
            n_features=n_features, 
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale,
            bidirectional=bidirectional,
            use_catblock=use_catblock,
        )

        # Get number of features per depth
        self.n_features_per_depth = self.main_unet.n_features_per_depth

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
            'bidirectional': self.main_unet.bidirectional,
            'use_catblock': self.main_unet.use_catblock,
        }
    
    def forward(self, *context, target=None, return_loss=False):

        # Check argmuents
        if return_loss and target is None:
            raise ValueError("If return_loss is True, target must be provided.")

        # Encode context
        feats_context = self.main_unet.encode_context(*context)

        # Initialize x and x_pred
        C = self.in_channels
        B, _, D, H, W = context[0].shape
        x = torch.randn((B, C, D, H, W), device=context[0].device)
        x_pred = torch.zeros_like(x, device=x.device)

        # Initialize loss
        if return_loss:
            loss = 0.0

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Get constants 
            t_step = t * torch.ones(B, device=x.device, dtype=torch.long)
            a_t = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
            a_t1 = self.alpha_cumprod[t-1].view(-1, 1, 1, 1, 1)
            sigma = self.eta * torch.sqrt( (1 - a_t/a_t1) * (1 - a_t) / (1 - a_t1) )

            # Predict x and noise
            x_pred = self.main_unet(t_step, x, x_pred, feats_context)
            noise_pred = (x - torch.sqrt(a_t) * x_pred) / torch.sqrt(1 - a_t)

            # Calculate loss
            if return_loss:
                w_t = 1 / (self.n_steps - 1)  # Evenly weight all steps
                noise = ((x - torch.sqrt(a_t) * target) / torch.sqrt(1 - a_t)).detach()  # Detach is important
                loss += w_t * F.mse_loss(noise_pred, noise)

            # Update position 
            x = (
                torch.sqrt(a_t1) * x_pred
                + torch.sqrt(1 - a_t1 - sigma**2) * noise_pred 
                + sigma * torch.randn_like(x, device=x.device)
            )

        # Return output
        if return_loss:
            return x_pred, loss
        else:
            return x_pred
    


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

    # Estimate memory usage
    estimate_memory_usage(model, *y_list, print_stats=True)

    # Done
    print('Done!')


