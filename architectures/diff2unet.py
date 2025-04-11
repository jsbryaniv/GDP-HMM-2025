
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
from architectures.unet import Unet3d
from architectures.crossunet import CrossUnetModel
from architectures.blocks import ConvBlockFiLM3d, DyTanh3d, ConvformerDecoder3d


# Define Diffusion Model Unet
class Diff2Unet3d(nn.Module):
    def __init__(self, 
        in_channels, n_cross_channels_list,
        n_features=16, n_blocks=5, 
        n_layers_per_block=4, n_mixing_blocks=4,
        scale=2, n_steps=16, eta=.1,
    ):
        super(Diff2Unet3d, self).__init__()
        
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

        # Set up main Unet
        self.main_unet = Unet3d(
            in_channels, in_channels,
            n_features=n_features, 
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            scale=scale,
        )
        
        # Set up context unets
        self.context_unets = nn.ModuleList()
        for n_channels in n_cross_channels_list:
            context_unet = Unet3d(
                n_channels, n_channels,
                n_features=n_features, 
                n_blocks=n_blocks,
                n_layers_per_block=n_layers_per_block,
                scale=scale,
            )
            self.context_unets.append(context_unet)

        # Get number of features per depth
        n_features_per_depth = self.main_unet.n_features_per_depth
        self.n_features_per_depth = n_features_per_depth

        # Create input regularization
        self.input_reg = DyTanh3d(init_alpha=0.1)

        # Create time aware denoiser
        self.time_aware_denoiser = nn.ModuleList()
        for n_feats in n_features_per_depth:
            self.time_aware_denoiser.append(
                ConvBlockFiLM3d(2*n_feats, n_feats, kernel_size=1),
                ConvBlockFiLM3d(n_feats, n_feats, kernel_size=5, groups=n_feats),
                ConvBlockFiLM3d(n_feats, n_feats, kernel_size=1),
            )

        # Create fusion blocks
        self.context_fusion_blocks = nn.ModuleList()
        for i in range(n_context):
            fusion_block = nn.ModuleList()
            for f in n_features_per_depth:
                fusion_block.append(
                    ConvformerDecoder3d(
                        f, dropout=0, n_heads=max(1, min(f // 8, 4))
                    )
                )
            self.context_fusion_blocks.append(fusion_block)

    def rev_ae(self, x, index=None):
        # Get the encoder and decoder
        if index is None:
            encoder = self.main_unet.encoder
            decoder = self.main_unet.decoder
        else:
            encoder = self.context_unets[index].encoder
            decoder = self.context_unets[index].decoder
        # Apply the encoder and decoder in a checkpoint
        x = x.require_grad_(True)
        x = checkpoint(self._rev_ae, x, encoder, decoder, use_reentrant=False)
        # Return the output
        return x
    
    def _rev_ae(self, x, encoder, decoder):
        x = decoder(x)
        x = encoder(x)
        return x

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
        }
    
    def forward(self, *context, target=None, return_loss=False):

        # Check argmuents
        B = context[0].shape[0]
        device = context[0].device
        if return_loss and target is None:
            raise ValueError("If return_loss is True, target must be provided.")

        # Encode context
        feats_context = [ae.encoder(y) for ae, y in zip(self.context_unets, context)]

        # Initialize x and x_pred
        x = [torch.randn_like(f) for f in context[0]]
        x_pred = [torch.zeros_like(f) for f in x]

        # Initialize loss
        if return_loss:
            loss = 0.0

        # Diffusion steps
        for t in reversed(range(1, self.n_steps)):

            # Get constants 
            t_step = t * torch.ones(B, device=device, dtype=torch.long)
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
                torch.sqrt(a_t1/a_t) * (x - torch.sqrt(1 - a_t) * noise_pred)
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
    model = Diff2Unet3d(
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


