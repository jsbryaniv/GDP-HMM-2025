
# Import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


### ACTIVATION LAYERS ###

# Dynamic Tanh
class DyTanh(nn.Module):
    """Dynamic tanh activation. DyT(x) = gamma * tanh(alpha*x) + beta."""
    def __init__(self, n_features, init_alpha=1.0):
        super(DyTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(n_features)*init_alpha)
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.gamma = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta
    
# Dynamic Tanh 3d
class DyTanh3d(nn.Module):
    """Dynamic tanh activation. DyT(x) = gamma * tanh(alpha*x) + beta."""
    def __init__(self, n_features, init_alpha=1.0):
        super(DyTanh3d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(n_features, 1, 1, 1)*init_alpha)
        self.beta = nn.Parameter(torch.zeros(n_features, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(n_features, 1, 1, 1))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


### MODULATION LAYERS ###

# FiLM Layer, for time embeddings in diffusion models, linear version
class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, n_features, expansion=4):
        super(FiLM, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.expansion = expansion

        # Define scale layer
        self.scale_shift = nn.Sequential(
            nn.Linear(1, expansion*n_features),
            nn.ReLU(inplace=True),
            nn.Linear(expansion*n_features, 2*n_features),
        )

        # Initialize final weights to zero
        nn.init.zeros_(self.scale_shift[-1].weight)
        nn.init.zeros_(self.scale_shift[-1].bias)

    def forward(self, x, t):

        # Get modulation parameters
        scale, shift = self.scale_shift(t.view(-1, 1, 1).float()).chunk(2, dim=-1)

        # Apply modulation
        x = x * (1 + torch.tanh(scale)) + shift

        # Return output
        return x

# FiLM Layer, for time embeddings in diffusion models, volume version
class FiLM3d(nn.Module):
    """Feature-wise Linear Modulation layer for 3D volumes."""
    def __init__(self, n_features, expansion=2):
        super(FiLM3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.expansion = expansion

        # Define scale layer
        self.scale_shift = nn.Sequential(
            nn.Conv3d(1, expansion*n_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(expansion*n_features, 2*n_features, kernel_size=1),
        )

        # Initialize final weights to zero
        nn.init.zeros_(self.scale_shift[-1].weight)
        nn.init.zeros_(self.scale_shift[-1].bias)

    def forward(self, x, t):

        # Get modulation parameters
        scale, shift = self.scale_shift(t.view(-1, 1, 1, 1, 1).float()).chunk(2, dim=1)

        # Apply modulation
        x = x * (1 + torch.tanh(scale)) + shift

        # Return output
        return x


### NORMALIZATION LAYERS ###

# Volume normalization
class VoxelNorm3d(nn.Module):
    """
    Normalize each voxel independently across channels.
    """
    def __init__(self, num_channels, eps=1e-5):
        super(VoxelNorm3d, self).__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):

        # Reshape to (B, D, H, W, C) so LayerNorm normalizes over C
        x = x.permute(0, 2, 3, 4, 1)

        # Apply normalization
        x = self.norm(x)

        # Restore original shape (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        # Return output
        return x

### VOLUME SHAPING BLOCKS ###

# Define sparse Voxel Contract class
class VolumeContract3d(nn.Module):
    """Patch Contraction module."""
    def __init__(self, n_features, scale):
        super(VolumeContract3d, self).__init__()

        # Asset scale is power of 2
        assert scale & (scale - 1) == 0, "Scale must be power of 2."
        
        # Set attributes
        self.n_features = n_features
        self.scale = scale

        # Calculate constants
        n = int(scale.bit_length() - 1)
        kernel_size = 4
        stride = 2
        padding = 1

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        n_features, n_features,
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding,
                        groups=n_features,
                    ),
                    nn.Conv3d(n_features, n_features, kernel_size=1),
                )
            )
        
    def forward(self, x):
        """
        Forward pass.
        """
        
        # Loop over layers
        for layer in self.layers:
            x = layer(x)
        
        # Return tensor
        return x
    
# Define sparse Voxel Expand class
class VolumeExpand3d(nn.Module):
    """Patch Expansion module."""
    def __init__(self, n_features, scale):
        super(VolumeExpand3d, self).__init__()

        # Asset scale is power of 2
        assert scale & (scale - 1) == 0, "Scale must be power of 2."
        
        # Set attributes
        self.n_features = n_features
        self.scale = scale

        # Calculate constants
        n = int(scale.bit_length() - 1)
        kernel_size = 4
        stride = 2
        padding = 1
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(n_features, n_features, kernel_size=1),
                    nn.ConvTranspose3d(
                        n_features, n_features,
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding,
                        groups=n_features,
                    ),
                )
            )
        
    def forward(self, x):
        """
        Forward pass.
        """
        
        # Loop over layers
        for layer in self.layers:
            x = layer(x)
        
        # Return tensor
        return x
    

### CONVOLUTIONAL BLOCKS ###

# ConvBlock3d
class ConvBlock3d(nn.Module):
    def __init__(self, 
        in_channels, out_channels, 
        kernel_size=5, groups=1, scale=1, dropout=0.0,
    ):
        super(ConvBlock3d, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.scale = scale
        self.dropout = dropout

        # Reshaping
        if scale > 1:
            convargs = {
                'in_channels': in_channels, 
                'out_channels': out_channels, 
                'kernel_size': 2*scale,
                'padding': scale//2,
                'stride': scale,
                'groups': groups,
            }
            self.conv = nn.ConvTranspose3d(**convargs)
            self.residual = nn.ConvTranspose3d(**convargs)
        elif scale < 1:
            convargs = {
                'in_channels': in_channels, 
                'out_channels': out_channels, 
                'kernel_size': 2*round(1/scale),
                'padding': round(1/scale)//2,
                'stride': round(1/scale),
                'groups': groups,
            }
            self.conv = nn.Conv3d(**convargs)
            self.residual = nn.Conv3d(**convargs)
        else:
            convargs = {
                'in_channels': in_channels, 
                'out_channels': out_channels, 
                'kernel_size': kernel_size,
                'padding': kernel_size//2,
                'groups': groups,
            }
            self.conv = nn.Conv3d(**convargs)
            self.residual = nn.Identity() if in_channels == out_channels else nn.Conv3d(**convargs)

        # Voxel normalization
        self.norm = VoxelNorm3d(out_channels)

        # Define activation
        self.activation = nn.ReLU(inplace=True)

        # Define mixing layer to allow negative values
        self.mixing = nn.Conv3d(out_channels, out_channels, kernel_size=1)

        # Define dropout
        self.drop = nn.Dropout3d(dropout)

        # Define gamma
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # Residual connection
        x0 = self.residual(x)

        # Convolutional block
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.mixing(x)
        x = self.drop(x)

        # Combine with residual
        x = x0 + x * self.gamma

        # Return output
        return x

# ConvBlock3d with time FiLM
class ConvBlockFiLM3d(ConvBlock3d):
    def __init__(self, 
        in_channels, out_channels, 
        kernel_size=5, groups=1, scale=1, dropout=0.0
    ):
        super(ConvBlockFiLM3d, self).__init__(
            in_channels, out_channels, 
            kernel_size=kernel_size, groups=groups, scale=scale, dropout=dropout,
        )

        # Set up FiLM layer
        self.film = FiLM3d(out_channels)

        # Set up output scale
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):

        # Extract inputs
        x, t = inputs

        # Residual connection
        x0 = self.residual(x)

        # Convolutional block
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.mixing(x)
        x = self.drop(x)

        # Apply FiLM
        x = self.film(x, t)

        # Combine with residual
        x = x0 + x * self.gamma

        # Return output and time
        return (x, t)


### TRANSFORMER BLOCKS ###

# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=4, expansion=1, dropout=0.2):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Linear(n_features_inner, n_features),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):

        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        # Return output
        return x

# Define cross attention transformer block
class CrossTransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=4, expansion=1, dropout=0.2):
        super(CrossTransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Linear(n_features_inner, n_features),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
        self.norm3 = nn.LayerNorm(n_features)
        self.norm4 = nn.LayerNorm(n_features)

    def forward(self, x, y):
        
        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output

        # Apply cross-attention
        x_normed = self.norm2(x)
        y_normed = self.norm3(y)  # Normalize context separately
        attn_output, _ = self.cross_attn(x_normed, y_normed, y_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm4(x))

        return x


### CONVOLUTIONAL TRANSFORMER BLOCKS ###

# Define convolutional attention block
class ConvAttn3d(nn.Module):
    """Applies attention within a local receptive field defined by a kernel"""
    def __init__(self, n_features, kernel_size=5):
        super(ConvAttn3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.kernel_size = kernel_size

        # Set projections
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

        # Create relative positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(n_features, kernel_size, kernel_size, kernel_size))

    def forward(self, q, k, v):
        """
        The forward function uses lots of fast operations. This causes lots of intermediate
        tensors to be stored in memory. To avoid this, we use the checkpoint function to
        avoid storing intermediate tensors in memory. Instead of storing the intermediate
        tensors, the checkpoint function will recompute the intermediate tensors during the
        backward pass. We trade memory for compute time.
        """
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()
        return checkpoint(self._forward, q, k, v, use_reentrant=False)
    
    def _forward(self, q, k, v):

        # Get constants
        B, C, D, H, W = q.shape
        device = q.device
        
        # Pad keys and values
        k = F.pad(k, [self.kernel_size // 2] * 6)
        v = F.pad(v, [self.kernel_size // 2] * 6)

        # Initialize attention weights
        attn_weights = torch.zeros(B, self.kernel_size**3, C, D, H, W, device=device)

        # Get attention from each kernel position
        index = -1
        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                for z in range(self.kernel_size):
                    index += 1

            # Get shifted key and positional embedding
            pos_emb_shift = self.pos_emb[:, x, y, z].view(1, self.n_features, 1, 1, 1)
            k_shift = k[:, :, x:x+D, y:y+H, z:z+W] + pos_emb_shift

            # Calculate attention weights
            attn_weights[:, index] += (q * k_shift).sum(dim=1, keepdim=True) / (self.n_features ** 0.5)

        # Softmax attention weights
        attn_weights = F.softmax(attn_weights, dim=1)

        # Initialize output
        out = torch.zeros_like(q)

        # Get output from each kernel position
        index = -1
        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                for z in range(self.kernel_size):
                    index += 1

            # Get shifted values
            v_shift = v[:, :, x:x+D, y:y+H, z:z+W]

            # Apply attention weights
            dout = attn_weights[:, index] * v_shift

            # Update output
            out += dout

        # Finalize output
        out = self.out_proj(out)

        # Return output
        return out


# Make confolutional multi-head attention block
class MultiheadConvAttn3d(nn.Module):
    """Applies multi-head self-attention within a local receptive field defined by a kernel"""
    def __init__(self, n_features, kernel_size=5, n_heads=4):
        super(MultiheadConvAttn3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.n_heads = n_heads

        # Query, Key, Value projections
        self.q_proj = nn.Conv3d(n_features, n_features, kernel_size=1) 
        self.k_proj = nn.Conv3d(n_features, n_features, kernel_size=1) 
        self.v_proj = nn.Conv3d(n_features, n_features, kernel_size=1) 

        # Define attention heads
        self.attention_heads = nn.ModuleList([
            ConvAttn3d(n_features // n_heads, kernel_size) for _ in range(n_heads)
        ])

        # Define output projection
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

    def forward(self, query, key, value):

        # Project inputs into Q, K, V
        B, C, D, H, W = query.shape
        Qs = self.q_proj(query).chunk(self.n_heads, dim=1)
        Ks = self.k_proj(key).chunk(self.n_heads, dim=1)
        Vs = self.v_proj(value).chunk(self.n_heads, dim=1)

        # Apply attention heads
        out = [head(q, k, v) for (head, q, k, v) in zip(self.attention_heads, Qs, Ks, Vs)]

        # Merge heads
        out = torch.cat(out, dim=1)
        out = self.out_proj(out)

        # Return output
        return out


# Define Convformer block
class ConvformerBlock3d(nn.Module):
    def __init__(self, n_features, kernel_size=5, n_heads=4, expansion=1, dropout=0.2):
        super(ConvformerBlock3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = MultiheadConvAttn3d(n_features, kernel_size=kernel_size, n_heads=n_heads)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features_inner, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Set up normalization layers
        self.norm1 = VoxelNorm3d(n_features)
        self.norm2 = VoxelNorm3d(n_features)

        # Set up gamma
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output * self.gamma

        # Feedforward layer
        x = x + self.mlp(self.norm2(x)) * self.gamma

        # Return output
        return x


# Define convolutional cross-attention transformer block
class ConvformerCrossBlock3d(nn.Module):
    def __init__(self, n_features, kernel_size=5, n_heads=4, expansion=1, dropout=0.2):
        super(ConvformerCrossBlock3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout

        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Multi-head cross-attention
        self.self_attn = MultiheadConvAttn3d(n_features, kernel_size=kernel_size, n_heads=n_heads)
        self.cross_attn = MultiheadConvAttn3d(n_features, kernel_size=kernel_size, n_heads=n_heads)

        # Feedforward layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features_inner, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Normalization layers
        self.norm1 = VoxelNorm3d(n_features)
        self.norm2 = VoxelNorm3d(n_features)
        self.norm3 = VoxelNorm3d(n_features)
        self.norm4 = VoxelNorm3d(n_features)

        # Set up gamma
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        x is the query tensor
        y is the context tensor
        """

        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output * self.gamma

        # Apply cross-attention
        x_normed = self.norm2(x)
        y_normed = self.norm3(y)  # Normalize context separately
        attn_output = self.cross_attn(x_normed, y_normed, y_normed)
        x = x + attn_output * self.gamma

        # Feedforward layer
        x = x + self.mlp(self.norm4(x)) * self.gamma

        return x


# Define convformer encoder
class ConvformerEncoder3d(nn.Module):
    """Stack of ConvformerBlock3d layers acting as an encoder."""
    def __init__(self, n_features, n_layers=1, kernel_size=5, n_heads=4, expansion=1, dropout=0.2):
        super(ConvformerEncoder3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout

        # Create a list of Convformer layers
        self.layers = nn.ModuleList([
            ConvformerBlock3d(
                n_features=n_features,
                kernel_size=kernel_size,
                n_heads=n_heads,
                expansion=expansion,
                dropout=dropout,
            ) 
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # Pass through each Convformer layer
        for layer in self.layers:
            x = layer(x)
        # Return output
        return x


# Define convformer decoder
class ConvformerDecoder3d(nn.Module):
    """Stack of ConvformerCrossBlock3d layers acting as a decoder."""
    def __init__(self, n_features, n_layers=1, kernel_size=5, n_heads=4, expansion=1, dropout=0.2):
        super(ConvformerDecoder3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.expansion = expansion
        self.dropout = dropout

        # Create a list of Convformer cross-attention layers
        self.layers = nn.ModuleList([
            ConvformerCrossBlock3d(
                n_features=n_features,
                kernel_size=kernel_size,
                n_heads=n_heads,
                expansion=expansion,
                dropout=dropout,
            ) 
            for _ in range(n_layers)
        ])

    def forward(self, x, y):
        # Pass through each cross-attention layer
        for layer in self.layers:
            x = layer(x, y)
        # Return output
        return x


