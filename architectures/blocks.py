
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


### NORMALIZATION BLOCKS ###

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


### PATCH SHAPING BLOCKS ###

# Define sparse Patch Contract class
class VolumeContractSparse3d(nn.Module):
    """Patch Contraction module."""
    def __init__(self, n_features, scale):
        super(VolumeContractSparse3d, self).__init__()

        # Asset scale is power of 2 and buffer is either 0 or scale/2
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
    
# Define sparse Patch Expand class
class VolumeExpandSparse3d(nn.Module):
    """Patch Expansion module."""
    def __init__(self, n_features, scale):
        super(VolumeExpandSparse3d, self).__init__()

        # Asset scale is power of 2 and buffer is either 0 or scale/2
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

# Convolutional block
class ConvBlock3d(nn.Module):
    def __init__(self, 
        in_channels, out_channels, 
        kernel_size=None, groups=1, beta=.1, 
        upsample=False, downsample=False, scale=2
    ):
        super(ConvBlock3d, self).__init__()

        # Check inputs
        if upsample and downsample:
            raise ValueError('Cannot upsample and downsample at the same time.')
        if upsample or downsample:
            assert scale & (scale - 1) == 0, "Scale must be power of 2."
            # Define constants
            stride = scale
            padding = scale // 2
            kernel_size = scale + 2 * padding
        else:
            # Define constants
            kernel_size = 3
            padding = kernel_size // 2
            stride = 1

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.beta = beta
        self.upsample = upsample
        self.downsample = downsample
        self.scale = scale

        # Define convolutional and residual layers
        if upsample:
            # Residual layer
            self.residual = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
            )
            # Convolutional layer
            self.conv = nn.ConvTranspose3d(  
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
            )
        elif downsample:
            # Residual layer
            self.residual = nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
            )
            # Convolutional layer
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
            )
        else:
            # Residual layer
            if in_channels == out_channels:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, groups=groups
                )
            # Convolutional layer
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups,
                ),
            )

        # Define norm
        self.norm = nn.GroupNorm(groups, out_channels)

        # Define activation
        self.activation = nn.ReLU(inplace=True)

        # Define mixing layer to allow negative values
        self.mixing = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        # Residual connection
        x0 = self.residual(x)

        # Convolutional block
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.mixing(x)

        # Combine with residual
        x = self.beta * x0 + (1 - self.beta) * x

        # Return the output
        return x

# Define Volume Encoder
class ConvVolEncoder3d(nn.Module):
    """SDM Volume Encoder module."""
    def __init__(self, in_channels, n_features=4, n_blocks=3, n_layers_per_block=4):
        super(ConvVolEncoder3d, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block

        # Define input block
        self.input_block = nn.Sequential(
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=1),
            # Additional convolutional layers
            *(ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1))
        )

        # Define downsample blocks
        self.down_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.down_blocks.append(
                nn.Sequential(
                    # Downsample layer
                    ConvBlock3d(n_features, n_features, downsample=True),
                    # Additional convolutional layers
                    *[ConvBlock3d(n_features, n_features) for _ in range(n_layers_per_block - 1)]
                )
            )

    def forward(self, x):
        
        # Initialize list of features
        feats = []

        # Input block
        x = self.input_block(x)
        feats.append(x.clone())

        # Downsample blocks
        for block in self.down_blocks:
            x = block(x)
            feats.append(x.clone())

        # Return features
        return feats


### TRANSFORMER BLOCKS ###

# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=4, expansion=1, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.GELU(),
            nn.Dropout(dropout),
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
    def __init__(self, n_features, n_heads=4, expansion=1, dropout=0.1):
        super(CrossTransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
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
            nn.Dropout(dropout),
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
    def __init__(self, n_features, kernel_size=3):
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
        return checkpoint(self._forward, q, k, v, use_reentrant=False)
        # return self._forward(q, k, v)  # TODO: Test if this results in major memory savings
    
    def _forward(self, q, k, v):

        # Get constants
        B, C, D, H, W = q.shape
        device = q.device
        
        # Pad keys and values
        k = F.pad(k, [self.kernel_size // 2] * 6)
        v = F.pad(v, [self.kernel_size // 2] * 6)

        # Initialize attention weights
        attn_weights = torch.zeros(B, self.kernel_size**3, D, H, W, device=device)

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
            attn_weights[:, index] += (q * k_shift).sum(dim=1) / (self.n_features ** 0.5)

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
    def __init__(self, n_features, kernel_size=3, n_heads=1):
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
    def __init__(self, n_features, kernel_size=3, n_heads=1, expansion=1):
        super(ConvformerBlock3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = MultiheadConvAttn3d(n_features, kernel_size=kernel_size, n_heads=n_heads)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features_inner, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Set up normalization layers
        self.norm1 = VoxelNorm3d(n_features)
        self.norm2 = VoxelNorm3d(n_features)

    def forward(self, x):

        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        # Return output
        return x


# Define convolutional cross-attention transformer block
class ConvformerCrossBlock3d(nn.Module):
    def __init__(self, n_features, kernel_size=3, n_heads=1, expansion=1):
        super(ConvformerCrossBlock3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion

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
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Normalization layers
        self.norm1 = VoxelNorm3d(n_features)
        self.norm2 = VoxelNorm3d(n_features)
        self.norm3 = VoxelNorm3d(n_features)
        self.norm4 = VoxelNorm3d(n_features)

    def forward(self, x, y):
        """
        x is the query tensor
        y is the context tensor
        """

        # Apply self-attention
        x_normed = self.norm1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_output

        # Apply cross-attention
        x_normed = self.norm2(x)
        y_normed = self.norm3(y)  # Normalize context separately
        attn_output = self.cross_attn(x_normed, y_normed, y_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm4(x))

        return x


# Define convformer encoder
class ConvformerEncoder3d(nn.Module):
    """Stack of ConvformerBlock3d layers acting as an encoder."""
    def __init__(self, n_features, n_layers=1, kernel_size=3, n_heads=1, expansion=1):
        super(ConvformerEncoder3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.expansion = expansion

        # Create a list of Convformer layers
        self.layers = nn.ModuleList([
            ConvformerBlock3d(
                n_features=n_features,
                kernel_size=kernel_size,
                n_heads=n_heads,
                expansion=expansion,
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
    def __init__(self, n_features, n_layers=1, kernel_size=3, n_heads=1, expansion=1):
        super(ConvformerDecoder3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.expansion = expansion

        # Create a list of Convformer cross-attention layers
        self.layers = nn.ModuleList([
            ConvformerCrossBlock3d(
                n_features=n_features,
                kernel_size=kernel_size,
                n_heads=n_heads,
                expansion=expansion,
            ) 
            for _ in range(n_layers)
        ])

    def forward(self, x, y):
        # Pass through each cross-attention layer
        for layer in self.layers:
            x = layer(x, y)
        # Return output
        return x


### VOLUMETRIC TRANSFORMER BLOCKS ###

# Define volumentric attention block
class VolAttn3d(nn.Module):
    """Applies attention for each voxel in a 3D volume given context features."""
    def __init__(self, n_features):
        super(VolAttn3d, self).__init__()

        # Set attributes
        self.n_features = n_features

        # Set projections
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

    def forward(self, q, k_list, v_list):

        # Concatenate context
        k = torch.cat([K.unsqueeze(1) for K in k_list], dim=1)
        v = torch.cat([V.unsqueeze(1) for V in v_list], dim=1)

        # Get attention weights
        attn_weights = (q.unsqueeze(1) * k).sum(dim=2) / (self.n_features ** 0.5)
        # attn_weights = torch.einsum('bfxyz,bcfxyz->bcxyz', Q, K) / (self.n_features ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention weights
        x = (attn_weights.unsqueeze(2) * v).sum(dim=1)
        # x = torch.einsum('bcxyz,bcfxyz->bfxyz', attn_weights, V)

        # Finalize output
        x = self.out_proj(x)

        # Return output
        return x


# Make volumetric multi-head attention block
class MultiheadVolAttn3d(nn.Module):
    """Applies multi-head attention for each voxel in a 3D volume given context features."""
    def __init__(self, n_features, kernel_size=3, n_heads=4):
        super(MultiheadVolAttn3d, self).__init__()

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
            VolAttn3d(n_features // n_heads) for _ in range(n_heads)
        ])

        # Define output projection
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

    def forward(self, query, key_list, value_list):

        # Project inputs into Q, K, V
        Q = self.q_proj(query).chunk(self.n_heads, dim=1)
        K_list = [self.k_proj(key).chunk(self.n_heads, dim=1) for key in key_list]        # [context][head]
        V_list = [self.v_proj(value).chunk(self.n_heads, dim=1) for value in value_list]  # [context][head]
        K_list = list(zip(*K_list))                                                       # [head][context]
        V_list = list(zip(*V_list))                                                       # [head][context]

        # Apply attention heads
        out = [head(q, k_list, v_list) for (head, q, k_list, v_list) in zip(self.attention_heads, Q, K_list, V_list)]

        # Merge heads
        out = torch.cat(out, dim=1)
        out = self.out_proj(out)

        # Return output
        return out


# Define volumetric cross-attention transformer block
class VolCrossTransformer3d(nn.Module):
    """Volumetric cross-attention transformer block."""
    def __init__(self, n_features, n_context, n_heads=1, expansion=1):
        super(VolCrossTransformer3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_context = n_context
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = MultiheadVolAttn3d(n_features, n_heads=n_heads)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features_inner, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Set up normalization layers
        self.norm1 = VoxelNorm3d(n_features)
        self.norm2 = VoxelNorm3d(n_features)
        self.norm_context = nn.ModuleList([VoxelNorm3d(n_features) for _ in range(n_context)])

    def forward(self, x, y_list):

        # Apply self-attention
        x_normed = self.norm1(x)
        y_list_normed = [norm(y) for norm, y in zip(self.norm_context, y_list)]
        attn_output = self.self_attn(x_normed, y_list_normed, y_list_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        # Return output
        return x
    


