
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False, beta=.5):
        super(ConvBlock, self).__init__()

        # Check inputs
        if upsample and downsample:
            raise ValueError('Cannot upsample and downsample at the same time.')

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.beta = beta

        # Define residual layer
        if upsample:
            self.residual = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif downsample:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        elif in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        # Define convolutional layers
        if upsample:
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )
        elif downsample:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        
        # Residual connection
        x0 = self.residual(x)

        # Convolutional block
        x = self.conv(x)

        # Combine with residual
        x = self.beta * x + (1 - self.beta) * x0

        # Return the output
        return x


# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=4, expansion=2):
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
            nn.ReLU(),
            nn.Linear(n_features_inner, n_features),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):

        # Apply self-attention
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        return x


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

    def forward(self, Q, K, V):
        """
        The forward function uses lots of fast operations. This causes lots of intermediate
        tensors to be stored in memory. To avoid this, we use the checkpoint function to
        avoid storing intermediate tensors in memory. Instead of storing the intermediate
        tensors, the checkpoint function will recompute the intermediate tensors during the
        backward pass. We trade memory for compute time.
        """
        return checkpoint(self._forward, Q, K, V, use_reentrant=False)
    
    def _forward(self, Q, K, V):

        # Get constants
        B, C, D, H, W = Q.shape
        device = Q.device
        
        # Pad keys and values
        K = F.pad(K, [self.kernel_size // 2] * 6)
        V = F.pad(V, [self.kernel_size // 2] * 6)

        # Initialize attention weights
        attn_weights = torch.zeros(B, self.kernel_size**3, D, H, W, device=device)

        # Get attention from each kernel position
        ijk = -1
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    ijk += 1

                    # Get shifted key and positional embedding
                    pos_emb = self.pos_emb[:, i, j, k].view(1, self.n_features, 1, 1, 1)
                    K_shift = K[:, :, i:i+D, j:j+H, k:k+W] + pos_emb

                    # Calculate attention weights
                    attn_weights[:, ijk] = (Q * K_shift).sum(dim=1) / (self.n_features ** 0.5)

        # Softmax attention weights
        attn_weights = F.softmax(attn_weights, dim=1)

        # Initialize output
        x = torch.zeros_like(Q)

        # Get output from each kernel position
        ijk = -1
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    ijk += 1

                    # Get shifted values
                    V_shift = V[:, :, i:i+D, j:j+H, k:k+W]

                    # Apply attention weights
                    dx = attn_weights[:, ijk] * V_shift

                    # Update output
                    x = x + dx

        # Finalize output
        x = self.out_proj(x)

        # Return output
        return x
    

# Make confolutional multi-head attention block
class MultiheadConvAttn3d(nn.Module):
    """Applies multi-head self-attention within a local receptive field defined by a kernel"""
    def __init__(self, n_features, kernel_size=3, n_heads=4):
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
            nn.ReLU(),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Set up normalization layers
        self.norm1 = nn.InstanceNorm3d(n_features)
        self.norm2 = nn.InstanceNorm3d(n_features)

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
    def __init__(self, n_features, kernel_size=3, n_heads=1, expansion=2):
        super(ConvformerCrossBlock3d, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion

        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Multi-head cross-attention
        self.cross_attn = MultiheadConvAttn3d(n_features, kernel_size=kernel_size, n_heads=n_heads)

        # Feedforward layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features_inner, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(n_features_inner, n_features, kernel_size=1),
        )

        # Normalization layers
        self.norm1 = nn.InstanceNorm3d(n_features)
        self.norm2 = nn.InstanceNorm3d(n_features)
        self.norm3 = nn.InstanceNorm3d(n_features)

    def forward(self, x, y):
        """
        x is the query tensor
        y is the context tensor
        """

        # Apply cross-attention
        x_normed = self.norm1(x)
        y_normed = self.norm2(y)  # Normalize context separately
        attn_output = self.cross_attn(x_normed, y_normed, y_normed)
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm3(x))

        return x


