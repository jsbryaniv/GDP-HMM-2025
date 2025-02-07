
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Define full convolutional transformer model
class ConvformerModel(nn.Module):
    """Full Convolutional Transformer model"""
    def __init__(self,
        in_channels, out_channels,
        n_features=8, num_layers=4, n_heads=2, kernel_size=3
    ):
        super(ConvformerModel, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        # Define input block
        self.input_block = nn.Sequential(
            # Normalize
            nn.GroupNorm(in_channels, in_channels),
            # Merge input channels to n_features
            nn.Conv3d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    ConvformerBlock3d(n_features, kernel_size=kernel_size, n_heads=n_heads)
                )
            )

        # Define output block
        self.output_block = nn.Sequential(
            nn.Conv3d(n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):
        
        # Input block
        x = self.input_block(x)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x


# Test the model
if __name__ == '__main__':

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a model
    model = ConvformerModel(30, 1)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} parameters')

    # Create data
    x = torch.randn(1, 30, 64, 64, 64)

    # Forward pass
    x = x.to(device)
    model = model.to(device)
    y = model(x)

    # Done
    print('Done!')

