
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define kernelized self-attention block
class AttentionKernelBlock(nn.Module):
    """Applies self-attention within a local receptive field defined by a kernel"""
    def __init__(self, n_features, kernel_size=3):
        super(AttentionKernelBlock, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.kernel_size = kernel_size

        # Query, Key, Value projections
        self.qkv_proj = nn.Conv3d(n_features, n_features * 3, kernel_size=1)  # 1x1 Conv as linear layer
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

        # Create relative positional embeddings
        self.rel_pos = nn.Parameter(torch.randn(n_features, kernel_size, kernel_size, kernel_size))

        # Create window extractor kernel that reshapes a kernel window (B,C,k,k,k) into a vector (B,C*k^3)
        wek = torch.zeros(n_features*kernel_size**3, n_features, kernel_size, kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    wek[i*kernel_size**2 + j*kernel_size + k, :, i, j, k] = 1
        self.window_extractor_kernel = wek
        

    def forward(self, x):
        return self.forward_v1(x)
    
    def forward_v1(self, x):

        # Project inputs into Q, K, V
        B, C, D, H, W = x.shape
        Q, K, V = self.qkv_proj(x).chunk(3, dim=1)  # Split into Q, K, V
        
        # Pad keys and values
        K = F.pad(K, [self.kernel_size // 2] * 6)
        V = F.pad(V, [self.kernel_size // 2] * 6)

        # Initialize attention weights
        attn = torch.zeros(B, self.kernel_size**3, D, H, W, device=x.device)

        # Loop over displacements
        ijk = -1
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    ijk += 1

                    # Get shifted key and positional embedding
                    pos_emb = self.rel_pos[:, i, j, k].view(1, self.n_features, 1, 1, 1)
                    K_shift = K[:, :, i:i+D, j:j+H, k:k+W] + pos_emb

                    # Calculate attention weights
                    attn[:, ijk] = (Q * K_shift).sum(dim=1) / (self.n_features ** 0.5)

        # Softmax attention weights
        attn_weights = F.softmax(attn, dim=1)

        # Loop over displacements
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
        x = self.out_proj(x)  # Final projection

        # Return output
        return x
    
    def forward_v2(self, x):

        # Project inputs into Q, K, V
        B, C, D, H, W = x.shape
        Q, K, V = self.qkv_proj(x).chunk(3, dim=1)  # Split into Q, K, V
        
        # Pad keys and values
        K = F.pad(K, [self.kernel_size // 2] * 6)
        V = F.pad(V, [self.kernel_size // 2] * 6)
        
        # Extract windows into a single channel
        wek = self.window_extractor_kernel.to(x.device)  # Window extractor kernel (wek)
        K_kernels = F.conv3d(K, wek)
        V_kernels = F.conv3d(V, wek)

        # Add relative positional embeddings
        pos_emb = self.rel_pos.view(1, self.n_features*self.kernel_size**3, 1, 1, 1)
        K_kernels = K_kernels + pos_emb

        # Initialize attention weights
        attn = torch.zeros(B, self.kernel_size**3, D, H, W, device=x.device)
        
        # Get attention weights
        for ijk in range(self.kernel_size**3):
            attn[:, ijk] = (Q * K_kernels[:, ijk:ijk+self.n_features]).sum(dim=1) / (self.n_features ** 0.5)

        # Softmax attention weights
        attn = F.softmax(attn, dim=1)

        # Apply attention weights to values
        out = torch.zeros(B, self.n_features, D, H, W, device=x.device)
        for ijk in range(self.kernel_size**3):
            out = out + attn[:, ijk] * V_kernels[:, ijk:ijk+self.n_features]

        # Finalize output
        x = x = self.out_proj(out)  # Final projection

        # Return output
        return x
    

# Make kernalized multi-head attention block
class MultiHeadAttentionKernel(nn.Module):
    """Applies multi-head self-attention within a local receptive field defined by a kernel"""
    def __init__(self, n_features, n_heads=4, kernel_size=3):
        super(MultiHeadAttentionKernel, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        # Define attention heads
        self.attention_heads = nn.ModuleList([
            AttentionKernelBlock(n_features // n_heads, kernel_size) for _ in range(n_heads)
        ])

        # Define output projection
        self.out_proj = nn.Conv3d(n_features, n_features, kernel_size=1)

    def forward(self, x):

        # Split input into chunks
        x_list = x.chunk(self.n_heads, dim=1)

        # Apply attention heads
        x_list = [head(x) for head, x in zip(self.attention_heads, x_list)]

        # Merge heads
        x = torch.cat(x_list, dim=1)
        x = self.out_proj(x)

        # Return output
        return x



# Define convolutional transformer block
class ConvTransformerBlock(nn.Module):
    """Convolutional Transformer Block using kernelized self-attention"""
    def __init__(self, n_features, kernel_size=3, n_heads=4, expansion=2):
        super(ConvTransformerBlock, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.expansion = expansion

        # Define layers
        self.norm1 = nn.GroupNorm(1, n_features)  # Normalization layer
        self.attn = MultiHeadAttentionKernel(n_features, n_heads, kernel_size)  # Multi-head attention
        self.norm2 = nn.GroupNorm(1, n_features)  # Normalization layer
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features * expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(n_features * expansion, n_features, kernel_size=1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Local attention
        x = x + self.mlp(self.norm2(x))   # MLP for depth-wise feature mixing
        return x


# Define full convolutional transformer model
class Convformer(nn.Module):
    """Full Convolutional Transformer model"""
    def __init__(self,
        in_channels, out_channels,
        n_features=8, num_layers=6, n_heads=1, kernel_size=3
    ):
        super(Convformer, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        # Define input block
        self.input_block = nn.Sequential(
            nn.Conv3d(in_channels, n_features, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_features, affine=True),
            nn.ReLU(inplace=True),
        )

        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    ConvTransformerBlock(n_features, kernel_size=kernel_size, n_heads=n_heads)
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
        for layer in self.layers:
            print('------layer------')
            x = layer(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x

# Example usage
if __name__ == '__main__':

    # Import libraries
    import time

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    # Create data
    x = torch.randn(1, 35, 128, 128, 128)

    # Create model
    model = Convformer(in_channels=35, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} parameters')

    # Forward pass
    t0 = time.time()
    y = model(x)
    print(f'Forward pass took {time.time() - t0:.2f} seconds on {device}')

    # Done
    print("Done!")


