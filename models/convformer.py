
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

        # Learnable kernel weights
        self.q_kernel = nn.Parameter(.1*torch.randn(1, n_features, kernel_size, kernel_size, kernel_size))
        self.v_kernel = nn.Parameter(.1*torch.randn(1, n_features, kernel_size, kernel_size, kernel_size))

    def forward(self, x):

        # Project inputs into Q, K, V
        B, C, D, H, W = x.shape
        Q, K, V = self.qkv_proj(x).chunk(3, dim=1)  # Split into Q, K, V

        # Multiply Q and K with kernel weights
        Q = F.conv3d(Q, self.q_kernel, padding=self.kernel_size // 2)
        V = F.conv3d(V, self.v_kernel, padding=self.kernel_size // 2)

        # Calculate attention weights
        attn_weights = (Q * K).softmax(dim=2)

        # Apply attention weights
        x = attn_weights * V

        # Finalize output
        x = x.view(B, C, D, H, W)  # Reshape back
        x = self.out_proj(x)  # Final projection

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
            x = layer(x)

        # Output block
        x = self.output_block(x)

        # Return output
        return x

# Example usage
if __name__ == '__main__':
    
    # Create data
    x = torch.randn(1, 35, 128, 128, 128)

    # Create model
    model = Convformer(in_channels=35, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {n_params} parameters')

    # Forward pass
    y = model(x)

    # Done
    print("Done!")


