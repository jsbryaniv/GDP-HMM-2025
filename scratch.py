
import torch
import torch.nn as nn

"""
Instead of expanding with nn.ConvTranspose3d(n_features, n_features, kernel_size=scale, stride=scale),
which has n_features^2*scale^3 parameters, we instead break expansion into a series of smaller 
expansions, each with n_features*2^3 parameters.
"""


# Define PatchExpand class
class PatchExpand3d(nn.Module):
    """Patch Expansion module."""
    def __init__(self, n_features, scale, buffer=0, groups=1):
        super(PatchExpand3d, self).__init__()

        # Asset scale is power of 2 and buffer is either 0 or scale/2
        assert scale & (scale - 1) == 0, "Scale must be power of 2."
        assert buffer in [0, scale//2], "Buffer must be 0 or scale/2."
        
        # Set attributes
        self.n_features = n_features
        self.scale = scale
        self.buffer = buffer
        self.groups = groups

        # Calculate constants
        n = int(scale.bit_length() - 1)
        scaled_buffer = buffer // n
        kernel_size = 2 * (1 + scaled_buffer)
        stride = 2
        padding = scaled_buffer

        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        n_features, n_features,
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding,
                        groups=groups,
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
    
# Define PatchContract class
class PatchContract3d(nn.Module):
    """Patch Contraction module."""
    def __init__(self, n_features, scale, buffer=0, groups=1):
        super(PatchContract3d, self).__init__()

        # Asset scale is power of 2 and buffer is either 0 or scale/2
        assert scale & (scale - 1) == 0, "Scale must be power of 2."
        assert buffer in [0, scale//2], "Buffer must be 0 or scale/2."
        
        # Set attributes
        self.n_features = n_features
        self.scale = scale
        self.buffer = buffer
        self.groups = groups

        # Calculate constants
        n = int(scale.bit_length() - 1)
        scaled_buffer = buffer // n
        kernel_size = 2 * (1 + scaled_buffer)
        stride = 2
        padding = scaled_buffer

        
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
                        groups=groups,
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


# Test
if __name__ == '__main__':

    # Set constants
    n_features = 128
    scale = 16
    buffer = 8
    
    # Create layers
    expand1 = PatchExpand3d(
        n_features=n_features,
        scale=scale,
        buffer=buffer,
    )
    print(f"Expand1 number of params={sum(p.numel() for p in expand1.parameters())}")
    expand2 = nn.ConvTranspose3d(  # Comparison
        n_features, n_features, 
        kernel_size=scale+2*buffer,
        stride=scale,
        padding=buffer,
    )
    print(f"Expand2 number of params={sum(p.numel() for p in expand2.parameters())}")
    contract1 = PatchContract3d(
        n_features=n_features,
        scale=scale,
        buffer=buffer,
    )
    print(f"Contract1 number of params={sum(p.numel() for p in contract1.parameters())}")
    contract2 = nn.Conv3d(  # Comparison
        n_features, n_features, 
        kernel_size=scale+2*buffer,
        stride=scale,
        padding=buffer,
    )
    print(f"Contract2 number of params={sum(p.numel() for p in contract2.parameters())}")
    
    # Create input tensor
    x = torch.randn(1, n_features, 4, 4, 4)
    
    # Forward pass
    y1 = expand1(x)
    y2 = expand2(x)
    z1 = contract1(y1)
    z2 = contract2(y2)
    
    # Print sizes
    print("Input size:", x.size())
    print("Expand1 size:", y1.size())
    print("Expand2 size:", y2.size())
    print("Contract1 size:", z1.size())
    print("Contract2 size:", z2.size())
    print("Done.")
    