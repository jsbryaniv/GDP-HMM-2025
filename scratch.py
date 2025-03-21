
# Import libraries
import torch
import torch.nn as nn

# Volume normalization
class VoxelNorm3d(nn.Module):
    """
    Normalize each voxel independently across channels.
    """
    def __init__(self, num_channels, eps=1e-10):
        super(VoxelNorm3d, self).__init__()
        self.norm = nn.LayerNorm((num_channels,), eps=eps, elementwise_affine=False)

    def forward(self, x):

        # Reshape to (B, D, H, W, C) so LayerNorm normalizes over C
        x = x.permute(0, 2, 3, 4, 1)

        # Apply normalization
        x = self.norm(x)

        # Restore original shape (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        # Return output
        return x

# Test the model
model = VoxelNorm3d(16)

# Create input
x = torch.rand((1, 16, 32, 32, 32))
x[:, 2] += 1

# Normalize
y = model(x)

# Print stats
print('x', x[0, :, 0, 0, 0].mean(), x[0, :, 0, 0, 0].std())
print('y', y[0, :, 0, 0, 0].mean(), y[0, :, 0, 0, 0].std())

# Done
print('Done!')

# Unet v0 (Original)        -- Test loss after 1 epoch = 4.6469  #5
# Unet v1 (GroupNorm)       -- Test loss after 1 epoch = 4.1289  #2
# Unet v2 (DyTanh)          -- Test loss after 1 epoch = 4.1599  #3
# Unet v3 (Sparse ConvNeXt) -- Test loss after 1 epoch = 4.1753  #4
# Unet v4 (Full ConvNeXt)   -- Test loss after 1 epoch = 4.0517  #1


# Unet v0 outfiles/logs/out_job5.txt:-- Average loss on test dataset: 3.8400  #1
# Unet v1 outfiles/logs/out_job6.txt:-- Average loss on test dataset: 4.4291  #3
# Unet v2 outfiles/logs/out_job7.txt:-- Average loss on test dataset: 4.9513  #4
# Unet v3 outfiles/logs/out_job8.txt:-- Average loss on test dataset: 3.9984  #2
# Unet v4 outfiles/logs/out_job9.txt:-- Average loss on test dataset: 3.6912  
