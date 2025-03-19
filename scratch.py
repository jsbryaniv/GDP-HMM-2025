
# Import libraries
import torch
import torch.nn as nn



# Define memory-efficient MLP function for 3D volumes  
class EfficientMLPFunction3D(torch.autograd.Function):
    """Memory-efficient MLP for 3D volumes."""
    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2):
        """Forward pass without storing intermediate activations"""

        # Save shape (Batch, Channels, Depth, Height, Width)
        B, C, D, H, W = x.shape
        ctx.shape = (B, C, D, H, W)

        # Reshape to (B, D*H*W, C) for matrix multiplication
        out = x.view(B, C, -1).transpose(1, 2).contiguous()

        # Save only inputs, not activations
        ctx.save_for_backward(out, w1, w2, b1, b2)

        # Forward pass
        out = torch.matmul(out, w1) + b1     # Expansion
        out = torch.nn.functional.relu(out)  # ReLU activation
        out = torch.matmul(out, w2) + b2     # Contraction

        # Reshape back to (Batch, Channels, Depth, Height, Width)
        out = out.transpose(1, 2).view(B, C, D, H, W).contiguous()

        # Return
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Recompute forward pass before calculating gradients"""

        # Get saved tensors
        B, C, D, H, W = ctx.shape
        x, w1, w2, b1, b2 = ctx.saved_tensors

        # Reshape to (B, D*H*W, C)
        grad_output = grad_output.view(B, C, -1).transpose(1, 2).contiguous()

        # Recompute forward pass
        hidden = torch.nn.functional.relu(torch.matmul(x, w1) + b1)

        # Compute gradients of hidden layer
        grad_relu = torch.where(hidden > 0, torch.ones_like(hidden), torch.zeros_like(hidden))
        grad_hidden = torch.matmul(grad_output, w2.T)     # Backprop through second layer
        grad_hidden = grad_hidden * grad_relu             # Apply ReLU

        # Calculate gradients of other tensors
        grad_w2 = torch.matmul(hidden.transpose(1, 2).contiguous(), grad_output)
        grad_b2 = grad_output.sum(dim=0)
        grad_w1 = torch.matmul(x.transpose(1, 2).contiguous(), grad_hidden)
        grad_b1 = grad_hidden.sum(dim=0)
        grad_x = torch.matmul(grad_hidden, w1.T)

        # Reshape back to (Batch, Channels, Depth, Height, Width)
        grad_x = grad_x.transpose(1, 2).view(B, C, D, H, W).contiguous()

        # Return gradients
        return grad_x, grad_w1, grad_w2, grad_b1, grad_b2


# Define memory-efficient MLP layer for 3D volumes
class EfficientMLP3D(nn.Module):
    """Memory-efficient MLP for 3D feature maps"""
    def __init__(self, n_features, expansion=4):
        super().__init__()
        hidden_dim = n_features * expansion
        self.w1 = nn.Parameter(torch.randn(n_features, hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, n_features) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        return EfficientMLPFunction3D.apply(x, self.w1, self.w2, self.b1, self.b2)

# Define standard MLP layer for 3D volumes
class MLP3d(nn.Module):
    def __init__(self, n_features, expansion=4):
        super(MLP3d, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.expansion = expansion

        # Define MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(n_features, n_features * expansion, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(n_features * expansion, n_features, kernel_size=1),
        )

    def forward(self, x):
        return self.mlp(x)
    

# Define benchmarking function
def benchmark_memory_and_speed(model, input_tensor, runs=20):
    """Benchmark memory usage and execution speed of forward & backward passes."""

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Move model to evaluation mode to disable batch norm updates (if any)
    model.eval()

    # Ensure memory tracking is reset
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    forward_times = []
    backward_times = []
    peak_memories = []

    for _ in range(runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        input_tensor.requires_grad_(True)
        
        # Start measuring memory
        torch.cuda.synchronize()
        start_event.record()

        # Forward pass
        output = model(input_tensor)
        
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)  # ms
        forward_times.append(forward_time)

        # Backward pass
        grad_tensor = torch.ones_like(output)
        start_event.record()
        output.backward(grad_tensor)
        end_event.record()
        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event)  # ms
        backward_times.append(backward_time)

        # Peak memory usage
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
        peak_memories.append(peak_memory)

    # Compute averages
    avg_forward_time = sum(forward_times) / runs
    avg_backward_time = sum(backward_times) / runs
    avg_peak_memory = sum(peak_memories) / runs

    # Return results
    return {
        "avg_frwd_time_ms": avg_forward_time,
        "avg_bkwd_time_ms": avg_backward_time,
        "avg_peak_mem_mb": avg_peak_memory
    }


# Test MLP3d
if __name__ == '__main__':

    # Force consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Set up MLPs 
    mlp1 = MLP3d(n_features=8, expansion=4)
    mlp2 = EfficientMLP3D(n_features=8, expansion=4)

    # Force MPS to have same weights
    mlp2.w1.data = mlp1.mlp[0].weight.data[:, :, 0, 0, 0].T
    mlp2.w2.data = mlp1.mlp[2].weight.data[:, :, 0, 0, 0].T
    mlp2.b1.data = mlp1.mlp[0].bias.data
    mlp2.b2.data = mlp1.mlp[2].bias.data

    # Set up input
    x1 = torch.randn(3, 8, 128, 128, 128, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    # Forward pass
    y1 = mlp1(x1)
    y2 = mlp2(x2)

    # Test backward pass
    y1.sum().backward()
    y2.sum().backward()

    # Compare outputs
    print("--- Output and Gradient Comparison ---")
    print('Output difference:', (y1 - y2).abs().max().item())
    print('Gradient difference:', (x1.grad - x2.grad).abs().max().item())

    # # Run benchmarks
    # x1 = x1.detach().clone().requires_grad_(True)  # Reset x1
    # x2 = x1.detach().clone().requires_grad_(True)  # Reset x2
    # network1 = nn.Sequential(*[MLP3d(n_features=8, expansion=4) for _ in range(3)])
    # network2 = nn.Sequential(*[EfficientMLP3D(n_features=8, expansion=4) for _ in range(3)])
    # mlp1_results = benchmark_memory_and_speed(network1, x1)
    # mlp2_results = benchmark_memory_and_speed(network2, x2)

    # # Print results
    # print("\n--- Benchmark Results ---")
    # for ((k1, v1), (_, v2)) in zip(mlp1_results.items(), mlp2_results.items()):
    #     print(f"{k1}: Standard={v1:.2f}ms, Efficient={v2:.2f}ms, Ratio={v1/v2:.2f}.")

    # Done
    print('Done.')

