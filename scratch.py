import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x and y values
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)

# Define a 2D unit Gaussian (without normalization)
Z = np.exp(-0.5 * (X**2 + Y**2))

# List of colormaps to use
colormaps = [
    'viridis', 'plasma', 'inferno',
    'cividis', 'RdBu', 'jet',
    'tab10', 'tab20', 'tab20b',
]

# Create a figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot the Gaussian in each subplot using a different colormap
for ax, cmap in zip(axes, colormaps):
    im = ax.imshow(Z, extent=(-1, 1, -1, 1), origin='lower', cmap=cmap)
    ax.set_title(cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Optionally add a colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
plt.savefig('_image.png')
plt.pause(0.1)

