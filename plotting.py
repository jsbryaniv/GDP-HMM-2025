
# Import libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Copy axes function
def copy_axis(ax_from, ax_to):
    """Copy all attributes, plots, and artist objects dynamically from one axis to another."""

    # # Copy all artists (lines, scatter, etc.)
    # artist_attrs = [attr for attr in dir(ax_from) if isinstance(getattr(ax_from, attr, None), list)]
    # for attr in artist_attrs:
    #     for artist in getattr(ax_from, attr):
    #         try:
    #             ax_to.add_artist(artist)
    #         except:
    #             pass  # Skip non-artist objects

    # Copy all line plots
    for line in ax_from.lines:
        ax_to.plot(
            line.get_xdata(), 
            line.get_ydata(), 
            label=line.get_label(),
            color=line.get_color(), 
            linestyle=line.get_linestyle(), 
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize()
        )

    # Copy all scatter plots
    for collection in ax_from.collections:
        if isinstance(collection, plt.PathCollection):  # Ensures it's from scatter()
            offsets = collection.get_offsets()
            facecolors = collection.get_facecolors()
            edgecolors = collection.get_edgecolors()
            sizes = collection.get_sizes()

            ax_to.scatter(
                offsets[:, 0], offsets[:, 1], 
                s=sizes if sizes.size > 0 else None,  # Preserve size if available
                c=facecolors if len(facecolors) > 0 else None,  # Preserve colors
                edgecolors=edgecolors if len(edgecolors) > 0 else None,
                alpha=collection.get_alpha(),
                label="Copied Scatter"
            )

    # Copy all image plots
    for img in ax_from.images:
        ax_to.imshow(
            img.get_array(), 
            cmap=img.get_cmap(), 
            extent=img.get_extent(),
            alpha=img.get_alpha(), 
            interpolation=img.get_interpolation()
        )

    # Copy labels, limits, and grid
    ax_to.set_title(ax_from.get_title())
    ax_to.set_xlabel(ax_from.get_xlabel())
    ax_to.set_ylabel(ax_from.get_ylabel())
    ax_to.set_xlim(ax_from.get_xlim())
    ax_to.set_ylim(ax_from.get_ylim())

    # Copy legend if it exists
    legend = ax_from.get_legend()
    if legend:
        ax_to.legend(loc=legend._loc)

    # Done
    return ax_to

# Plot training and validation losses
def plot_losses(losses_train, losses_val):
    """
    Plot training and validation losses.
    Args:
        losses_train (list): Training losses.
        losses_val (list): Validation losses.
    """

    # Set up figure
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    # Plot losses
    ax.set_title('Losses')
    ax.plot(losses_train, label='Train')
    ax.plot(losses_val, label='Validation')

    # Finalize plot
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

# Plot dvhs
def plot_dvh(dose, structures, labels=None, bins=100, ax=None):
    """
    Plot the dose-volume histogram.
    """

    # Check inputs
    if isinstance(dose, torch.Tensor):
        # Convert dose to numpy
        dose = dose.cpu().detach().numpy()
    if isinstance(structures, torch.Tensor):
        # Convert structures to numpy
        structures = structures.cpu().detach().numpy()
    if len(dose.shape) == 4:
        # Remove batch dimension
        dose = dose[0]
    if len(structures.shape) == 4:
        # Remove batch dimension
        structures = structures[0]
    if labels is None:
        labels = [f'Structure {i+1}' for i in range(structures.shape[1])]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        plt.ion()
        plt.show()

    # Get dose range
    bins = np.linspace(0, np.max(dose), bins)

    # Loop over structures
    for i in range(structures.shape[1]):
        structure = structures[i]
        if np.sum(structure) == 0:
            continue

        # Get histogram
        hist, bin_edges = np.histogram(dose[structure.astype(bool)], bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get cumulative histogram (reverse order)
        cum_hist = np.cumsum(hist[::-1])[::-1] * 100

        # Plot
        ax.plot(bin_centers, cum_hist, label=labels[i])
    
    # If ax is provided, return it
    if ax is not None:
        return ax
    
    # Finalize plot
    ax.set_title('DVH')
    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume (%)')
    ax.legend()
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

# Quickly plot images
@torch.no_grad()
def plot_images(images=None, labels=None, cmap=None, **image_dict):
    """
    Quickly plot multiple images at once. Image arrays should be in the form (Batch, Channels, Height, Width).
    This function will plot each image in the batch as a separate row, row labels are optionally specified with
    the 'labels' argument. If multiple image arrays are provided, they will be plotted in different columns
    with column labels given by the keys of the 'image_dict' dictionary. Accepts both numpy arrays and torch tensors.

    Example use:

    n_images = 5
    x = torch.randn(n_images, 1, 128, 128)
    y = torch.randn(n_images, 3, 128, 128)
    labels = [0, 1, 2, 3, 'last']
    plot_images(hello=x, goodbye=y, labels=labels, cmap='gray')

    Output is a image with 2 columns and 5 rows:

    |      | "hello" | "goodbye" |
    |------|---------|-----------|
    | 0    |  x[0]   |   y[0]    |
    | 1    |  x[1]   |   y[1]    |
    | 2    |  x[2]   |   y[2]    |
    | 3    |  x[3]   |   y[3]    |
    | last |  x[4]   |   y[4]    |

    """

    # Check inputs
    if cmap is None:
        cmap = 'jet'

    # Set up image_dict
    if images is not None:
        image_dict['images'] = images
    num_arrays = len(image_dict.keys())

    # Send tensors to cpu and numpy
    for key in image_dict.keys():
        val = image_dict[key]
        if isinstance(val, torch.Tensor):
            image_dict[key] = image_dict[key].float().cpu().detach().numpy()
        elif isinstance(val, np.ndarray):
            image_dict[key] = image_dict[key].astype(float)

    # Assert that all image arrays have the same number of images
    num_images = image_dict[list(image_dict.keys())[0]].shape[0]
    for key in image_dict.keys():
        if image_dict[key].shape[0] != num_images:
            raise ValueError("All image arrays must have the same number of images.")

    # Set up colunm labels
    if labels is None:
        labels = [f'Image {i}' for i in range(num_images)]

    # Set up figure
    fig = plt.gcf()
    num_rows = num_images
    num_cols = num_arrays
    fig.set_size_inches(num_cols, num_rows)
    plt.clf()
    plt.ion()
    plt.show()
    ax = np.empty((num_rows, num_cols), dtype=object)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j] = fig.add_subplot(ax.shape[0], ax.shape[1], i * ax.shape[1] + j + 1)

    # Loop over image lists
    for i, (key, val) in enumerate(image_dict.items()):

        # Loop over images
        for j in range(val.shape[0]):

            # Get image
            img = val[j]

            # Slice batch if necessary
            if len(img.shape) > 3:
                img = img[0]

            # Pad image channels if necessary
            if img.shape[0] == 1:
                # Grayscale images, get rid of channel dimension
                img = np.squeeze(img)
            elif img.shape[0] == 2:
                # Two-channel images, add empty channel and transpose to RGB format
                img = np.concatenate((img, np.zeros((1,*img[0].shape))), axis=0)
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[0] == 3:
                # Three-channel images, transpose to RGB format
                img = np.transpose(img, (1, 2, 0))

            # Normalize image
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()

            # Plot image
            if len(img.shape) == 2:
                ax[j, i].imshow(img, cmap=cmap)  # Use cmap for grayscale images
            else:
                ax[j, i].imshow(img)

    # Finalize plot
    for j in range(ax.shape[1]):
        ax[0, j].set_title(list(image_dict.keys())[j])
    for i in range(ax.shape[0]):
        ax[i, 0].set_ylabel(f'Example {labels[i]}')
        for j in range(ax.shape[1]):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.pause(1)
    
    # Return
    return fig, ax


# Test
if __name__ == "__main__":

    # Test plot_images
    x = torch.randn(5, 1, 128, 128)
    y = torch.randn(5, 3, 128, 128)
    labels = [0, 1, 2, 3, 'last']
    plot_images(hello=x, goodbye=y, labels=labels, cmap='gray')
    plt.savefig('_image.png')

    # Done
    print('Done!')
