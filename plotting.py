
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

# Plot results of a single slice
def plot_prediction(scan, target, prediction, z=None):
    """
    Plot results of a single slice.
    Args:
        scan (array): Input scan.
        target (array): Ground truth.
        prediction (array): Predicted.
        z (int): Slice to plot.
    """

    # Convert to numpy
    if isinstance(scan, torch.Tensor):
        scan = scan.cpu().detach().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()

    # Set up figure
    fig, ax = plt.subplots(1, 3)
    plt.ion()
    plt.show()

    # Find middle slice
    if z is None:
        z = target.shape[-3] // 2

    # Plot data
    ax[0].set_title('Input')
    ax[0].imshow(scan[0, 0, z, :, :], cmap='gray')
    ax[1].set_title('Target')
    ax[1].imshow(target[0, 0, z, :, :], cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].imshow(prediction[0, 0, z, :, :] > .5, cmap='gray')

    # Finalize plot
    plt.tight_layout()
    plt.pause(0.1)

    # Return figure and axis
    return fig, ax

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
    ax.set_ylim([0, max(losses_val)])
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
        structure = structures[0, i]
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



