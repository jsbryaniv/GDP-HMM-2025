
# Import libraries
import torch
import torch.nn.functional as F


# Define function to resize 3D image
def resize_3d_image(image, target_shape):
    """
    Resize a 3D image to the target shape while maintaining aspect ratio.
    Automatically handles boolean data by using nearest-neighbor interpolation.

    Args:
        image (torch.Tensor): 3D image tensor of shape (C, D, H, W)
        target_shape (tuple): Desired shape (D_target, H_target, W_target)

    Returns:
        torch.Tensor: Resized and padded image.
        dict: Resize and padding parameters for reversal.
    """

    # Get image shape
    _, D, H, W = image.shape
    D_target, H_target, W_target = target_shape

    # Determine interpolation mode based on dtype
    is_boolean = image.dtype == torch.bool
    interp_mode = 'nearest' if is_boolean else 'trilinear'
    interp_align = False if interp_mode == 'trilinear' else None

    # Compute uniform scale factor
    scale = min(D_target / D, H_target / H, W_target / W)
    new_size = (int(D * scale), int(H * scale), int(W * scale))

    # Resize the image
    resized_image = F.interpolate(
        image.float().unsqueeze(0), 
        size=new_size, 
        mode=interp_mode, 
        align_corners=interp_align
    ).squeeze(0)
    if is_boolean:
        resized_image = resized_image.bool()

    # Compute padding
    pad_d = (D_target - new_size[0]) / 2
    pad_h = (H_target - new_size[1]) / 2
    pad_w = (W_target - new_size[2]) / 2
    pad = (int(pad_w), int(pad_w + 0.5), int(pad_h), int(pad_h + 0.5), int(pad_d), int(pad_d + 0.5))

    # Apply padding with the correct default value
    pad_value = False if is_boolean else image.min()
    padded_image = F.pad(resized_image, pad, mode='constant', value=pad_value)

    # Store parameters for inverse transformation
    transform_params = {
        "original_shape": (D, H, W),
        "scale": scale,
        "pad": pad
    }

    # Return output
    return padded_image, transform_params


# Define function to reverse resize 3D image
def reverse_resize_3d(image, transform_params):
    """
    Reverse the resize and padding operation.

    Args:
        image (torch.Tensor): Padded image tensor of shape (C, D_target, H_target, W_target)
        transform_params (dict): Parameters from forward transform

    Returns:
        torch.Tensor: Restored original-sized image.
    """

    # Get variables
    _, D_target, H_target, W_target = image.shape
    D, H, W = transform_params["original_shape"]
    pad = transform_params["pad"]

    # Remove padding
    unpadded_image = image[:, pad[4]:D_target-pad[5], pad[2]:H_target-pad[3], pad[0]:W_target-pad[1]]

    # Determine interpolation mode
    is_boolean = image.dtype == torch.bool
    interp_mode = 'nearest' if is_boolean else 'trilinear'
    interp_align = False if interp_mode == 'trilinear' else None

    # Resize back to original shape
    restored_image = F.interpolate(
        unpadded_image.float().unsqueeze(0), 
        size=(D, H, W), 
        mode=interp_mode, 
        align_corners=interp_align
    ).squeeze(0)
    if is_boolean:
        restored_image = restored_image.bool()

    # Return output
    return restored_image



# Example usage
if __name__ == "__main__":

    # Import libraries
    import matplotlib.pyplot as plt
    from dataset import GDPDataset

    # Create dataset
    dataset = GDPDataset(
        treatment='Lung', 
        # shape=(128, 128, 128),
        return_dose=True,
    )

    # Get first item
    ct, beam, ptvs, oars, body, dose = dataset[0]

    # Test function
    x0 = oars[0].unsqueeze(0)
    x1, tp = resize_3d_image(x0, (64, 65, 66))
    x2 = reverse_resize_3d(x1, tp)

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    plt.ion()
    plt.show()
    ax[0].set_title('Original')
    ax[0].imshow(x0[0, x0.shape[1]//2])
    ax[1].set_title('Transformed')
    ax[1].imshow(x1[0, x1.shape[1]//2])
    ax[2].set_title('Restored')
    ax[2].imshow(x2[0, x2.shape[1]//2])
    ax[3].set_title('Difference')
    # ax[3].imshow((x0 - x2)[0, x0.shape[1]//2])
    plt.tight_layout()
    plt.pause(.1)
    plt.savefig('_image.png')

    # Done
    print('Done.')
