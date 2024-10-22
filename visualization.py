import numpy as np
import matplotlib.pyplot as plt

# Visualize original images, masks, and overlays
def visualize_combined(images, masks, num_samples=1):
    """
    Displays a set of images, masks, and their overlays.
    
    Parameters:
    images (numpy array): Array of images.
    masks (numpy array): Array of masks.
    num_samples (int): Number of samples to display.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    # Ensure compatibility for single or multiple samples
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # Show original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image {i + 1}')
        axes[i, 0].axis('off')

        # Show mask
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth Mask {i + 1}')
        axes[i, 1].axis('off')

        # Overlay the mask on the image
        overlay = images[i].copy()
        overlay[masks[i] > 0] = [255, 0, 0]  # Mark masked areas in red
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay {i + 1}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Plot images with masks overlaid and category names
def plot_images_with_masks(images, masks, categories, category_names):
    """
    Plots a batch of images with their corresponding masks and categories.
    
    Parameters:
    images (list): List of image file paths.
    masks (list): List of mask file paths.
    categories (list): List of category IDs corresponding to each image.
    category_names (dict): Dictionary mapping category IDs to category names.
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
    for i, (image_path, mask_path, category) in enumerate(zip(images, masks, categories)):
        row, col = divmod(i, 4)

        # Show the image
        ax_image = axes[0, col]
        ax_image.imshow(image_path)
        ax_image.axis('off')
        ax_image.set_title(f"Category: {category_names[category]}")

        # Show the mask overlay
        ax_mask = axes[1, col]
        ax_mask.imshow(image_path)
        ax_mask.imshow(mask_path, cmap='jet', alpha=0.55)  # Overlay with transparency
        ax_mask.axis('off')
    
    plt.tight_layout()
    plt.show()
