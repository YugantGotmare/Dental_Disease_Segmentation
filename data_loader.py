import os
import json
import numpy as np
from PIL import Image
from collections import Counter

def load_annotations(json_path):
    """
    Loads COCO-style annotations from a JSON file.
    
    Parameters:
    json_path (str): Path to the JSON file containing annotations.
    
    Returns:
    dict: Loaded annotations.
    """
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

# Load and preprocess image and mask, resizing to the target size
def load_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    """
    Loads an image and its corresponding mask, resizes them to the target size.
    
    Parameters:
    image_path (str): Path to the image file.
    mask_path (str): Path to the mask file.
    target_size (tuple): Target size to resize images and masks.
    
    Returns:
    tuple: Resized image and mask as numpy arrays.
    """
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    mask = mask.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image), np.array(mask)

# Count samples by category in the annotations
def count_samples_by_category(annotations):
    """
    Counts the number of samples for each category in the annotations.
    
    Parameters:
    annotations (dict): Loaded annotations containing category information.
    
    Returns:
    Counter: A dictionary-like object with category counts.
    """
    category_counts = Counter()
    for annotation in annotations['annotations']:
        category_counts[annotation['category_id']] += 1
    return category_counts

# Generator that yields batches of images and masks for training
def dataset_generator(image_dir, mask_dir, annotations, batch_size, target_size=(256, 256)):
    """
    Generates batches of images and masks for training.
    
    Parameters:
    image_dir (str): Directory containing image files.
    mask_dir (str): Directory containing mask files.
    annotations (dict): COCO-style annotations.
    batch_size (int): Number of samples per batch.
    target_size (tuple): Target size for images and masks.
    
    Yields:
    tuple: Batches of images and masks as numpy arrays.
    """
    image_info = annotations['images']
    while True:
        np.random.shuffle(image_info)
        for batch_start in range(0, len(image_info), batch_size):
            images, masks = [], []
            for i in range(batch_start, min(batch_start + batch_size, len(image_info))):
                image_data = image_info[i]
                image_filename = image_data['file_name']
                image_path = os.path.join(image_dir, image_filename)
                mask_filename = f"{image_filename}_mask.png"
                mask_path = os.path.join(mask_dir, mask_filename)
                try:
                    image, mask = load_image_and_mask(image_path, mask_path, target_size)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error loading image or mask: {e}")
                    continue
                images.append(image / 255.0)  # Normalize images
                masks.append(mask / 255.0)    # Normalize masks
            yield np.array(images), np.array(masks)
