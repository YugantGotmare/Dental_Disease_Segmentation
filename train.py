import os
from data_loader import dataset_generator, load_annotations
from model import unet_vgg16_model

# Train the U-Net model with a data generator
def train_model(image_dir, mask_dir, annotations_path, batch_size, epochs, input_shape=(256, 256, 3)):
    """
    Trains the U-Net model using image and mask generators.
    
    Parameters:
    image_dir (str): Path to the directory containing images.
    mask_dir (str): Path to the directory containing masks.
    annotations_path (str): Path to the COCO-style annotations JSON file.
    batch_size (int): Batch size for training.
    epochs (int): Number of training epochs.
    input_shape (tuple): Input shape of the model.
    
    Returns:
    Model: Trained U-Net model.
    """
    annotations = load_annotations(annotations_path)
    model = unet_vgg16_model(input_shape)
    
    # Create a generator for training data
    train_gen = dataset_generator(image_dir, mask_dir, annotations, batch_size, target_size=input_shape[:2])

    # Train the model
    steps_per_epoch = len(annotations['images']) // batch_size
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    return model

# Save the trained model to a file
def save_model(model, model_path):
    """
    Saves the trained model to a file.
    
    Parameters:
    model (Model): Trained Keras model.
    model_path (str): Path to save the model.
    """
    model.save(model_path)
