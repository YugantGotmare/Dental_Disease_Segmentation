# Dental Disease Segementation

This project uses a U-Net architecture with a VGG16 encoder model to segment dental X-ray images, identifying conditions like Caries, Crown, Filling, Implant, and more. The model is trained on annotated data, aiming to improve diagnostic accuracy in dental practice.

## Project Structure

The project is divided into several Python files for modularity:

1. **`data_loader.py`**:
    - Responsible for loading and preprocessing the images and masks.
    - Contains functions to read COCO-style annotations, resize images, and generate batches for model training.
  
2. **`visualization.py`**:
    - Provides functions for visualizing the original images, ground truth masks, and their overlays.
    - Useful for visually inspecting the dataset and results.

3. **`model.py`**:
    - Defines the U-Net model architecture using a VGG16 backbone.
    - The model is built using Keras and TensorFlow.
      
4. **`train.py`**:
    - Contains the training loop and functions for loading data using the generator.
    - Handles model compilation and training.

5. **`main.py`**:
    - The main script to train the model and save the trained model to disk.
    - Orchestrates the overall workflow, including data loading, training, and saving.
