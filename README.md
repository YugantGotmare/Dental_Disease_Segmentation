# Dental Disease Segmentation

This project leverages a U-Net architecture with a VGG16 encoder to segment dental X-ray images, aiming to identify dental conditions like Caries, Crown, Filling, Implant, and more. The model is trained on annotated data to enhance diagnostic accuracy in dental practices.

## Project Structure

The project is divided into modular Python files for easier management and scalability:

1. **`data_loader.py`**
   - Handles loading and preprocessing of images and masks.
   - Functions include reading COCO-style annotations, resizing images, and generating batches for training.

2. **`visualization.py`**
   - Contains functions to visualize original images, ground truth masks, and their overlays.
   - Useful for dataset inspection and result analysis.

3. **`model.py`**
   - Defines the U-Net model architecture with a VGG16 backbone, using Keras and TensorFlow.

4. **`train.py`**
   - Contains the training loop and functions for loading data.
   - Manages model compilation, training, and evaluation.

5. **`main.py`**
   - The main script for orchestrating the workflow, including data loading, model training, and saving the trained model.

## How to Run the Project

### 1. Install Dependencies

Install the required Python packages by running:

pip install -r requirements.txt

### 2. Prepare the Dataset

Download the dataset from [roboflow](https://universe.roboflow.com/arshs-workspace-radio/vzrad2)

### 3. Sample of Dataset
![Sample Dental X-ray Image](https://github.com/YugantGotmare/Dental_Disease_Segmentation/raw/main/Data/data.png)





