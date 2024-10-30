import os
from train import train_model, save_model

def main():
    """
    Main function to execute the model training and saving pipeline.
    """
    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'
    annotations_path = 'path/to/annotations.json'
    model_path = 'saved_model.h5'
    batch_size = 8
    epochs = 10
    input_shape = (256, 256, 3)
    
    # Train the model
    model = train_model(image_dir, mask_dir, annotations_path, batch_size, epochs, input_shape)
    
    # Save the trained model
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
