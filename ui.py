import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Set page configuration
st.set_page_config(page_title="Dental X-Ray Segmentation", layout="wide")

# Cache the model to load it only once
@st.cache_resource
def load_segmentation_model(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess input image
def preprocess_image(image, target_size=(256, 256)):
    image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension: (1, 256, 256, 3)

# Function to predict segmentation
def predict_segmentation(model, image, num_classes=4):
    try:
        pred = model.predict(image, verbose=0)
        if pred.shape[-1] == 1:  # Binary model
            return (pred[0] >= 0.5).astype(np.int32)  # Binary threshold
        else:  # Multi-class model
            return np.argmax(pred[0], axis=-1)  # Class indices: (256, 256)
    except Exception as e:
        st.error(f"Error predicting: {e}")
        return np.zeros((256, 256), dtype=np.int32)

# Function to visualize results
def visualize_results(image, prediction, category_names={0: 'vzrad2', 1: 'Caries', 2: 'Crown', 3: 'Filling'}):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original Image
    ax[0].imshow(image[0])  # Remove batch dimension
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Predicted Segmentation
    ax[1].imshow(image[0])
    ax[1].imshow(prediction, cmap='jet', alpha=0.5)
    ax[1].set_title('Predicted Segmentation')
    ax[1].axis('off')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("Dental X-Ray Segmentation")
    st.write("Upload a dental X-ray image to get a segmentation map highlighting regions like Caries, Crown, and Filling.")

    # Load model
    model_path = r'G:\MyZone\Dental_disease\vgg16_unet_model.h5'  # Adjust path if running locally or on cloud
    model = load_segmentation_model(model_path)
    
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png'])
    
    if uploaded_file is not None:
        # Read and display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Predict segmentation
        with st.spinner('Generating segmentation...'):
            prediction = predict_segmentation(model, processed_image)
        
        # Visualize results
        fig = visualize_results(processed_image, prediction)
        st.pyplot(fig)
        
        # Display class distribution
        unique, counts = np.unique(prediction, return_counts=True)
        st.subheader("Predicted Class Distribution")
        category_names = {0: 'vzrad2', 1: 'Caries', 2: 'Crown', 3: 'Filling'}
        for cls, count in zip(unique, counts):
            st.write(f"{category_names.get(cls, 'Unknown')}: {count} pixels")

if __name__ == "__main__":
    main()