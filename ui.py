import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

# Set page configuration with custom styling
st.set_page_config(
    page_title="🦷 Dental X-Ray AI Analyzer", 
    layout="wide",
    page_icon="🦷",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .analysis-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🦷 AI Dental X-Ray Analyzer</h1>
    <p>Advanced segmentation and analysis of dental X-ray images using deep learning</p>
</div>
""", unsafe_allow_html=True)

# Cache the model to load it only once
@st.cache_resource
def load_segmentation_model(model_path):
    try:
        with st.spinner('🔄 Loading AI model...'):
            model = load_model(model_path)
            st.success("✅ Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Function to preprocess input image
def preprocess_image(image, target_size=(256, 256)):
    image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Function to predict segmentation
def predict_segmentation(model, image, num_classes=4):
    try:
        pred = model.predict(image, verbose=0)
        if pred.shape[-1] == 1:
            return (pred[0] >= 0.5).astype(np.int32)
        else:
            return np.argmax(pred[0], axis=-1)
    except Exception as e:
        st.error(f"❌ Error predicting: {e}")
        return np.zeros((256, 256), dtype=np.int32)

# Enhanced visualization function
def create_interactive_visualization(image, prediction, category_names):
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original X-Ray Image', 'AI Segmentation Analysis'),
        specs=[[{"type": "image"}, {"type": "image"}]]
    )
    
    # Original image
    fig.add_trace(
        go.Image(z=image[0]),
        row=1, col=1
    )
    
    # Segmentation overlay
    fig.add_trace(
        go.Heatmap(
            z=prediction,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Dental Regions", x=1.02)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Dental X-Ray Analysis Results",
        title_x=0.5,
        height=500,
        showlegend=False
    )
    
    return fig

# Function to create analysis statistics
def create_analysis_stats(prediction, category_names):
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    # Create DataFrame for better visualization
    stats_data = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (cls, count) in enumerate(zip(unique, counts)):
        percentage = (count / total_pixels) * 100
        stats_data.append({
            'Region': category_names.get(cls, 'Unknown'),
            'Pixels': count,
            'Percentage': percentage,
            'Color': colors[i % len(colors)]
        })
    
    df = pd.DataFrame(stats_data)
    return df

# Main app function
def main():
    # Sidebar for controls and information
    with st.sidebar:
        st.markdown("### 📊 Analysis Controls")
        
        # Model information
        st.markdown("""
        <div class="info-card">
            <h4>🤖 AI Model Info</h4>
            <p><strong>Architecture:</strong> VGG16-UNet</p>
            <p><strong>Classes:</strong> 4 dental regions</p>
            <p><strong>Input Size:</strong> 256x256</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature information
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Detection Capabilities</h4>
            <ul>
                <li>🦷 <strong>Caries:</strong> Tooth decay detection</li>
                <li>👑 <strong>Crown:</strong> Dental crown identification</li>
                <li>🔧 <strong>Filling:</strong> Dental filling analysis</li>
                <li>📊 <strong>Background:</strong> Normal tissue</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage instructions
        with st.expander("📋 How to Use"):
            st.markdown("""
            1. **Upload** your dental X-ray image
            2. **Wait** for AI processing
            3. **Review** the segmentation results
            4. **Analyze** the statistical breakdown
            """)

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload section
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Dental X-Ray</h3>
            <p>Supported formats: JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image file",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a clear dental X-ray image for analysis"
        )
    
    with col2:
        # Quick stats placeholder
        if uploaded_file is None:
            st.markdown("""
            <div class="info-card">
                <h4>🔍 Waiting for Image Upload</h4>
                <p>Upload a dental X-ray image to begin AI analysis</p>
            </div>
            """, unsafe_allow_html=True)

    # Load model
    model_path = r'G:\MyZone\Dental_disease\weight\vgg16_unet_model.h5'
    model = load_segmentation_model(model_path)
    
    if model is None:
        st.error("❌ Model not available. Please check the model path.")
        st.stop()

    # Process uploaded image
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        img_col1, img_col2 = st.columns([1, 1])
        
        with img_col1:
            st.markdown("### 📷 Original X-Ray")
            st.image(image, use_column_width=True, caption="Uploaded dental X-ray")
            
            # Image info
            st.markdown(f"""
            <div class="info-card">
                <h4>📊 Image Information</h4>
                <p><strong>Dimensions:</strong> {image.size[0]} × {image.size[1]}</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>Mode:</strong> {image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with img_col2:
            # Preprocess and predict
            processed_image = preprocess_image(image)
            
            with st.spinner('🧠 AI is analyzing your X-ray...'):
                prediction = predict_segmentation(model, processed_image)
            
            st.markdown("### 🎯 AI Analysis Result")
            
            # Create and display interactive visualization
            category_names = {0: 'Normal/Background', 1: 'Caries', 2: 'Crown', 3: 'Filling'}
            
            # Create matplotlib visualization for display
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(processed_image[0])
            masked_prediction = np.ma.masked_where(prediction == 0, prediction)
            ax.imshow(masked_prediction, cmap='tab10', alpha=0.7, vmin=0, vmax=3)
            ax.set_title('AI Segmentation Analysis', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=plt.cm.tab10(i/3), label=category_names[i+1]) 
                             for i in range(3)]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            st.pyplot(fig)
        
        # Analysis results section
        st.markdown("### 📈 Detailed Analysis Results")
        
        # Create statistics
        stats_df = create_analysis_stats(prediction, category_names)
        
        # Display metrics in columns
        metric_cols = st.columns(len(stats_df))
        
        for i, (_, row) in enumerate(stats_df.iterrows()):
            with metric_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['Percentage']:.1f}%</h3>
                    <p>{row['Region']}</p>
                    <small>{row['Pixels']} pixels</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed breakdown
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Region Distribution")
            
            # Create pie chart
            fig_pie = px.pie(
                stats_df, 
                values='Percentage', 
                names='Region',
                title="Dental Region Distribution",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Analysis Summary")
            
            # Findings summary
            total_pathology = stats_df[stats_df['Region'].isin(['Caries', 'Crown', 'Filling'])]['Percentage'].sum()
            
            if total_pathology > 20:
                status = "⚠️ Significant findings detected"
                color = "#FF6B6B"
            elif total_pathology > 5:
                status = "⚡ Moderate findings detected"
                color = "#FFD93D"
            else:
                status = "✅ Minimal findings detected"
                color = "#6BCF7F"
            
            st.markdown(f"""
            <div class="info-card" style="border-left: 4px solid {color};">
                <h4>{status}</h4>
                <p><strong>Total pathology coverage:</strong> {total_pathology:.1f}%</p>
                <p><strong>Recommendation:</strong> Consult with dental professional for detailed evaluation</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data table
            st.markdown("#### 📋 Detailed Breakdown")
            st.dataframe(
                stats_df[['Region', 'Percentage', 'Pixels']].round(2),
                use_container_width=True,
                hide_index=True
            )
        
        # Download section
        st.markdown("### 💾 Export Results")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📊 Download Analysis Report"):
                # Create a simple text report
                report = f"""
Dental X-Ray Analysis Report
============================
Image: {uploaded_file.name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Region Analysis:
{stats_df.to_string(index=False)}

Total Pathology Coverage: {total_pathology:.1f}%
Status: {status}
                """
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"dental_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Disclaimer
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #FF6B6B;">
            <h4>⚠️ Important Disclaimer</h4>
            <p>This AI analysis is for educational and research purposes only. 
            It should not replace professional dental diagnosis or treatment recommendations. 
            Always consult with qualified dental professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()