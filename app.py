import os

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Model download (if not available)
file_id = "1qQDXt0FnGMISyaqpLhbfoJv0TtFjdouC"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive (only once)...")
    gdown.download(url, model_path, quiet=False)
else:
    st.success("✅ Model already downloaded.")

# Load model function
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of the predicted class

# Sidebar navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.radio("Select Page:", ["Home", "Disease Recognition"])

# Display banner image
img = Image.open("diseases.jpg")
st.image(img, use_container_width=True)

# Homepage content
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)

# Disease Recognition page
elif app_mode == "Disease Recognition":
    st.header("🔬 Plant Disease Recognition System")

    # File uploader
    test_image = st.file_uploader("📸 Choose an Image:")

    # Layout buttons in two columns
    col1, col2 = st.columns([1, 1])

    # Show uploaded image only if test_image is not None
    with col1:
        if st.button("📷 Show The Image"):
            if test_image is not None:
                st.image(test_image, use_container_width=True)
            else:
                st.warning("⚠️ Please upload an image first.")

# Define information for each disease category
    disease_info = {
        "Potato___Early_blight": """🛑 **Early Blight**  
        - **Cause**: Fungus *Alternaria solani*  
        - **Symptoms**: Dark brown spots with concentric rings on leaves.  
        - **Prevention**: Avoid overhead watering, use resistant varieties, and apply fungicides.  
        """,
        
        "Potato___Late_blight": """⚠️ **Late Blight**  
        - **Cause**: Pathogen *Phytophthora infestans*  
        - **Symptoms**: Dark, water-soaked lesions that spread rapidly in humid conditions.  
        - **Prevention**: Remove infected plants, improve air circulation, and use fungicides.  
        """,

        "Potato___healthy": """✅ **Healthy Plant**  
        - Your plant appears **healthy** with no visible disease symptoms. 🎉  
        - Keep monitoring for any changes and maintain good farming practices!  
        """
    }

    # Predict button only if test_image is not None
    with col2:
        if st.button("🔍 Predict"):
            if test_image is not None:
                st.balloons()
                st.write("✅ **Our Prediction**")
                result_index = model_prediction(test_image)
                class_name = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
                predicted_category = class_name[result_index]

                # Display the prediction result with additional information
                st.success(f"🌿 **Model predicts:** {predicted_category}")
                st.write(disease_info[predicted_category])  # Show relevant disease details
            else:
                st.warning("⚠️ Please upload an image before predicting.")
