import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

file_id = '1822ImFpXX8cuknje0mUVj8VFLPOdaU2w'

@st.cache_resource
def load_model_from_drive():
    output_path = 'plant_disease_model.h5'
    
    # Check if file already exists to avoid redownloading
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        st.write("Downloading model... this may take a minute.")
        gdown.download(url, output_path, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(output_path)
    return model

# Load the model (this triggers the download only on the first run)
try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Define Class Names (Same as before)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# 3. App Interface
st.title("Plant Disease Classifier 🌿")
st.write("Upload an image of a plant leaf to detect diseases.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    img_array = image.resize((224, 224)) # Ensure this matches your training input
    img_array = np.array(img_array)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if model:
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")


