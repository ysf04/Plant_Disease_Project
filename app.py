import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --- CONFIGURATION ---
# REPLACE THIS with your actual Google Drive File ID
file_id = '1cE1YQoCdWpvxbJwXvJsm3Le9XXW0Jqb7' 
model_filename = 'plant_disease_model.tflite'

# Class Names
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

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(model_filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_filename, quiet=False)

    interpreter = tf.lite.Interpreter(model_path=model_filename)
    interpreter.allocate_tensors()
    return interpreter

# Load the model once
try:
    interpreter = load_tflite_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# App Interface
st.title("Plant Disease Classifier 🌿")
st.write("Upload an image of a plant leaf to detect diseases.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])
st.info("📸 Tip: For best results, take a close-up photo containing ONE leaf in the center.")
if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # --- PREPROCESSING (The Winning Logic: 0 to 255) ---
    # 1. Resize to 224x224
    img = image.resize((224, 224))
    
    # 2. Convert to Array
    img_array = np.array(img)
    
    # 3. Convert to Float32, but DO NOT DIVIDE by 255
    # This keeps the values between 0.0 and 255.0
    input_data = img_array.astype(np.float32)
    
    # 4. Add Batch Dimension
    input_data = np.expand_dims(input_data, axis=0)

    # --- INFERENCE ---
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # --- RESULTS ---
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # Handle confidence display (0-1 vs 0-100)
    if confidence <= 1.0:
        confidence_percent = confidence * 100
    else:
        confidence_percent = confidence

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_percent:.2f}%")


