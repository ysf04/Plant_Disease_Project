import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf # Or 'import tflite_runtime.interpreter as tflite'
import gdown
import os

# --- CONFIGURATION ---
# REPLACE THIS with the Google Drive ID of your NEW .tflite file
file_id = '1cE1YQoCdWpvxbJwXvJsm3Le9XXW0Jqb7' 
model_filename = 'plant_disease_model.tflite'

# Class names (same as before)
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

    # Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=model_filename)
    interpreter.allocate_tensors()
    return interpreter

# Load the model (This will be instant now!)
interpreter = load_tflite_model()

# Get input and output details to know how to feed data
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Plant Disease Classifier (TFLite) 🌿")
file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

# ... inside your main code block ...

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 1. Resize
    # Ensure this matches your training (try 224 first, if that fails try 256)
    img = image.resize((224, 224))
    img_array = np.array(img)

    # 2. Get Model Input Properties
    input_details = interpreter.get_input_details()
    input_type = input_details[0]['dtype']
    
    # 3. Apply the Correct Normalization
    if input_type == np.int8:
        # IF Model expects Integers (-128 to 127)
        # We shift 0..255 down to -128..127
        input_data = (img_array - 128).astype(np.int8)
        
    elif input_type == np.uint8:
        # IF Model expects Unsigned Integers (0 to 255)
        input_data = img_array.astype(np.uint8)
        
    else:
        # IF Model expects Floats (Standard for MobileNet/ResNet)
        # We apply the "Inception" normalization: (x - 127.5) / 127.5
        # This forces the range to be between -1.0 and 1.0
        input_data = img_array.astype(np.float32)
        input_data = (input_data - 127.5) / 127.5

    # 4. Add Batch Dimension
    input_data = np.expand_dims(input_data, axis=0)

    # 5. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_details = interpreter.get_output_details()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # ... rest of results code ...
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")



