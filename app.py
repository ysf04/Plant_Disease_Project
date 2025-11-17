import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --- CONFIGURATION ---
file_id = '1cE1YQoCdWpvxbJwXvJsm3Le9XXW0Jqb7' 
model_filename = 'plant_disease_model.tflite'

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

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🔍 Model Diagnostic Mode")
st.write("Upload a leaf image to find the correct settings.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # GET MODEL EXPECTATIONS
    expected_shape = input_details[0]['shape']
    expected_dtype = input_details[0]['dtype']
    
    st.write(f"**Model Expects Shape:** `{expected_shape}`")
    st.write(f"**Model Expects Data Type:** `{expected_dtype}`")

    # Prepare Base Image
    target_h, target_w = expected_shape[1], expected_shape[2]
    img_resized = image.resize((target_w, target_h))
    img_array = np.array(img_resized)

    # TEST 1: STANDARD (0 to 1)
    # This is what we tried first
    input_1 = img_array.astype(np.float32) / 255.0
    input_1 = np.expand_dims(input_1, axis=0)
    
    # TEST 2: INCEPTION (-1 to 1)
    # This is what you remembered
    input_2 = img_array.astype(np.float32)
    input_2 = (input_2 - 127.5) / 127.5
    input_2 = np.expand_dims(input_2, axis=0)

    # TEST 3: RAW FLOATS (0 to 255)
    # This is likely the fix if the others failed
    input_3 = img_array.astype(np.float32)
    input_3 = np.expand_dims(input_3, axis=0)

    # Run Inference on ALL 3
    results = []
    
    for name, data in [("0 to 1", input_1), ("-1 to 1", input_2), ("0 to 255", input_3)]:
        try:
            interpreter.set_tensor(input_details[0]['index'], data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            top_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100
            results.append(f"**{name}:** {top_class} ({confidence:.2f}%)")
        except Exception as e:
            results.append(f"**{name}:** Failed ({e})")

    st.success("### Test Results")
    for res in results:
        st.markdown(res)
    
    st.info("👉 The one with high confidence (>80%) is the correct one!")
