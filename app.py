import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load TFLite model (FAST)
# ---------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------
# Prediction function
# ---------------------------
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((180, 180))

    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    result_index = np.argmax(output)

    return result_index

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)

# ---------------------------
# Home Page
# ---------------------------
if app_mode == "Home":
    st.header("🍃 APPLE PLANT DISEASE RECOGNITION SYSTEM")

    st.markdown("""
Welcome to the **Apple Plant Disease Recognition System**.

Upload an apple leaf image and the model will detect the disease.

### Steps
1. Go to **Disease Recognition**
2. Upload image
3. Click **Predict**

### Features
- Fast detection
- Simple UI
- ML-based prediction
""")

# ---------------------------
# About Page
# ---------------------------
elif app_mode == "About":
    st.header("About")

    st.markdown("""
### Dataset

Classes:
- Apple Healthy
- Apple Scab
- Apple Cedar Rust
- Apple Black Rot

Model: CNN
""")

# ---------------------------
# Prediction Page
# ---------------------------
elif app_mode == "Disease Recognition":

    st.header("🍃 Apple Leaf Disease Detection")

    test_image = st.file_uploader("Upload Leaf Image")

    if test_image is not None:

        image = Image.open(test_image)
        st.image(image, width=400)

        if st.button("Predict"):

            with st.spinner("Analyzing..."):

                result_index = model_prediction(test_image)

                class_names = [
                    "Apple Healthy",
                    "Apple Scab",
                    "Apple Cedar Rust",
                    "Apple Black Rot"
                ]

                st.success(f"Prediction: {class_names[result_index]}")

    else:
        st.info("Please upload an image.")
