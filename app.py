import gdown
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
MODEL_PATH = "leaf_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1YH_P1G9k_ekC7eoYqEowJkG0Yo2P03VX"
    gdown.download(id=file_id, output=MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Load model once

# Prediction function
def model_prediction(test_image):
    image=Image.open(test_image)
    image = image.resize((180,180))

    input_arr = np.array(image)/255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)

    return result_index


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home","About","Disease Recognition"]
)

# Home Page
if(app_mode=="Home"):

    st.header("🍃 APPLE PLANT DISEASE RECOGNITION SYSTEM")

    st.markdown("""
Welcome to the **Apple Plant Disease Recognition System**.

Our system helps in identifying apple plant diseases efficiently.
Upload an image of an apple leaf and the model will detect the disease.

### How It Works

1. Go to **Disease Recognition**
2. Upload a leaf image
3. Click **Predict**
4. The system will show the predicted disease

### Why Use This System

- Fast detection
- Simple interface
- Machine Learning based prediction
""")


# About Page
elif(app_mode=="About"):

    st.header("About")

    st.markdown("""
### Dataset Information

The dataset contains images of healthy and diseased apple leaves.

Classes used in this project:

1. Apple Healthy
2. Apple Scab
3. Apple Cedar Rust
4. Apple Black Rot

The model was trained using a **Convolutional Neural Network (CNN)**.
""")


# Prediction Page
elif(app_mode=="Disease Recognition"):

    st.header("🍃 Apple Leaf Disease Detection")

    test_image = st.file_uploader("Upload Leaf Image")

    if test_image is not None:

        if st.button("Show Image"):
            image = Image.open(test_image)
            st.image(image, width=400)

        if st.button("Predict"):

            with st.spinner("Analyzing Leaf Image..."):

                result_index = model_prediction(test_image)

                class_names = [
                    "Apple Healthy",
                    "Apple Scab",
                    "Apple Cedar Rust",
                    "Apple Black Rot"
                ]

                st.success(
                    "Model Prediction: {}".format(class_names[result_index])
                )

    else:
        st.info("Please upload an image to continue.")
