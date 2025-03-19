import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load pre-trained model (Ensure you have a trained model file named 'model.h5')
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input size
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.title("Handwritten Letter Recognition")
st.write("Upload a handwritten letter image and the model will predict the letter.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_letter = chr(np.argmax(prediction) + ord('A'))  # Assuming model is trained on uppercase letters A-Z
    
    st.write(f"### Predicted Letter: {predicted_letter}")
