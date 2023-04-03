import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('yawn_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image
    image = image.resize(img_size)
    # Convert the image to a NumPy array
    image = img_to_array(image)
    # Scale the pixel values to [0, 1]
    image = image / 255.0
    # Add an extra dimension to the array to represent the batch size
    image = np.expand_dims(image, axis=0)
    return image

# Define the Streamlit app
def app():
    st.title('Yawn Detection App')

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Preprocess the image
        image = preprocess_image(image)
        # Make a prediction
        pred = model.predict(image)
        pred_label = 'Yawning' if pred[0][0] > 0.5 else 'No Yawning'
        pred_prob = pred[0][0]
        # Show the prediction result
        st.write(f'Prediction: {pred_label}')
        st.write(f'Probability Of Yawning: {pred_prob:.2f}')

# Run the app
if __name__ == '__main__':
    app()
