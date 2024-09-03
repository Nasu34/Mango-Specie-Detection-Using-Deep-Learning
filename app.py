import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define the path to the model
model_path = 'mango_identification_model2.h5'

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define image dimensions based on model input
img_height, img_width = 224, 224  # Update this to match your model's expected input

# Define class labels
class_labels = [
    'Alphanso', 'Amarpali', 'Amarpali 13_1', 'Amarpali 4_9', 'Amarpali Desi',
    'Ambika', 'Austin', 'Chausa 13_1', 'Chausa Desi', 'Dasheri Desi',
    'Dasherion 13_1', 'Dusheri4_9', 'Duthpedha', 'Farnadeen', 'Kent',
    'Keshar', 'Langra Desi', 'Langra 13_1', 'Langra 4_9', 'Lilly',
    'Mallika', 'Neelam', 'Pamar', 'Pusa Lalima', 'Pusa Peetamber',
    'Pusa Pratibha', 'Pusa Shresth', 'Pusa Surya', 'Ram Kela Desi',
    'Ramkela 13_1', 'Ramkela4_9', 'Ratna'
]

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((img_width, img_height))  # Resize image to match model input
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
st.title('Mango Specie Detection using Deep Learning')
st.write('Upload an image of a mango leaf to get the prediction.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f'Predicted Class: {predicted_class}')
