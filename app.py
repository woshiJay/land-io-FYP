import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load your trained model
model = load_model('models/resnet50.h5')

def preprocess_image(image):
    # Resize and preprocess the image as done during training/validation
    image = image.resize((224, 224), Image.LANCZOS)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # Expand dims to match input shape of model

def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction, axis=1)

st.title('Satellite Image Classification')
st.write("Upload a satellite image to classify its land use.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classify_image(image)
    # Mapping the model's numerical labels to actual class names
    labels = {0: 'Agricultural', 1: 'Airplane', 2: 'Baseball Diamond', 3: 'Beach',
                4: 'Buildings', 5: 'Chaparral', 6: 'Dense Residential', 7: 'Forest',
                8: 'Freeway', 9: 'Golf Course', 10: 'Harbor', 11: 'Intersection',
                12: 'Medium Residential', 13: 'Mobile Home Park', 14: 'Overpass',
                15: 'Parking Lot', 16: 'River', 17: 'Runway', 18: 'Sparse Residential',
                19: 'Storage Tanks', 20: 'Tennis Court'
              } 
    st.write(f"Prediction: {labels[label[0]]}")