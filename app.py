# Imports and setup
import streamlit as st
import streamlit.components.v1 as components
import os
import random
import base64
import io
import cv2
import time
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Sections 4 & 5
import model_performance
import model_visualization

# --------------------------------------------------------------------
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Preprocessing", "Model Prediction", "Model Performance", "Model Visualization"])

# --------------------------------------------------------------------
# Main title
st.title("FYP APP")

# --------------------------------------------------------------------
# Section 1: Introduction and Overview
if page == "Introduction":
    st.header("UC Merced Dataset Showcase")

    # Function to load images from a folder
    def load_images_from_folder(folder, limit=15):
        images = []
        if not os.path.exists(folder):
            return images
        
        for filename in os.listdir(folder)[:limit]:
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img.thumbnail((250, 250))  # Resize image for faster loading
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        encoded_string = base64.b64encode(buffered.getvalue()).decode()
                        images.append(encoded_string)
                except Exception as e:
                    st.error(f"Error loading image {filename}: {str(e)}")
        return images

    # Define the base path of images
    base_path = "data/images_train_test_val/train"

    # Check if base path exists
    if not os.path.exists(base_path):
        st.error(f"Base path not found: {base_path}")
    else:
        # Initialize class_images in session state if not already present
        if 'class_images' not in st.session_state:
            st.session_state.class_images = {}
            random.seed(int(time.time()))  # Use current time as seed for carousel
            class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for class_name in class_names:
                class_folder = os.path.join(base_path, class_name)
                images = load_images_from_folder(class_folder)
                if images:  # Only add classes with images
                    random.shuffle(images)
                    st.session_state.class_images[class_name] = images
            random.seed()  # Reset the random seed

        # Carousel section
        selected_class = st.selectbox("Select a class:", list(st.session_state.class_images.keys()))

        if selected_class:
            st.subheader(selected_class)
            base64_images = st.session_state.class_images[selected_class]
            
            if base64_images:
                if 'carousel_index' not in st.session_state:
                    st.session_state.carousel_index = 0

                cols = st.columns(3)
                for i in range(3):
                    idx = (st.session_state.carousel_index + i) % len(base64_images)
                    cols[i].image(f"data:image/jpeg;base64,{base64_images[idx]}", use_column_width=True)

                col1, col2, col3 = st.columns([1,2,1])
                with col1:
                    if st.button('Previous Image'):
                        st.session_state.carousel_index = (st.session_state.carousel_index - 3) % len(base64_images)
                        st.rerun()
                with col3:
                    if st.button('Next Image'):
                        st.session_state.carousel_index = (st.session_state.carousel_index + 3) % len(base64_images)
                        st.rerun()
            else:
                st.write("No images found for this class.")

# --------------------------------------------------------------------
# Section 2: Preprocessing images
elif page == "Preprocessing":
    st.header("Applying Preprocessing Steps")

    def get_random_image(base_path):
        class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        random_class = class_names[int(time.time() * 1000) % len(class_names)]
        class_folder = os.path.join(base_path, random_class)
        image_files = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random_image = image_files[int(time.time() * 1000000) % len(image_files)]
        return os.path.join(class_folder, random_image)
    
    # Define the base path of images
    base_path = "data/images_train_test_val/train"

    def preprocess_image(image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Increase sharpness
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = image.astype(np.float32) / 255.0
        return image

    # Initialize session state for preprocessing
    if 'preprocess_current_image' not in st.session_state:
        st.session_state.preprocess_current_image = cv2.imread(get_random_image(base_path))
    if 'preprocess_preprocessed_image' not in st.session_state:
        st.session_state.preprocess_preprocessed_image = None

    # Create two columns for the layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.preprocess_current_image, channels="BGR", use_column_width=True)
        
        if st.button("Load Random Image"):
            new_image_path = get_random_image(base_path)
            st.session_state.preprocess_current_image = cv2.imread(new_image_path)
            st.session_state.preprocess_preprocessed_image = None
            st.rerun()

    with col2:
        st.subheader("Preprocessed Image")
        if st.session_state.preprocess_preprocessed_image is not None:
            st.image(st.session_state.preprocess_preprocessed_image, channels="BGR", use_column_width=True)
        else:
            st.write("Press 'Preprocess Image' to see the result")
        
        if st.button("Preprocess Image"):
            st.session_state.preprocess_preprocessed_image = preprocess_image(st.session_state.preprocess_current_image)
            st.rerun()
    
# --------------------------------------------------------------------
# Section 3: Testing model with uploaded images
elif page == "Model Prediction":
    st.header("Model Prediction")

    # Function to safely load models
    @st.cache_resource
    def safe_load_model(model_path):
        try:
            return load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {str(e)}")
            return None
        
    # Load the models
    with st.spinner("Loading models... This may take a moment."):
        base_cnn_model = safe_load_model('models/best_CNN_model.keras')
        residual_net_model = safe_load_model('models/best_ResNet_model.keras')
        efficient_net_model = safe_load_model('models/best_EffNet_model.keras')

    models = {
        "Base CNN": base_cnn_model,
        "ResidualNet": residual_net_model,
    }

    models = {k: v for k, v in models.items() if v is not None} # Filter out failed models

    if not models:
        st.error("No models could be loaded. Please check your model files and paths.")
        st.stop()
    else:
        st.success(f"Successfully loaded {len(models)} model(s).")

    def preprocess_model_image(image, target_size=(224, 224)):
        if isinstance(image, np.ndarray):
            # Convert OpenCV BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array

    def predict_image(image, model):
        # Preprocess the image
        img_array = preprocess_model_image(image)

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        
        classes = ['Agricultural', 'Airplane', 'Baseball Diamond',
                'Beach', 'Buildings', 'Chaparral', 'Dense Residential',
                'Forest', 'Freeway', 'Golf Course', 'Harbor', 'Intersection',
                'Medium Residential', 'Mobile Home Park', 'Overpass',
                'Parking Lot', 'River', 'Runway', 'Sparse Residential',
                'Storage Tanks', 'Tennis Court']
        predicted_class = classes[class_index]
        confidence = prediction[0][class_index]

        return predicted_class, confidence

    uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_array = np.array(image)

        selected_model = st.selectbox("Select a model for prediction:", list(models.keys()))

        if st.button("Predict"):
            with st.spinner('Processing...'):
                predicted_class, confidence = predict_image(image_array, models[selected_model])

            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

            st.success(f"This image is classified as {predicted_class}.")

# --------------------------------------------------------------------
# Section 4 
elif page == "Model Performance":
    model_performance.display_model_performance()

# --------------------------------------------------------------------
# Section 5
elif page == "Model Visualization":
    model_visualization.display_model_visualization()

# --------------------------------------------------------------------
st.markdown("---")
st.write("Note: This app uses pre-trained models to classify satellite images into various categories.")