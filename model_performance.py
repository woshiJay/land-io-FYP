import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Function to safely load and compile models
@st.cache_resource
def safe_load_model(model_path):
    if model_path not in st.session_state:
        try:
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            st.session_state[model_path] = model
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    return st.session_state[model_path]

# Function to evaluate model
def evaluate_model(model, data_generator):
    return model.evaluate(data_generator, verbose=0)

# Function to evaluate model
def evaluate_model(model, data_generator):
    return model.evaluate(data_generator, verbose=0)

def display_model_performance():
    st.subheader("Model Performance")

    # Check if models are already loaded
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    if not st.session_state.models_loaded:
        progress_bar = st.progress(0)
        status_text = st.empty()

        models = {}
        model_names = ["Base CNN", "DenseNet", "EfficientNet", "InceptionNet", "ResidualNet", "VGGNet", "XceptionNet"]
        model_paths = [
            'models/best_CNN_model.keras',
            'models/best_DenseNet_model.keras',
            'models/best_EffNet_model.keras',
            'models/best_Inception_model.keras',
            'models/best_ResNet_model.keras',
            'models/best_VIT_model.keras',
            'models/best_xception_model.keras'
        ]

        for i, (name, path) in enumerate(zip(model_names, model_paths)):
            status_text.text(f"Loading {name}...")
            models[name] = safe_load_model(path)
            progress = (i + 1) / len(model_names)
            progress_bar.progress(progress)
            time.sleep(0.1)  # Small delay to show progress

        st.session_state.models = models
        st.session_state.models_loaded = True
        
        status_text.text("All models loaded!")
        time.sleep(1)  # Show the "All models loaded!" message for a second
        status_text.empty()
        progress_bar.empty()

    else:
        models = st.session_state.models

    # Filter out failed models
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        st.error("No models could be loaded. Please check your model files and paths.")
        return
    else:
        st.success(f"Successfully loaded and compiled {len(models)} model(s).")

    # Prepare data generator for performance
    val_data_path = "data/images_train_test_val/validation"
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate models
    results = {}
    for model_name, model in models.items():
        with st.spinner(f'Evaluating {model_name}...'):
            loss, accuracy = evaluate_model(model, val_generator)
            results[model_name] = {'Test Loss': loss, 'Test Accuracy': accuracy}

    # Create DataFrame from results
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_values('Test Accuracy', ascending=False)
    df = df.reset_index().rename(columns={'index': 'Model'})

    # Create the plot
    fig = go.Figure()

    # Add Test Accuracy bars
    fig.add_trace(go.Bar(
        x=df['Model'],
        y=df['Test Accuracy'],
        name='Test Accuracy',
        marker_color='rgb(26, 118, 255)'
    ))

    # Add Test Loss line
    fig.add_trace(go.Scatter(
        x=df['Model'],
        y=df['Test Loss'],
        name='Test Loss',
        yaxis='y2',
        line=dict(color='rgb(219, 64, 82)', width=2)
    ))

    # Update layout
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model Type',
        yaxis_title='Accuracy',
        yaxis2=dict(
            title='Loss',
            overlaying='y',
            side='right'
        ),
        barmode='group',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        template='plotly_white'
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Display the data table
    st.write("Model Performance Data:")
    st.dataframe(df)