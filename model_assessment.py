import streamlit as st
import pandas as pd
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

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

# Function to evaluate ensemble model
def evaluate_ensemble_model(xception_model, vgg16_model, data_generator):
    # This is a placeholder implementation. Adjust according to your actual ensemble method.
    xception_pred = xception_model.predict(data_generator)
    vgg16_pred = vgg16_model.predict(data_generator)
    ensemble_pred = (xception_pred + vgg16_pred) / 2
    
    true_labels = data_generator.classes
    ensemble_accuracy = (ensemble_pred.argmax(axis=1) == true_labels).mean()
    
    # For simplicity, we're using accuracy as negative loss here
    # You might want to implement a proper loss calculation
    return -ensemble_accuracy, ensemble_accuracy

# Function to load precomputed results
def load_precomputed_results():
    try:
        with open('precomputed_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Function to save precomputed results
def save_precomputed_results(results):
    with open('precomputed_results.json', 'w') as f:
        json.dump(results, f)

# Function to create the overview plot
def create_overview_plot(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Test Accuracy'], name="Test Accuracy", marker_color='rgb(26, 118, 255)'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df['Model'], y=df['Test Loss'], name="Test Loss", line=dict(color='rgb(219, 64, 82)', width=2)),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Model Assessment Comparison",
        xaxis_title="Model Type",
        yaxis_title="Accuracy",
        yaxis2=dict(title="Loss", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        template='plotly_white'
    )

    return fig

# Function to create top 3 models plot
def create_top3_plot(df):
    top3 = df.nlargest(3, 'Test Accuracy')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=top3['Model'], y=top3['Test Accuracy'], name="Test Accuracy", marker_color='rgb(26, 118, 255)'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=top3['Model'], y=top3['Test Loss'], name="Test Loss", marker_color='rgb(219, 64, 82)'),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Top 3 Models Assessment",
        xaxis_title="Model Type",
        yaxis_title="Accuracy",
        yaxis2_title="Loss",
        template='plotly_white'
    )

    return fig

def display_model_assessment():
    st.header("Model Assessment")

    if 'results' not in st.session_state or st.sidebar.button('Reload'):
        # Check for precomputed results
        results = load_precomputed_results()
        
        if results is None:
            st.warning("No precomputed results found. Running model evaluation...")
            
            # Load and evaluate models
            models = {}
            model_names = [
                "Base CNN", 
                "DenseNet121", 
                "InceptionNet",
                "ResidualNet50",
                "VGG16Net", 
                "XceptionNet",
                "DenseNet121 + RNN (HybridV1)",
                "Xception + VGG16 (Ensemble)"
            ]
            model_paths = [
                'models/best_CNN_model.keras',
                'models/best_DenseNet_model.keras',
                'models/best_Inception_model.keras',
                'models/best_ResNet_model.keras',
                'models/best_VIT_model.keras',
                'models/best_xception_model.keras',
                'models/best_HybridV1_model.keras',
            ]

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, path) in enumerate(zip(model_names[:-1], model_paths)):  # Exclude Ensemble from this loop
                status_text.text(f"Loading {name}...")
                models[name] = safe_load_model(path)
                progress = (i + 1) / len(model_names)
                progress_bar.progress(progress)
                time.sleep(0.1)  # Small delay to show progress

            # Prepare data generator for assessment
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
                if model is not None:
                    with st.spinner(f'Evaluating {model_name}...'):
                        loss, accuracy = evaluate_model(model, val_generator)
                        results[model_name] = {'Test Loss': loss, 'Test Accuracy': accuracy}

            # Evaluate Ensemble model
            with st.spinner('Evaluating Ensemble model...'):
                ensemble_loss, ensemble_accuracy = evaluate_ensemble_model(
                    models['XceptionNet'], models['VGG16Net'], val_generator
                )
                results['Ensemble (Xception+VGG16)'] = {
                    'Test Loss': ensemble_loss,
                    'Test Accuracy': ensemble_accuracy
                }

            # Save precomputed results
            save_precomputed_results(results)
            
            status_text.text("Evaluation complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

        st.session_state.results = results
        st.success("Data loaded successfully!")

    results = st.session_state.results
    df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    df = df.sort_values('Test Accuracy', ascending=False)

    # Overview section
    st.subheader("Overview")
    st.write("This overview shows the test accuracy and loss for each model.")
    st.plotly_chart(create_overview_plot(df), use_container_width=True)

    st.markdown("---")

    # Performance Table section
    st.subheader("Performance Table")
    st.write("The table below shows the performance of each model on the test dataset.")
    st.dataframe(df)

    st.markdown("---")

    # Top 3 Models section
    st.subheader("Top 3 Models")
    st.write("This section highlights the top 3 models based on test accuracy and loss.")
    st.plotly_chart(create_top3_plot(df), use_container_width=True)

if __name__ == "__main__":
    display_model_assessment()