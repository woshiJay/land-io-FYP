import streamlit as st
import os
from PIL import Image
import inspect
from models import MODEL_FUNCTIONS

def load_and_display_image(path):
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, use_column_width=True)
    else:
        st.write("This model does not have any plot.")

def get_model_code(model):
    if model in MODEL_FUNCTIONS:
        return inspect.getsource(MODEL_FUNCTIONS[model])
    return f"# Code for {model} not available"

def display_model_details(model):
    st.subheader(f"Model Selected: {model}")
    
    # Description of model (placeholder)
    st.write(f"This is a placeholder description for {model}.")
    
    # Code snippet
    st.subheader("Code Snippet")
    st.code(get_model_code(model), language="python")

    # Plotting and confusion matrix
    tab1, tab2 = st.tabs(["Model Plotting", "Confusion Matrix"])
    
    with tab1:
        plot_path = f"plotting/{model}_output.png"
        load_and_display_image(plot_path)
    
    with tab2:
        matrix_path = f"confusion_matrix/{model}_output.png"
        load_and_display_image(matrix_path)
    
    # Performance data (placeholder)
    st.subheader("Performance Data")
    st.write(f"Loss: 0.X\nAccuracy: 0.Y")

def display_model_visualization():
    st.header("Performance and Visualization")
    
    # Sidebar
    st.sidebar.header("Model Performance & Visualization")
    
    # Step 1: Choose Model Type
    model_type = st.sidebar.selectbox("Choose Model Type", ["Base Model", "Pretrain Model", "Hybrid Model"])
    
    # Step 2: Load Model Option
    if model_type:
        if model_type == "Base Model":
            models = ["BaseCNN"]
        elif model_type == "Pretrain Model":
            models = ["DenseNet", "Inception", "ResNet", "VGG", "Xception"]
        else:  # Hybrid Model
            models = ["Hybridv1", "EnsembleV1"]
        
        selected_model = st.sidebar.selectbox("Load model option", models)
    else:
        selected_model = ""
    
    # Show Results button
    show_results = st.sidebar.button("Show Results")
    
    # Main content area
    if not model_type or not selected_model:
        st.write("Waiting user selection...")
    elif show_results:
        display_model_details(selected_model)
    else:
        st.write(f"Model Selected: {selected_model}")
        st.write("Click 'Show Results' to view model details.")

if __name__ == "__main__":
    display_model_visualization()