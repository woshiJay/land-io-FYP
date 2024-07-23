import streamlit as st
import os
from PIL import Image

def display_model_visualization():
    st.title("Model Visualization")

    # Model selection using radio buttons
    models = ["baseCNN", 
              "denseNet", 
            #   "EfficientNet", 
              "inception", 
              "ResNet", 
              "vgg", 
              "xception"]
    selected_model = st.sidebar.radio("Select Model:", models)

    # Tabbed interface for plotting and confusion matrix
    tab1, tab2 = st.tabs(["Model Plotting", "Confusion Matrix"])

    with tab1:
        plot_path = f"plotting/{selected_model}_output.png"
        load_and_display_image(plot_path)

    with tab2:
        matrix_path = f"confusion_matrix/{selected_model}_output.png"
        load_and_display_image(matrix_path)

def load_and_display_image(path):
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, use_column_width=True)
    else:
        st.error(f"Image not found: {path}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Model Visualization", layout="wide")
    display_model_visualization()

if __name__ == "__main__":
    main()