import streamlit as st
from PIL import Image
import pandas as pd
import io

# Import our YOLO helper class
from src.yolo_helper import YolosHelper

# Configure the Streamlit page
st.set_page_config(
    page_title="YOLO Object Detection & Segmentation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model():
    """Load the YOLO model once and cache it."""
    # This assumes yolov8n-seg.pt will be downloaded automatically by ultralytics
    # if it doesn't exist in the current directory.
    return YolosHelper(model_path="yolov8n-seg.pt")

def main():
    st.title("🔍 YOLO Object Detection & Segmentation")
    st.markdown("""
    Upload an image to perform object detection and segmentation.
    The application will outline objects, draw segmentation masks, and list the detected items.
    """)

    # Initialize model
    with st.spinner("Loading YOLO model..."):
        yolo_helper = load_yolo_model()

    st.sidebar.title("Settings")
    st.sidebar.markdown("Upload your image here.")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.header("Segmented Output")
            # Run inference
            with st.spinner("Processing image..."):
                try:
                    annotated_image, detected_counts = yolo_helper.predict_and_annotate(image)
                    st.image(annotated_image, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    return
        
        # Display detection statistics
        st.subheader("Detected Objects")
        if detected_counts:
            # Create a DataFrame for nice tabular display
            df = pd.DataFrame(
                list(detected_counts.items()), 
                columns=["Object Name", "Count"]
            )
            # Sort by count descending
            df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
            
            # Display metrics in columns
            st.markdown("### Summary")
            metric_cols = st.columns(min(len(df), 4))
            for i, row in df.head(4).iterrows():
                with metric_cols[i]:
                    st.metric(label=row["Object Name"].capitalize(), value=row["Count"])
            
            st.markdown("### Detail Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected in the image.")

if __name__ == "__main__":
    main()
