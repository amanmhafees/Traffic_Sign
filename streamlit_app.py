import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from traffic_sign_recognition import TrafficSignRecognition

# Add the current directory to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from notification_handler import NotificationHandler  # Import NotificationHandler for audio alerts
from langcodes import Language  # For converting language codes to full names

st.set_page_config(page_title="Traffic Sign Recognition (YOLOv11)", layout="centered")
st.title("🚦 Indian Traffic Sign Recognition (YOLO v11)")
st.write("Upload an image to detect Indian traffic signs using your trained YOLO v11 model.")

# Sidebar for model selection and settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input(
    "Path to trained model (.pt)",
    value="output/traffic_sign_model/weights/best.pt"
)
conf_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05
)

# Language selection for audio alerts
notification_handler = NotificationHandler()
available_languages = notification_handler.available_languages
selected_languages = st.sidebar.multiselect(
    "Select languages for audio alerts",
    options=available_languages,
    default=["en", "hi"],  # Default to English and Hindi
    format_func=lambda lang: Language.get(lang).display_name()  # Convert language code to full name
)

# Load model (cache for performance)
@st.cache_resource(show_spinner=True)
def load_tsr(model_path):
    tsr = TrafficSignRecognition()
    tsr.load_model(model_path)
    return tsr

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Load model
    try:
        tsr = load_tsr(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Run detection
    with st.spinner("Detecting traffic signs..."):
        detections = tsr.process_frame(img_bgr, conf_threshold=conf_threshold)
        result_img = tsr.draw_detections(img_bgr, detections)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Show results
    st.subheader("Detection Results")
    st.image(result_img_rgb, caption="Detected Traffic Signs", use_container_width=True)

    # Show detection details and play audio alerts
    if detections:
        st.markdown("**Detected Signs:**")
        for i, (cls_name, conf_score, bbox, area) in enumerate(detections):
            x1, y1, x2, y2 = bbox
            st.write(f"{i+1}. **{cls_name}** (Confidence: {conf_score:.2f}) [Box: ({x1},{y1})-({x2},{y2})]")
            
            # Display a temporary notification using a placeholder
            notification_placeholder = st.empty()
            notification_placeholder.success(f"🔔 Detected: {cls_name}")
            
            # Close the notification after 5 seconds
            import time
            time.sleep(5)
            notification_placeholder.empty()

            # Play audio alert for detected sign
            notification_handler.notify_traffic_sign(cls_name, selected_languages)
    else:
        st.info("No traffic signs detected with the current confidence threshold.")
else:
    st.info("Please upload an image to begin.")

st.markdown("---")
st.caption("Built with Streamlit · Powered by YOLO v11 · Indian Traffic Sign Recognition")