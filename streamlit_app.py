import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from traffic_sign_recognition import TrafficSignRecognition
import time
import hashlib

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
max_display_height = st.sidebar.number_input(
    "Max display image height (px)",
    min_value=200, max_value=1200, value=500, step=50
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
# Notification toggles
enable_audio = st.sidebar.checkbox("Enable Audio Alerts", value=True)
enable_visual = st.sidebar.checkbox("Enable Visual Alerts", value=True)

# Session state for preventing repeat audio per sign
if "last_upload_hash" not in st.session_state:
    st.session_state["last_upload_hash"] = None
if "announced_signs" not in st.session_state:
    st.session_state["announced_signs"] = set()

# Cleanup old custom notification placeholders (time-based)
now_ts = time.time() if "time" in dir() else __import__("time").time()
for k in list(st.session_state.keys()):
    if k.endswith("_ts"):
        if now_ts - st.session_state[k] > 3.5:
            del st.session_state[k]

# Load model (cache for performance)
@st.cache_resource(show_spinner=True)
def load_tsr(model_path):
    tsr = TrafficSignRecognition()
    tsr.load_model(model_path)
    return tsr

# Image/Video upload
uploaded_file = st.file_uploader(
    "Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.type.split('/')[0]
    
    # Compute hash of current upload
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state["last_upload_hash"] != current_hash:
        # New upload => reset
        st.session_state["announced_signs"].clear()
        st.session_state["last_upload_hash"] = current_hash

    # Load model
    try:
        tsr = load_tsr(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    if file_type == "image":
        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run detection
        with st.spinner("Detecting traffic signs..."):
            detections = tsr.process_frame(img_bgr, conf_threshold=conf_threshold)
            result_img = tsr.draw_detections(img_bgr, detections)
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Show results (Compact Image View)
        st.subheader("Detection Results")
        
        col1, col2 = st.columns(2)
        with col1:
             st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
             st.image(result_img_rgb, caption="Detected Signs", use_container_width=True)

        # Show detection details and play audio alerts
        if detections:
            st.markdown("**Detected Signs:**")
            for i, (cls_name, conf_score, bbox, area) in enumerate(detections):
                x1, y1, x2, y2 = bbox
                st.write(f"{i+1}. **{cls_name}** (Confidence: {conf_score:.2f}) [Box: ({x1},{y1})-({x2},{y2})]")
                notification_handler.notify_traffic_sign(
                    cls_name, 
                    selected_languages, 
                    visual_alert=enable_visual, 
                    audio_alert=enable_audio
                )
        else:
            st.info("No traffic signs detected with the current confidence threshold.")
            
    elif file_type == "video" or uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
        # Save temp input file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"Processing Video: {total_frames} frames @ {fps} fps ({width}x{height})")
        
        # Output temp file
        outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out_path = outfile.name
        outfile.close()
        
        # Use mp4v codec for temp storage
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detection
            detections = tsr.process_frame(frame, conf_threshold=conf_threshold)
            result_frame = tsr.draw_detections(frame, detections)
            
            # Write
            out.write(result_frame)
            
            # Update progress
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
            # Optional: Real-time visual/audio alerts (throttled) might be too heavy here.
            # We skip audio/visual alerts per-frame to avoid UI flooding.
            
        cap.release()
        out.release()
        status_text.text("Processing Complete!")
        
        # Re-encode for browser compatibility (using ffmpeg if available, otherwise raw mp4)
        # Browsers often struggle with raw OpenCV mp4v. 
        # For now, we display side-by-side.
        
        st.subheader("Video Results (Compact View)")
        vcol1, vcol2 = st.columns(2)
        
        with vcol1:
            st.caption("Original Video")
            st.video(tfile.name)
            
        with vcol2:
            st.caption("Processed Video (with Detections)")
            # Note: If codec issues occur, we might need to convert. 
            # Trying to read back the manually written file.
            if os.path.exists(out_path):
                 st.video(out_path)
            else:
                 st.error("Error creating output video.")

        # Cleanup
        os.unlink(tfile.name)
        # os.unlink(out_path) # Keep output for viewing for now (stream cleanup handles eventually)

else:
    st.info("Please upload an image or video to begin.")

st.markdown("---")
st.caption("Built with Streamlit · Powered by YOLO v11 · Indian Traffic Sign Recognition")