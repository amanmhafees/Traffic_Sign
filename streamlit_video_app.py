import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path
from ultralytics import YOLO
import PIL.Image

# Configuration
MODEL_PATH = "output/traffic_sign_model/weights/best.pt"
LOGO_PATH = "assets/logo.png"  # Placeholder if you have one, or remove

# Page Setup
st.set_page_config(
    page_title="Traffic Sign Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF0000;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #41424C;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load model with caching"""
    if 'model' not in st.session_state:
        try:
            model_path_obj = Path(MODEL_PATH)
            if not model_path_obj.exists():
                st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
                return None
            st.session_state['model'] = YOLO(MODEL_PATH)
            st.toast("Model loaded successfully!", icon="✅")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return st.session_state['model']

def process_video(video_path, model, conf_threshold, iou_threshold):
    """
    Process video frame by frame and display results.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st_frame = st.empty()
    st_metrix = st.empty()
    progress_bar = st.progress(0)
    
    # Statistics
    frame_count = 0
    detections_log = {} # Class -> Count
    
    stop_button = st.button("Stop Processing")
    
    while cap.isOpened():
        if stop_button:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        start_time = time.time()
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # Log detections
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            detections_log[cls_name] = detections_log.get(cls_name, 0) + 1

        # Display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Metrics
        frame_count += 1
        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(min(progress, 1.0))
        
        with st_metrix.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><b>FPS</b><br>{1000/inference_time:.1f}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><b>Detections</b><br>{sum(detections_log.values())}</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><b>Unique Signs</b><br>{len(detections_log)}</div>", unsafe_allow_html=True)

    cap.release()
    st.success("Video processing completed!")
    
    # Final Summary
    st.subheader("Detection Summary")
    st.bar_chart(detections_log)

def main():
    st.title("🚦 AI Traffic Sign Recognition")
    st.markdown("Upload a dashcam video to detect Cautionary and Mandatory traffic signs in real-time.")
    
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.divider()
        st.info("System optimized for Indian Traffic Signs.")
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Start Detection", type="primary"):
            model = load_model()
            if model:
                process_video(video_path, model, conf_threshold, iou_threshold)

if __name__ == "__main__":
    main()
