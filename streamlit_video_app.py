import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import PIL.Image
from notification_handler import NotificationHandler

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
    /* Compact primary button */
    .stButton>button {
        width: auto;
        min-width: 180px;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        margin: 8px auto 12px auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #ff3131;
        box-shadow: 0 6px 16px -6px rgba(255, 75, 75, 0.5);
    }
    /* Tighten container padding to reduce scrolling */
    .block-container {
        padding-top: 10px;
        padding-bottom: 10px;
        max-width: 1400px;
    }
    /* Video container rounded and compact */
    .element-container:has(video), .element-container:has(img) {
        margin-top: 6px;
        margin-bottom: 6px;
        border-radius: 12px;
        overflow: hidden;
    }
    /* Cap media to viewport height for visibility */
    video, img {
        max-height: 65vh;
        height: auto;
        object-fit: contain;
    }
    /* Sidebar spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 12px;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #41424C;
        text-align: center;
    }
    /* Compact metrics */
    .metric-card b { display:block; margin-bottom: 4px; }
    .metric-card { padding: 0.75rem; }
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

def banner(message: str, container, side: str = "left"):
    """Render a compact visual banner in the given container."""
    color = "#2a5298" if side == "left" else "#8a2a2a"
    container.markdown(
        f"""
        <div style="background:{color};color:#fff;padding:10px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);
                    font-family:system-ui;box-shadow:0 6px 18px -6px rgba(0,0,0,0.55);">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _laplacian_variance(img_roi: np.ndarray) -> float:
    """
    Blur score using Laplacian variance.
    Higher is sharper; low variance indicates blur.
    """
    if img_roi is None or img_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) if img_roi.ndim == 3 else img_roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _is_prohibitory_class(name: str) -> bool:
    """
    Heuristic: Treat classes containing 'PROHIBITED' or starting with 'NO_' as prohibitory.
    Prevents guessing specific prohibitory class under poor quality.
    """
    n = str(name).upper()
    return ('PROHIBITED' in n) or n.startswith('NO_')

def _roi_quality_ok(xyxy, frame_w, frame_h, frame_bgr, blur_threshold: float = 100.0) -> tuple[bool, float]:
    """
    ROI quality gate to reduce false positives in poor frames:
    - Reject tiny boxes (<1% of frame area)
    - Reject if width or height < 32 px
    - Reject if Laplacian variance < blur_threshold
    Returns (ok, blur_score)
    """
    x1, y1, x2, y2 = map(int, xyxy)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w < 32 or h < 32:
        return (False, 0.0)
    area = w * h
    if area < 0.01 * (frame_w * frame_h):  # <1% of frame
        return (False, 0.0)
    roi = frame_bgr[y1:y2, x1:x2]
    blur_score = _laplacian_variance(roi)
    if blur_score < blur_threshold:
        return (False, blur_score)
    return (True, blur_score)

def process_video(video_path_or_cam, model, conf_threshold, iou_threshold, languages, play_audio=True):
    """
    Process video/camera frames and display results with temporal consistency and quality gates.
    - Forces imgsz ≥ 1280 (maintains aspect ratio via letterbox inside YOLO)
    - Uses confidence ≥ 0.8 (from F1-confidence analysis)
    - Applies frame voting: accept a class only after ≥5 consecutive frames
    - Applies ROI quality gate to abstain on poor detections
    - Adds prohibitory family fallback to avoid guessing specific class
    """
    cap = cv2.VideoCapture(0) if video_path_or_cam == 0 else cv2.VideoCapture(video_path_or_cam)
    
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Layout: left alerts, center video, right stats (single side banner, fixed metrics)
    col_left, col_center, col_right = st.columns([1.0, 2.6, 1.0])
    st_frame = col_center.empty()
    alert_left = col_left.empty()
    metrics_container = col_right.container()
    m1 = metrics_container.empty()
    m2 = metrics_container.empty()
    m3 = metrics_container.empty()

    notifier = NotificationHandler(output_path="output")
    progress_bar = st.progress(0)
    
    # Statistics & Temporal State
    frame_count = 0
    detections_log = {}        # Class -> total detections
    class_streak = {}          # Class -> consecutive-frame count
    accepted_classes = set()   # Classes currently accepted
    accepted_count = 0
    rejected_count = 0
    stability_counter = 0      # Max streak across classes for UI
    last_alert_ts = {}         # Class -> last timestamp for debounce
    cooldown_s = 6.0
    
    stop_button = st.button("Stop Processing")
    
    while cap.isOpened():
        if stop_button:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference (force high-resolution; keep aspect ratio via YOLO internals)
        start_time = time.time()
        results = model.predict(frame, imgsz=max(1280, max(width, height)), conf=max(0.8, conf_threshold), iou=iou_threshold, verbose=False)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # Track classes seen this frame
        seen_this_frame = set()
        frame_bgr = frame if isinstance(frame, np.ndarray) else annotated_frame

        # Log detections with ROI quality and temporal voting
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            xyxy = list(map(float, box.xyxy[0].tolist()))

            detections_log[cls_name] = detections_log.get(cls_name, 0) + 1
            seen_this_frame.add(cls_name)

            # ROI quality gate
            ok, blur_score = _roi_quality_ok(xyxy, width, height, frame_bgr)
            if not ok:
                rejected_count += 1
                # Overlay 'Unclear Sign' near the box to indicate abstention
                x1, y1, _, _ = map(int, xyxy)
                cv2.putText(annotated_frame, "Unclear Sign", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 64, 255), 2)
                continue

            # Prohibitory family fallback under poor quality / low confidence
            if _is_prohibitory_class(cls_name) and (conf < 0.9):
                # Behave conservatively: do not guess specific class
                rejected_count += 1
                x1, y1, _, _ = map(int, xyxy)
                cv2.putText(annotated_frame, "Prohibitory Sign Ahead", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                # Visual alert only, avoid specific class audio to prevent wrong instruction
                banner("🚫 Prohibitory Sign Ahead", alert_left, side="left")
                continue

            # Temporal consistency: require ≥5 consecutive frames to accept
            class_streak[cls_name] = class_streak.get(cls_name, 0) + 1
            stability_counter = max(stability_counter, class_streak[cls_name])
            if class_streak[cls_name] >= 5:
                if cls_name not in accepted_classes:
                    accepted_classes.add(cls_name)
                    accepted_count += 1
                    # Debounced audio + visual alerts upon first acceptance
                    now = time.time()
                    if now - last_alert_ts.get(cls_name, 0) > cooldown_s:
                        notifier.notify_traffic_sign(cls_name, languages=languages, visual_alert=True, audio_alert=play_audio)
                        banner(f"✅ {cls_name.replace('_',' ').title()} confirmed", alert_left, side="left")
                        last_alert_ts[cls_name] = now

        # Reset streaks for classes not observed in this frame
        for k in list(class_streak.keys()):
            if k not in seen_this_frame:
                class_streak[k] = 0
                if k in accepted_classes:
                    accepted_classes.remove(k)

        # Display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Metrics
        frame_count += 1
        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(min(progress, 1.0))
        
        # Fixed metrics (update same placeholders, no stacking)
        m1.markdown(f"<div class='metric-card'><b>FPS</b><br>{max(1e-3, 1000/inference_time):.1f}</div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><b>Accepted/Rejected</b><br>{accepted_count} / {rejected_count}</div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><b>Stability (max streak)</b><br>{stability_counter}</div>", unsafe_allow_html=True)

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
        # Enforce conservative minimum confidence ≥ 0.8 (from F1-confidence analysis)
        conf_threshold = st.slider("Confidence Threshold (min 0.8)", 0.8, 1.0, 0.8, 0.01)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        play_audio = st.toggle("Play Audio Alerts", value=True)
        languages = st.multiselect(
            "Audio Languages (order matters; first tries autoplay)",
            options=["en","hi","ta","te","kn","ml","mr","gu","bn","pa"],
            default=["en","hi"],
        )
        use_camera = st.toggle("Use Live Camera", value=False)
        
        st.divider()
        st.info("System optimized for Indian Traffic Signs.")
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    # Start detection on either camera or uploaded video
    if use_camera:
        if st.button("Start Camera Detection", type="primary"):
            model = load_model()
            if model:
                process_video(0, model, conf_threshold, iou_threshold, languages, play_audio)
    elif uploaded_file is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Start Detection", type="primary"):
            model = load_model()
            if model:
                process_video(video_path, model, conf_threshold, iou_threshold, languages, play_audio)

if __name__ == "__main__":
    main()
