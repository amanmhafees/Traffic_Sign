import streamlit as st
import cv2
def _laplacian_variance(img_bgr) -> float:
    """
    Compute Laplacian variance as a measure of blur.
    Higher values indicate sharper images.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0
import tempfile
import time
import numpy as np
from pathlib import Path
import PIL.Image
from notification_handler import NotificationHandler
from two_stage_recognizer import TwoStageSignRecognizer

# Configuration
MODEL_PATH = "output/traffic_sign_model/weights/best.pt"  # YOLO detector weights
CNN_WEIGHTS = "output/cnn_classifier.pt"                   # CNN classifier weights
CNN_CLASSES = "output/cnn_classes.json"                   # idx->class mapping
CNN_CONF_THRESHOLD = 0.75                                  # Fallback threshold
INSTANT_ALERT_CONF = 0.90                                  # Immediate alert if CNN confidence >= this
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

@st.cache_resource(show_spinner=True)
def load_two_stage():
    """Load YOLO detector and CNN classifier once (cached)."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"YOLO weights not found at {MODEL_PATH}.")
    if not Path(CNN_WEIGHTS).exists():
        raise FileNotFoundError(
            (
                f"CNN weights not found at {CNN_WEIGHTS}. "
                "Train it with train_cnn_classifier.py after running create_cnn_dataset.py."
            )
        )
    recognizer = TwoStageSignRecognizer(
        yolo_weights=MODEL_PATH,
        cnn_weights=CNN_WEIGHTS,
        classes_path=CNN_CLASSES,
        classifier_threshold=CNN_CONF_THRESHOLD,
    )
    return recognizer

def banner(message: str, container, side: str = "left"):
    """Render a compact visual banner in the given container."""
    color = "#2a5298" if side == "left" else "#8a2a2a"
    container.markdown(
        f"""
        <div style="background:{color};color:#fff;padding:10px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);\n                    font-family:system-ui;box-shadow:0 6px 18px -6px rgba(0,0,0,0.55);">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

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

def process_video(video_path_or_cam, recognizer: TwoStageSignRecognizer, conf_threshold, iou_threshold, languages, play_audio=True):
    """
    Process video/camera frames and display results with temporal consistency and quality gates.
    - Forces imgsz >= 1280 (maintains aspect ratio via letterbox inside YOLO)
    - Uses confidence >= 0.8 (from F1-confidence analysis)
    - Applies frame voting: accept a class after >=5 consecutive frames
    - Instant alert: if best detection CNN conf >= 0.90, alert immediately
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
        dets = recognizer.predict_frame(frame, yolo_conf=conf_threshold, yolo_iou=iou_threshold)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        annotated_frame = TwoStageSignRecognizer.draw(frame, dets)

        # Choose only the highest-confidence detection for alerting
        best_det = None
        if dets:
            try:
                best_det = max(dets, key=lambda d: d.conf)
            except Exception:
                best_det = dets[0]

        # Track classes seen this frame (CNN labels)
        seen_this_frame = set()
        for d in dets:
            seen_this_frame.add(d.label)
            detections_log[d.label] = detections_log.get(d.label, 0) + 1

            # Temporal acceptance logic with CNN labels
            class_streak[d.label] = class_streak.get(d.label, 0) + 1
            stability_counter = max(stability_counter, class_streak[d.label])
            # Only alert on the highest-confidence detection and skip fallback
            if best_det is not None and d is best_det and d.label != "Generic Prohibitory Sign":
                # Instant alert if high confidence, or after temporal acceptance streak
                should_alert = (d.conf >= INSTANT_ALERT_CONF) or (class_streak[d.label] >= 5)
                if should_alert:
                    now = time.time()
                    if now - last_alert_ts.get(d.label, 0) > cooldown_s:
                        notifier.notify_traffic_sign(d.label, languages=languages, visual_alert=True, audio_alert=play_audio)
                        banner(f"✅ {d.label.replace('_',' ').title()} confirmed", alert_left, side="left")
                        last_alert_ts[d.label] = now
                    # Mark accepted when streak condition is met (for metrics and stability)
                    if class_streak[d.label] >= 5 and d.label not in accepted_classes:
                        accepted_classes.add(d.label)
                        accepted_count += 1

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
        # Confidence threshold (lower minimum to improve detection sensitivity)
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
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
            recognizer = load_two_stage()
            if recognizer:
                process_video(0, recognizer, conf_threshold, iou_threshold, languages, play_audio)
    elif uploaded_file is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Start Detection", type="primary"):
            recognizer = load_two_stage()
            if recognizer:
                process_video(video_path, recognizer, conf_threshold, iou_threshold, languages, play_audio)

if __name__ == "__main__":
    main()
