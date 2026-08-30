import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.efficientnet import preprocess_input
import streamlit as st
import av

# Try importing streamlit_webrtc
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase,
    )
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# ==========================================
# CONFIGURABLE MODEL PATHS AND CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_DETECTION_PATH = os.path.join(BASE_DIR, "best.pt")
ASL_MODEL_PATH = os.path.join(BASE_DIR, "best_asl_model.keras")

CLASSIFIER_INPUT_SIZE = (224, 224)
CROP_PADDING = 20
CLASS_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
    "del",
    "nothing",
    "space",
]

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner="Loading YOLOv8 Hand Detector...")
def load_hand_detector(path: str):
    if not os.path.exists(path):
        st.error(f"Hand detection model file not found at: {path}")
        st.stop()
    return YOLO(path)


@st.cache_resource(show_spinner="Loading EfficientNetB4 ASL Classifier (354 MB)...")
def load_asl_classifier(path: str):
    if not os.path.exists(path):
        st.error(
            f"ASL classifier model file not found at: {path}. "
            "If deployed on Streamlit Cloud, ensure Git LFS pulled the full weight file."
        )
        st.stop()
    return tf.keras.models.load_model(path)


# ==========================================
# PIPELINE UTILITY FUNCTIONS
# Preserve exact logic from reference pipeline
# ==========================================
def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, CLASSIFIER_INPUT_SIZE)
    crop_float = crop_resized.astype("float32")
    crop_preprocessed = preprocess_input(crop_float)
    batch = np.expand_dims(crop_preprocessed, axis=0)
    return batch


def classify_hand(asl_classifier, crop: np.ndarray):
    batch = preprocess_crop(crop)
    preds = asl_classifier.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else "?"
    return label, confidence


def draw_prediction(frame: np.ndarray, box: tuple, label: str, confidence: float):
    x1, y1, x2, y2 = box
    color = (0, 200, 0) if confidence >= 0.6 else (0, 165, 255)
    text = f"{label} ({confidence * 100:.0f}%)" if confidence >= 0.6 else "?"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


def process_frame(frame: np.ndarray, hand_detector, asl_classifier, conf_threshold=0.6):
    frame_h, frame_w = frame.shape[:2]
    results = hand_detector.predict(frame, conf=conf_threshold, verbose=False)[0]

    latest_label = None
    latest_conf = 0.0
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(frame_w, x2 + CROP_PADDING)
        y2 = min(frame_h, y2 + CROP_PADDING)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, confidence = classify_hand(asl_classifier, crop)
        draw_prediction(frame, (x1, y1, x2, y2), label, confidence)

        if confidence > latest_conf:
            latest_label = label
            latest_conf = confidence

        detections.append((label, confidence, (x1, y1, x2, y2)))

    return frame, latest_label, latest_conf, detections


# ==========================================
# WEBRTC PROCESSOR CLASS
# ==========================================
if HAS_WEBRTC:
    class ASLVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.hand_detector = load_hand_detector(HAND_DETECTION_PATH)
            self.asl_classifier = load_asl_classifier(ASL_MODEL_PATH)
            self.latest_label = "None"
            self.latest_confidence = 0.0
            self.hands_count = 0
            self.conf_threshold = 0.6

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            processed_img, label, conf, detections = process_frame(
                img, self.hand_detector, self.asl_classifier, self.conf_threshold
            )
            self.hands_count = len(detections)
            if label is not None:
                self.latest_label = label
                self.latest_confidence = conf
            else:
                self.latest_label = "No Hand Detected"
                self.latest_confidence = 0.0

            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# ==========================================
# MAIN APPLICATION APP
# ==========================================
def main():
    st.markdown('<div class="main-title">🤟 ASL Sign Language Detection Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time American Sign Language recognition using YOLOv8 hand detection & EfficientNetB4 classification</div>',
        unsafe_allow_html=True,
    )

    # Sidebar setup
    st.sidebar.title("⚙️ Controls & Options")
    input_mode = st.sidebar.radio(
        "Select Input Mode:",
        options=["📷 Live Webcam", "📁 Upload Video File"],
        index=0,
    )

    conf_threshold = st.sidebar.slider(
        "YOLO Hand Detection Confidence:",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Model Specifications")
    st.sidebar.markdown("- **Hand Detector**: YOLOv8 (`best.pt`)")
    st.sidebar.markdown("- **Classifier**: EfficientNetB4 (`best_asl_model.keras`)")
    st.sidebar.markdown("- **Input Resolution**: 224x224")
    st.sidebar.markdown("- **Padding**: 20 px")
    st.sidebar.markdown("- **Classes**: 29 (A-Z, del, nothing, space)")

    # Load models
    hand_detector = load_hand_detector(HAND_DETECTION_PATH)
    asl_classifier = load_asl_classifier(ASL_MODEL_PATH)

    # Display Top Metrics Container
    col1, col2, col3 = st.columns(3)
    metric_label_placeholder = col1.empty()
    metric_conf_placeholder = col2.empty()
    metric_count_placeholder = col3.empty()

    def update_metrics(label="None", conf=0.0, count=0):
        metric_label_placeholder.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted ASL Sign</div>
                <div class="metric-value">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        conf_text = f"{conf * 100:.1f}%" if conf > 0 else "0.0%"
        metric_conf_placeholder.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Classification Confidence</div>
                <div class="metric-value">{conf_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_count_placeholder.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Hands Detected</div>
                <div class="metric-value">{count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    update_metrics("Ready", 0.0, 0)
    st.markdown("<br>", unsafe_allow_html=True)

    # Mode 1: Live Webcam
    if input_mode == "📷 Live Webcam":
        if not HAS_WEBRTC:
            st.error("streamlit-webrtc package is not installed. Please install it using `pip install streamlit-webrtc`.")
            return

        st.subheader("🎥 Live Webcam Feed")
        st.info("Allow camera access in your browser. Live detection will annotate hands and classify ASL signs in real-time.")

        ctx = webrtc_streamer(
            key="asl-detection-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=ASLVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.conf_threshold = conf_threshold
            label = ctx.video_processor.latest_label
            conf = ctx.video_processor.latest_confidence
            count = ctx.video_processor.hands_count
            update_metrics(label, conf, count)

    # Mode 2: Upload Video File
    elif input_mode == "📁 Upload Video File":
        st.subheader("📼 Video File Processing")
        uploaded_file = st.file_uploader(
            "Choose a video file (.mp4, .mov, .avi):", type=["mp4", "mov", "avi"]
        )

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            st.write(f"Processing video: `{uploaded_file.name}` ({total_frames} frames, {fps:.1f} FPS)")

            stop_btn = st.button("⏹️ Stop Processing")
            st_frame = st.empty()

            frame_idx = 0
            while cap.isOpened():
                if stop_btn:
                    st.warning("Video processing stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                processed_frame, label, conf, detections = process_frame(
                    frame, hand_detector, asl_classifier, conf_threshold
                )

                # Convert BGR to RGB for Streamlit rendering
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                display_label = label if label else "No Hand"
                update_metrics(display_label, conf, len(detections))

            cap.release()
            try:
                os.unlink(tfile.name)
            except Exception:
                pass


if __name__ == "__main__":
    main()
