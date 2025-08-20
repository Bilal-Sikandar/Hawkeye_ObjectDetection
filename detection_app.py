import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Hawkeye - Object Detection", page_icon="🦅", layout="centered")

st.markdown(
    """
    <div style="text-align:center; padding: 10px;">
        <h1 style="color:purple; margin-bottom:0;">🦅 Hawkeye</h1>
        <p style="font-size:16px; color:#555; margin-top:6px;">
            Sharp Vision, Smarter Detection.<br>
            Upload an image or video — Hawkeye will detect and summarize objects in real time.
        </p>
        <hr style="border:1px solid #a64ca6; opacity:0.4;">
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Cached so the YOLO weights download happens once
    return YOLO("yolov8n.pt")

def draw_and_collect(frame, model):
    """Run YOLO on a single frame, draw boxes, and collect labels."""
    labels = []
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

            # Purple box + Yellow text
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 0, 128), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
    return frame, labels

def show_summary(labels):
    if not labels:
        st.info("No objects detected.")
        return
    counts = Counter(labels)
    st.markdown("### 🦅 Hawkeye Detected:")
    for name, cnt in sorted(counts.items()):
        st.write(f"- {name.capitalize()} ({cnt})")

# Detect if likely on Streamlit Cloud (webcam not supported)
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "0") == "1"

# ---------------------------
# Sidebar: Debug / Help
# ---------------------------
with st.sidebar:
    st.header("ℹ️ Help / Debug")
    st.write("• If the app seems stuck, it might be downloading the YOLO model (first run only).")
    st.write("• Webcam works locally. On Streamlit Cloud, use Image/Video.")
    with st.expander("Environment info"):
        import sys
        st.write(f"Python: {sys.version.split()[0]}")
        try:
            import ultralytics, cv2 as _cv2, numpy as _np
            st.write(f"Ultralytics: {ultralytics.__version__}")
            st.write(f"OpenCV: {_cv2.__version__}")
            st.write(f"NumPy: {_np.__version__}")
        except Exception as e:
            st.write("Import check error:", e)

# ---------------------------
# Load model lazily (with spinner)
# ---------------------------
with st.spinner("Loading YOLOv8 model... (first run may take a moment)"):
    model = load_model()

# ---------------------------
# UI - Input Choice
# ---------------------------
option = st.radio("Choose input type:", ["Image", "Video", "Webcam (local only)"])

# --------------------------------
# IMAGE
# --------------------------------
if option == "Image":
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up:
        file_bytes = up.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detecting objects..."):
            out, labels = draw_and_collect(img, model)

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
        show_summary(labels)

        # Download processed image
        _, buf = cv2.imencode(".jpg", out)
        st.download_button(
            "📥 Download Detected Image",
            buf.tobytes(),
            file_name="hawkeye_detected.jpg",
            mime="image/jpeg"
        )

# --------------------------------
# VIDEO
# --------------------------------
elif option == "Video":
    up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if up:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        label_box = st.empty()

        # Process every frame; for speed you can skip frames if needed
        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out, labels = draw_and_collect(frame, model)
                stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
                if labels:
                    counts = Counter(labels)
                    summary = " | ".join([f"{k.capitalize()} ({v})" for k, v in sorted(counts.items())])
                    label_box.markdown(f"**🦅 Hawkeye Detected:** {summary}")
        cap.release()

# --------------------------------
# WEBCAM (local only)
# --------------------------------
else:
    if IS_CLOUD:
        st.warning("Webcam is not supported on Streamlit Cloud. Please run locally to use this feature.")
    start = st.checkbox("Start Webcam", disabled=IS_CLOUD)
    if start and not IS_CLOUD:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        label_box = st.empty()
        st.info("Uncheck the box to stop the webcam.")

        while start and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out, labels = draw_and_collect(frame, model)
            stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
            if labels:
                counts = Counter(labels)
                summary = " | ".join([f"{k.capitalize()} ({v})" for k, v in sorted(counts.items())])
                label_box.markdown(f"**🦅 Hawkeye Detected:** {summary}")
            # Re-read checkbox state each loop to allow stopping
            start = st.checkbox("Start Webcam", value=True)
        cap.release()
