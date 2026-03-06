import streamlit as st
import cv2
import numpy as np
import torch
import pyttsx3
import threading
import time

# ---------- Page Settings ----------
st.set_page_config(
    page_title="Blind Navigation",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS Styling ----------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f0f8ff, #e6ffe6);
}
h1 {
    color: #2E8B57;
    font-family: 'Segoe UI', sans-serif;
}
h3 {
    color: #555555;
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button {
    background-color: #2E8B57;
    color: white;
    border-radius: 8px;
    height: 40px;
    width: 120px;
}
.stSlider>div>div>input {
    color: #2E8B57;
}
.card {
    background-color: white;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px #aaaaaa;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("Settings")
run_camera = st.sidebar.checkbox("Start Camera", value=False)
voice_enabled = st.sidebar.checkbox("Enable Voice Guidance", value=True)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
voice_interval = st.sidebar.slider("Voice Repeat Interval (seconds)", 1, 10, 3)

# ---------- Header ----------
st.markdown("""
    <h1 style='text-align: center;'>🦯 Blind Navigation</h1>
    <h3 style='text-align: center;'>Real-time object detection with voice guidance</h3>
""", unsafe_allow_html=True)

# ---------- Initialize Voice Engine ----------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_objects(objects):
    for obj in objects:
        engine.say(f"{obj} ahead")
    engine.runAndWait()

# ---------- Placeholders ----------
frame_placeholder = st.empty()
object_list_placeholder = st.empty()

# ---------- Load YOLOv5 model ----------
@st.cache_resource
def load_model(path="yolov5s.pt"):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model.conf = confidence_threshold
    return model

model = load_model()

# ---------- Live Camera ----------
if run_camera:
    st.success("Camera started! Point your camera to detect objects.")
    cap = cv2.VideoCapture(0)
    last_voice_time = 0

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update model confidence dynamically
        model.conf = confidence_threshold

        # YOLO detection
        results = model(frame_rgb)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        detected_objects = []

        for i in range(len(labels)):
            row = cords[i]
            x1, y1, x2, y2, conf = int(row[0]*frame.shape[1]), int(row[1]*frame.shape[0]), int(row[2]*frame.shape[1]), int(row[3]*frame.shape[0]), row[4]
            class_name = model.names[int(labels[i])]
            detected_objects.append(class_name)

            # Draw rectangle and label
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (46, 139, 87), 2)
            cv2.putText(frame_rgb, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 139, 87), 2)

        # Display camera feed
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Display detected objects as cards
        object_cards = ""
        unique_objects = list(set(detected_objects))
        for obj in unique_objects:
            count = detected_objects.count(obj)
            object_cards += f"<div class='card'><b>{obj}</b> : {count}</div>"
        object_list_placeholder.markdown(object_cards, unsafe_allow_html=True)

        # Voice guidance for all objects every 'voice_interval' seconds
        current_time = time.time()
        if voice_enabled and detected_objects and (current_time - last_voice_time > voice_interval):
            threading.Thread(target=speak_objects, args=(detected_objects,), daemon=True).start()
            last_voice_time = current_time

        time.sleep(0.03)

    cap.release()
    st.success("Camera stopped.")
else:
    st.info("Enable 'Start Camera' from the sidebar to begin detection.")