# app.py
import os, time, queue
from PIL import Image
import cv2
import torch
import xml.etree.ElementTree as ET
import streamlit as st
from ultralytics import YOLO
import timm, torch.nn as nn
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# === CONFIG STREAMLIT ===
st.set_page_config(page_title="YOLO + ConViT Demo", layout="wide")
st.title("Demo: YOLOv8 + ConViT trên Webcam (3s Auto-Capture)")

# Auto refresh mỗi 3 giây
st_autorefresh(interval=2800, limit=None, key="autorefresh")

# === PATHS & SETTINGS ===
YOLO_WEIGHTS   = "best_yolo.pt"   # đường dẫn đến checkpoint YOLO
CONVIT_WEIGHTS = "best_convit_stanford40.pth"
LABEL_PATH     = "labels.txt"
IMG_SIZE       = 224
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD CLASS NAMES CHO CONVIT ===
@st.cache_data
def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

CLASS_NAMES = load_labels(LABEL_PATH)

# === LOAD MODELS ===
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

NUM_CLASSES = len(CLASS_NAMES)

# === LOAD MODEL ===
@st.cache_resource(show_spinner=False)
def load_model(weights_path, num_classes):
    if not os.path.exists(weights_path):
        st.error(f"Không tìm thấy model: {weights_path}")
        st.stop()
    m = timm.create_model("convit_base", pretrained=False)
    in_f = m.head.in_features
    m.head = nn.Linear(in_f, num_classes)
    sd = torch.load(weights_path, map_location=DEVICE)
    m.load_state_dict(sd)
    m.to(DEVICE).eval()
    return m

convit_model = load_model(CONVIT_WEIGHTS, NUM_CLASSES)
yolo_model  = load_yolo(YOLO_WEIGHTS)

# === PREPROCESS CHO CONVIT ===
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

st.info("Done")

def predict_action(crop: Image.Image):
    t = preprocess(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = convit_model(t)
        p = torch.softmax(out, dim=1)
        conf, idx = torch.max(p, 1)
    return idx.item(), conf.item()

# === SESSION_STATE CHO LỊCH SỬ ẢNH ===
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of (PIL.Image, label, conf)

# === VIDEO PROCESSOR ===
class Detector(VideoProcessorBase):
    def __init__(self):
        self.last = time.time()
        self.interval = 3.0
        self.queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        # Chỉ xử lý khi đủ 3s
        if now - self.last >= self.interval:
            # 1) Detections
            results = yolo_model.predict(img, imgsz=640, conf=0.25, device=DEVICE)
            det = results[0]
            boxes = det.boxes.xyxy.cpu().numpy().astype(int)

            # 2) Với mỗi bbox: crop + predict action
            annotated = img.copy()
            for (x1,y1,x2,y2) in boxes:
                crop = Image.fromarray(img[y1:y2, x1:x2][:,:,::-1])  # BGR→RGB
                idx, conf = predict_action(crop)
                label = CLASS_NAMES[idx]
                # vẽ bbox + nhãn
                cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            # Đưa vào queue & cập nhật lịch sử
            self.queue.put((annotated, now))
            self.last = now

        # luôn trả lại frame gốc (không annotate) để stream mượt
        return frame

# === STREAMLIT UI ===
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📹 Webcam Stream")
    ctx = webrtc_streamer(
        key="auto-capture",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Detector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False
    )

with col2:
    st.markdown("## 📸 Auto-Capture mỗi 3s")
    if ctx.video_processor:
        # lấy hết queue
        while not ctx.video_processor.queue.empty():
            frame, ts = ctx.video_processor.queue.get()
            st.session_state.history.insert(0, (frame, ts))
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]

    if not st.session_state.history:
        st.info("Chưa có ảnh nào được capture")
    else:
        cols = st.columns(5)
        for i,(frame,ts) in enumerate(st.session_state.history):
            cols[i].image(frame[:,:,::-1], caption=time.strftime("%H:%M:%S", time.localtime(ts)))

