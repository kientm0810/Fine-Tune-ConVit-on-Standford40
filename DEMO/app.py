import os
import time
import queue
import torch
import streamlit as st
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import timm
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# === GIAO DIỆN STREAMLIT ===
st.set_page_config(page_title="YOLO + ConViT Demo", layout="wide")
st.title("YOLOv8 + ConViT trên Web với Streamlit (Auto‐Capture 5s)")
st.write("Webcam streaming, tự động capture và classify mỗi 5 giây\n")
st.write("---")
st_autorefresh(interval=4800, limit=None, key="autorefresh")

# === THÔNG SỐ CHUNG ===
YOLO_WEIGHTS   = "best_yoloS.pt"
CONVIT_WEIGHTS = "best_convit_stanford40.pth"
LABEL_PATH     = "labels.txt"
IMG_SIZE       = 224
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# === SESSION_STATE CHO LỊCH SỬ ẢNH ===
if "history" not in st.session_state:
    st.session_state.history = []  # list of (PIL.Image, timestamp)

# === LOAD NHÃN ===
@st.cache_data
def load_labels(path):
    if not os.path.exists(path):
        st.error(f"Không tìm thấy file nhãn: {path}")
        st.stop()
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

CLASS_NAMES = load_labels(LABEL_PATH)
NUM_CLASSES = len(CLASS_NAMES)

# === LOAD MÔ HÌNH ===
@st.cache_resource(show_spinner=False)
def load_yolo(path):
    if not os.path.exists(path):
        st.error(f"Không tìm thấy YOLO weights: {path}")
        st.stop()
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_convit(path, num_classes):
    if not os.path.exists(path):
        st.error(f"Không tìm thấy ConViT weights: {path}")
        st.stop()
    m = timm.create_model("convit_base", pretrained=False, num_classes=num_classes)
    sd = torch.load(path, map_location=DEVICE)
    m.load_state_dict(sd)
    return m.eval().to(DEVICE)

yolo_model   = load_yolo(YOLO_WEIGHTS)
convit_model = load_convit(CONVIT_WEIGHTS, NUM_CLASSES)

# === TIỀN XỬ LÝ & DỰ ĐOÁN ACTION ===
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
def predict_action(crop: Image.Image):
    t = preprocess(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = convit_model(t)
        p = torch.softmax(out, dim=1)
        conf, idx = torch.max(p, dim=1)
    return idx.item(), conf.item()

# === VIDEO PROCESSOR ===
class AutoCapture(VideoProcessorBase):
    def __init__(self):
        self.last    = time.time()
        self.interval= 4.8
        self.queue   = queue.Queue()
        self.start_time = time.time()

    def recv(self, frame):
        # chuyển frame WebRTC → numpy BGR → PIL RGB
        img_bgr = frame.to_ndarray(format="bgr24")
        img_pil = Image.fromarray(img_bgr[:,:,::-1])
        now      = time.time()

        if now - self.last >= self.interval:
            # 1) Dò bbox với YOLO
            results = yolo_model.predict(
                source=img_bgr, imgsz=640, conf=0.25, device=DEVICE
            )
            det = results[0]
            boxes = det.boxes.xyxy.cpu().numpy().astype(int)

            # 2) Annotate lên ảnh
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()
            for (x1,y1,x2,y2) in boxes:
                # crop rồi predict action
                crop = img_pil.crop((x1,y1,x2,y2))
                idx, conf = predict_action(crop)
                lbl = CLASS_NAMES[idx]

                # vẽ bbox
                draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
                # vẽ nhãn lên
                text = f"{lbl} {conf:.2f}"
                draw.text((x1, y1-12), text, fill="lime", font=font)

            # đẩy vào lịch sử
            self.queue.put((lbl, img_pil.copy(), now - self.start_time))
            self.last = now

        return frame  # stream gốc để mượt

# === STREAMLIT UI ===
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("## Webcam Streaming (Auto‐Capture 5s)")
    ctx = webrtc_streamer(
        key="auto-capture",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AutoCapture,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

with col2:
    st.markdown("## Lịch sử 5 ảnh gần nhất")
    if ctx.video_processor:
        while not ctx.video_processor.queue.empty():
            lbl, img, ts = ctx.video_processor.queue.get()
            st.session_state.history.insert(0, (lbl, img, ts))
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]

    if not st.session_state.history:
        st.info("Chưa có ảnh nào được capture")
    else:
        hist_cols = st.columns(5)
        for i, (lbl, im, ts) in enumerate(st.session_state.history):
            hist_cols[i].image(
                im, caption=f"{lbl}, {ts:0.2f}",
                use_container_width=True
            )

st.markdown("---")
st.caption(
    "• Ứng dụng stream video, tự động capture mỗi 5s, kết hợp YOLOv8 + ConViT để detect + classify."
)
