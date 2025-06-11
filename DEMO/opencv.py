import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import timm
import os

# --- THÔNG SỐ ---
MODEL_PATH = 'best_convit_stanford40.pth'
LABEL_PATH = 'labels.txt'
IMG_SIZE = 224           # Đổi nếu model bạn dùng kích thước khác
CAPTURE_INTERVAL = 3     # Số giây tự động capture

# --- ĐỌC NHÃN ---
def load_labels(path=LABEL_PATH):
    if not os.path.exists(path):
        print(f"Lỗi: Không tìm thấy file {path}")
        exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

CLASS_NAMES = load_labels(LABEL_PATH)
NUM_CLASSES = len(CLASS_NAMES)

# --- ĐỊNH NGHĨA MÔ HÌNH ConViT ---
class ConViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConViT, self).__init__()
        self.model = timm.create_model('convit_based', pretrained=False, num_classes=num_classes)
        # Nếu bạn train với 'convit_small' hay 'convit_base' thì sửa tên model ở đây!

    def forward(self, x):
        return self.model(x)



# --- LOAD MODEL ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(weights_path, num_classes):
    model = timm.create_model("convit_base", pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model


model = load_model(weights_path=MODEL_PATH, num_classes=NUM_CLASSES)
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file {MODEL_PATH}")
    exit(1)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- TIỀN XỬ LÝ ẢNH ---
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def predict(frame):
    tensor = preprocess(frame)
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
    return pred.item()

# --- MAIN ---
def main():
    cap = cv2.VideoCapture(0)
    last_capture = 0
    pred_label = ""
    print("Ấn 'c' để chụp ngay, 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được hình từ webcam.")
            break

        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        auto_capture = now - last_capture > CAPTURE_INTERVAL
        manual_capture = key == ord('c')

        if auto_capture or manual_capture:
            idx = predict(frame)
            if idx < len(CLASS_NAMES):
                pred_label = CLASS_NAMES[idx]
            else:
                pred_label = str(idx)
            last_capture = now

        # Hiện nhãn lên ảnh
        cv2.putText(frame, f'Prediction: {pred_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ConViT Stanford40 Webcam Demo', frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
