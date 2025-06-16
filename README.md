# 🎬 Human Action Recognition with ConViT + YOLOv8 (Demo)

This is a Streamlit-based demo for recognizing human actions from still images using a fine-tuned **ConViT model** on the Stanford40 dataset. **YOLOv8** is used only for detecting the person and cropping the input region to reduce background clutter.

---

## 🚀 How to Run the Demo

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights:
Place your fine-tuned ConViT weights file at: /DEMO/

### 3. Launch the Streamlit app
``` bash
streamlit run DEMO/app.py
```
The app will capture a frame from your webcam every 5 seconds.

YOLOv8 detects the person and crops the region.

ConViT predicts the action label and displays it with a bounding box.

📂 Project Structure
```pgsql
├── Phase1_89mAP/              # Phase 1: ConViT with frozen backbone
├── Phase2_93.25mAP/           # Phase 2: ConViT with last 2 blocks unfrozen
├── DEMO/
│   ├── app.py                 # Streamlit demo: YOLOv8 + ConViT pipeline
│   ├── best_yoloS.pt.py               # Utility functions for preprocessing
│   └── best_convit_stanford40.pth.pth   # Your trained ConViT model checkpoint
├── requirements.txt
└── README.md
```
