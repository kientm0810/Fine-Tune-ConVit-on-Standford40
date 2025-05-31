# prepare_stanford40_dirs.py
import os, shutil

ROOT = r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\JPEGImages"
SPLITS = r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\ImageSplits\train.txt"  # hoặc per-class, tuỳ Stanford40 bạn có
OUT   = r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\organized_data"

def read_split(fp):
    with open(fp) as f:
        return [l.strip() for l in f if l.strip()]

def label_from_fname(fn):
    parts = fn.split("_")
    # ví dụ ['riding','a','horse','0001'] → nối lại ['riding','a','horse']
    return " ".join(parts[:-1])


os.makedirs(os.path.join(OUT,"train"), exist_ok=True)
os.makedirs(os.path.join(OUT,"val"),   exist_ok=True)

for phase, split in [("train","train.txt"), ("val","test.txt")]:
    lines = read_split(os.path.join(r"path/to/Stanford40/ImageSplits", split))
    for name in lines:
        lbl = label_from_fname(name)
        src = os.path.join(ROOT, name+".jpg")
        dst_dir = os.path.join(OUT, phase, lbl)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name+".jpg")
        shutil.copy2(src, dst)

# print(label_from_fname("riding_a_horse_0001.jpg"))
