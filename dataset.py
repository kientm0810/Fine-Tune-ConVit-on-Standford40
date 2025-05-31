# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class Stanford40Dataset(Dataset):
    def __init__(self, img_dir, split_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(split_file) as f:
            # danh sách tên ảnh (không .jpg)
            self.names = [l.strip().split(".")[0] for l in f if l.strip().split(".")[0]]
        # build list nhãn đầy đủ
        labels = sorted({self._label_from_name(n) for n in self.names})
        self.cls2idx = {c:i for i,c in enumerate(labels)}

    def _label_from_name(self, name):
        parts = name.split("_")
        return "_".join(parts[:-1])  # hoặc " ".join(parts[:-1])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name + ".jpg")
        img = Image.open(img_path).convert("RGB")
        label_str = self._label_from_name(name)
        label = self.cls2idx[label_str]
        if self.transform:
            img = self.transform(img)
        return img, label
