#!/usr/bin/env python3
# finetune_convit_on_stanford40.py

import os
import timm
import torch
import torch.nn as nn
from torchvision import transforms
# from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.multiprocessing as mp
from dataset import Stanford40Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# =============== Hyperparameters & Paths ===============
# DATA_ROOT = "organized_data"       # đầu ra của step tổ chức dữ liệu
NUM_CLASSES = 40                   # Stanford40
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============== Transforms ===============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])

# Bọc lại vì WinDown cần thế
def main():
  # =============== Datasets & Loaders ===============

    train_ds = Stanford40Dataset(
        img_dir=r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\JPEGImages",
        split_file=r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\ImageSplits\train.txt",
        transform=train_transform
    )
    val_ds = Stanford40Dataset(
        img_dir=r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\JPEGImages",
        split_file=r"C:\Users\andin\OneDrive\Desktop\Project Phase1 VDT2025\Data\Stanford40\ImageSplits\test.txt",
        transform=val_transform
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # ======================
    # 1. Tạo ConViT (không load ImageNet)
    # ======================
    model = timm.create_model("convit_base", pretrained=False)

    # 2. Thay head để tương thích với 40 class
    #    (nếu file checkpoint của bạn lưu đúng state_dict gồm cả head với 40 outputs)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    model.to(DEVICE)

    # 3. Load weights từ checkpoint best_convit_stanford40.pth
    if os.path.isfile("best_convit_stanford40.pth"):
        print("=> Loading checkpoint weights from best_convit_stanford40.pth")
        state_dict = torch.load("best_convit_stanford40.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("   → Đã load weights; sẵn sàng tiếp tục training.")
    else:
        print("=> Không tìm thấy best_convit_stanford40.pth, sẽ train từ đầu (chỉ head đã được khởi tạo ngẫu nhiên).")

    # ======================
    # 4. (Tùy chọn) Freeze backbone, chỉ train head
    # ======================
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    # =============== Optimizer (chỉ train head) ===============
    optimizer = SGD(model.head.parameters(),
                    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)   
    # Tạo scheduler cho head
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

  # =============== Training Loop ===============
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        # =============== Validation ===============
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [ Val ]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"  >> Val Acc: {acc*100:.2f}%  |  Avg Train Loss: {running_loss/len(train_ds):.4f}")

        # lưu model tốt nhất
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_convit_stanford40.pth")

    print(f"Best Val Acc: {best_acc*100:.2f}%")

if __name__ == "__main__":
    mp.freeze_support()   # cho Windows
    main()