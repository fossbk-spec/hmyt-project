# ==================== HUẤN LUYỆN BASELINE RESNET-18 ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import numpy as np
import random
import os
import glob
from tqdm import tqdm

# Fix lỗi deadlock của OpenCV khi dùng DataLoader đa luồng
cv2.setNumThreads(0)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị đang dùng:", device)

# Tự động tìm BASE
train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))

PROC = "/kaggle/working/processed"
CHECKPOINT_PATH = "/kaggle/working/checkpoint_resnet18.pth"
BEST_MODEL_PATH = "/kaggle/working/mo_hinh_resnet18_tot_nhat.pth"

# Đọc DataFrame 1 lần duy nhất ở đây (lưu trên RAM) thay vì trong Dataset để chống nghẽn I/O
df_train = pd.read_csv(f"{BASE}/train/train.csv")
df_val   = pd.read_csv(f"{BASE}/val/val.csv")

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}.jpg"
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["DR_ICDR"])
        if self.transform: img = self.transform(image=img)["image"]
        return img, label

train_transform = A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])

train_dataset = DRDataset(df_train, f"{PROC}/train", train_transform)
val_dataset   = DRDataset(df_val,   f"{PROC}/val",   val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

y_train = df_train["DR_ICDR"].values
# Đã sửa lỗi keyword argument cho scikit-learn bản mới
weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2,3,4]), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
scaler = torch.amp.GradScaler('cuda')

start_epoch, best_acc, patience, patience_counter = 1, 0.0, 5, 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"\nTìm thấy checkpoint tại {CHECKPOINT_PATH}. Đang khôi phục...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    patience_counter = checkpoint['patience_counter']
    print(f"Khôi phục thành công! Tiếp tục train từ Epoch {start_epoch}.\n")
else:
    print("\nKhông tìm thấy checkpoint. Bắt đầu train từ đầu.\n")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

for epoch in range(start_epoch, 13):
    print(f"\nEpoch {epoch}/12")
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va = evaluate(model, val_loader, criterion)
    scheduler.step()
    print(f"Train Loss: {tl:.4f} | Train Acc: {ta:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.4f}")

    if va > best_acc:
        best_acc = va; patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f" -> Đã lưu mô hình tốt nhất (Val Acc: {best_acc:.4f})")
    else:
        patience_counter += 1
        print(f" -> Không cải thiện (Patience: {patience_counter}/{patience})")

    # SỬA LỖI SINH TỬ: Lưu Checkpoint TRƯỚC KHI kiểm tra Early Stopping
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(), 'best_acc': best_acc, 'patience_counter': patience_counter
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f" -> Đã lưu Checkpoint Epoch {epoch}")

    if patience_counter >= patience:
        print(f"\nEarly stopping tại epoch {epoch}")
        break

print(f"\nVal Accuracy tốt nhất: {best_acc:.4f}")
