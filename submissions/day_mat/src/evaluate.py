# ==================== ĐÁNH GIÁ TEST & METRICS ====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pandas as pd
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

cv2.setNumThreads(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN
train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))
PROC = "/kaggle/working/processed"

MODEL_PATH = "/kaggle/working/mo_hinh_resnet18_tot_nhat.pth" 

# SỬA LỖI: Kiểm tra file mô hình có thực sự tồn tại không trước khi load
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Không tìm thấy file mô hình .pth! Hãy chắc chắn Cell 8 đã chạy thành công.")

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

val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])

df_test = pd.read_csv(f"{BASE}/test/test.csv")
test_dataset = DRDataset(df_test, f"{PROC}/test", val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Đang dự đoán Test"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n" + "=" * 60)
print("BÁO CÁO PHÂN LOẠI (TEST SET)")
print("=" * 60)
print(classification_report(all_labels, all_preds, 
                            target_names=["0 Không có", "1 Nhẹ", "2 Trung bình", "3 Nặng", "4 Tăng sinh"], 
                            zero_division=0, digits=4))

cm = confusion_matrix(all_labels, all_preds)
specificities = []
for i in range(len(cm)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(spec)

print("Specificity (Độ đặc hiệu) từng lớp:")
for i, spec in enumerate(specificities):
    print(f"  Lớp {i}: {spec:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["0", "1", "2", "3", "4"], yticklabels=["0", "1", "2", "3", "4"])
plt.title("Confusion Matrix - Test Set", fontweight="bold")
plt.xlabel("Dự đoán (Predicted)"); plt.ylabel("Thực tế (True)")
plt.tight_layout(); plt.savefig("/kaggle/working/confusion_matrix_test.png", dpi=150); plt.show()

# TÍNH TOÁN FALSE POSITIVE (FP) VÀ FALSE NEGATIVE (FN)
mcm = multilabel_confusion_matrix(all_labels, all_preds)
class_names = ["0 Không có", "1 Nhẹ", "2 Trung bình", "3 Nặng", "4 Tăng sinh"]

data_fp_fn = []
for i in range(5):
    tn, fp, fn, tp = mcm[i].ravel()
    data_fp_fn.append({
        "Mức độ bệnh": class_names[i],
        "True Positive (Đoán đúng)": tp,
        "False Positive (Báo động giả - FP)": fp,
        "False Negative (Bỏ sót bệnh - FN)": fn,
        "True Negative (Đoán đúng không bệnh)": tn
    })

df_fp_fn = pd.DataFrame(data_fp_fn)
print("\n" + "=" * 80)
print("BẢNG PHÂN TÍCH FALSE POSITIVE (FP) VÀ FALSE NEGATIVE (FN) TRÊN TẬP TEST")
print("=" * 80)
print(df_fp_fn.to_string(index=False))

# SỬA LỖI: Xoay nhãn trục X 30 độ để không bị đè chữ
df_fp_fn.plot(x="Mức độ bệnh", y=["False Positive (Báo động giả - FP)", "False Negative (Bỏ sót bệnh - FN)"], 
              kind="bar", figsize=(10, 5), color=["#ff9999", "#66b3ff"])
plt.title("So sánh số ca Báo động giả (FP) và Bỏ sót bệnh (FN)", fontweight="bold")
plt.ylabel("Số lượng ca")
plt.xticks(rotation=30, ha='right') # Đã sửa ở đây
plt.tight_layout()
plt.savefig("/kaggle/working/fp_fn_chart.png", dpi=150)
plt.show()
