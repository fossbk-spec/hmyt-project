# ==================== GRAD-CAM & SANITY CHECK ====================
import os
try:
    import pytorch_grad_cam
except ImportError:
    # Nếu chạy ngầm bị lỗi mạng, lệnh này sẽ cố gắng cài lại
    os.system('pip install grad-cam -q')

import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Fix lỗi deadlock của OpenCV
cv2.setNumThreads(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN
train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))
PROC = "/kaggle/working/processed"
MODEL_PATH = "/kaggle/working/mo_hinh_resnet18_tot_nhat.pth"

# 2. LOAD MODEL
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# 3. KHỞI TẠO GRAD-CAM
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])

# 4. LẤY ẢNH VÀ SINH BẢN ĐỒ NHIỆT
df_test = pd.read_csv(f"{BASE}/test/test.csv")
fig, axes = plt.subplots(5, 2, figsize=(10, 20))

# SỬA LỖI 3: Tắt trục (axis off) cho toàn bộ lưới ngay từ đầu để tránh khung trắng nham nhở
for ax in axes.flatten():
    ax.axis("off")

for i in range(5):
    df_muc = df_test[df_test["DR_ICDR"] == i]
    if len(df_muc) == 0: continue
    
    # Lấy ngẫu nhiên 1 ảnh mỗi lớp (cố định seed)
    row = df_muc.sample(1, random_state=42).iloc[0]
    
    img_path = f"{PROC}/test/{row['image_id']}.jpg"
    img_bgr = cv2.imread(img_path)
    
    if img_bgr is not None:
        # SỬA LỖI 2: Resize ảnh gốc về đúng 384x384 để khớp khít với ma trận Grad-CAM
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (384, 384))
        img_float = img_rgb_resized.astype(np.float32) / 255.0
        
        # Tiền xử lý đưa vào model
        input_tensor = val_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        
        # Sinh Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        
        # Đè bản đồ nhiệt (đã khớp kích thước 384x384)
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Hiển thị lên lưới
        axes[i, 0].imshow(img_rgb_resized)
        axes[i, 0].set_title(f"Ảnh gốc - Lớp {i}", fontweight="bold")
        
        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Grad-CAM - Lớp {i}", fontweight="bold")

plt.suptitle("Sanity Check: Grad-CAM có bắt đúng tổn thương không?", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/gradcam_sanity_check.png", dpi=150)
plt.show()
