# Tóm tắt bài làm

- **Đề tài:** Predicting Breast Cancer Outcomes 
- **Nhóm:** 33
- **Mô hình chính:** XGBoost, Gradient Boosting, Logistic Regression, 
- **Kết quả nổi bật:** Tesst ROC-AUC = '0.8015', Test PR-AUC ='0.7103', Brief Score = '0.6481'

## Cấu trúc thư mục

- `report/` — báo cáo dạng Word 
- `src/` — 'eda.ipynb', 'preprocessing.ipynb', 'Model_selection.ipynb'

## Cách chạy lại (reproducibility)
Dataset: Incidence - SEER Research Data, 17 Registries, Nov 2024 Sub (2000-2022)
Tải dataset lên google drive cá nhân hoặc folder dự án
### 2. Chạy trực tiếp trên Google Colab (Khuyên dùng)
1. Mở lần lượt các file notebook trong thư mục `src/` bằng **Google Colab** (nhấn nút *"Open in Colab"* hoặc tải file `.ipynb` lên Google Drive và mở bằng Google Colaboratory)
2. Kết nối Google Drive (`drive.mount('/content/drive')`) và chỉnh lại biến đường dẫn file `.csv` (nếu cần).
3. Cài đặt bổ sung thư viện còn thiếu bằng lệnh sau tại cell đầu tiên (các thư viện phổ biến như `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn` đã có sẵn trên Colab):
