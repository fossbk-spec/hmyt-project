# [Đề tài 2.2 - Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt]

- **Đề tài:** 2.2 — Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt
- **Nhóm:** 14
- **Mô hình chính:** XAI/Grad-CAM

## Mục đích của bài nộp này
Đây là bài nộp chính thức cho đề tài Phân loại mức độ Bệnh võng mạc đái tháo đường (DR). Mục đích là minh họa:

- Cấu trúc thư mục và định dạng báo cáo mà đề tài mong đợi.
- Một pipeline chạy được đầu-cuối thật sự (không phải số liệu dựng sẵn) — từ EDA, tiền xử lý ảnh (ROI + CLAHE), huấn luyện Baseline, so sánh đối chứng các mô hình, đến XAI với Grad-CAM.
- Cách đối chiếu kết quả một cách trung thực — phản ánh đúng những hạn chế cố hữu của dữ liệu y tế (mất cân bằng lớp cực đoan, hiện tượng Shortcut Learning) thay vì chỉnh sửa để "trông có vẻ hoàn hảo".
- Số liệu, tham số mô hình, và kết luận của nhóm hoàn toàn dựa trên kết quả thực nghiệm thực tế từ môi trường Kaggle.

## Cấu trúc thư mục
- `report/BAO_CAO_DO_AN.docx` — Báo cáo đầy đủ các phần + hình ảnh biểu đồ (EDA, Confusion Matrix, biểu đồ FP/FN, và ảnh Grad-CAM).
- `src/` — Mã nguồn chạy được (`eda.py`, `preprocess.py`, `train_baseline.py`, `evaluate.py`, `xai_gradcam.py`, v.v.).
- `data/` — Bộ dữ liệu đã dùng (chứa file text ghi link tải bản sao công khai từ Kaggle: https://www.kaggle.com/datasets/tanzinabdul/fundus-patientwise-split)

## Cách chạy lại (reproducibility)
Mở terminal, di chuyển vào thư mục chứa mã nguồn:
`cd src`
Cài đặt các thư viện phụ thuộc:
`pip install pandas numpy matplotlib seaborn opencv-python scikit-learn`
`pip install torch torchvision albumentations`
`pip install pytorch-grad-cam`
Chạy các script theo đúng thứ tự của Pipeline:
- `python eda.py` (Phân tích khám phá dữ liệu - EDA)
- `python preprocess.py` (Tiền xử lý ảnh: Crop ROI + CLAHE + Resize)
- `python train_baseline.py` (Huấn luyện mô hình ResNet-18 - Baseline)
- `python train_densenet.py` (Huấn luyện mô hình DenseNet-121 - Đối chứng)
- `python train_efficientnet.py` (Huấn luyện mô hình EfficientNet-B0 - Đối chứng)
- `python evaluate.py` (Đánh giá mô hình: F1, Sensitivity, Confusion Matrix)
- `python xai_gradcam.py` (Trích xuất bản đồ nhiệt Grad-CAM)

**Dữ liệu đầu vào:**
*   **Nguồn dữ liệu:** Bộ dữ liệu ảnh võng mạc BRSET (`tanzinabdul/fundus-patientwise-split`) được tải công khai từ nền tảng Kaggle.
*   **Lưu trữ:** Do dung lượng dữ liệu rất lớn (~15GB), thư mục `data/` chỉ chứa file `link_tai_du_lieu.txt` ghi lại đường dẫn tải, không chứa file ảnh trực tiếp để tránh quá tải khi nộp bài.
*   **Đường dẫn chạy mặc định:** Trong các file script, biến `BASE` được trỏ mặc định tới đường dẫn `/kaggle/input/fundus-patientwise-split/dataset_split`.
*   **Chạy ở môi trường Local:** Nếu chạy trên máy tính cá nhân, người dùng cần tải dữ liệu về và **sửa lại biến `BASE`** trong các script cho trỏ đúng tới thư mục chứa dữ liệu đã tải.

