# [Đề tài 2.2 - Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt]

- **Đề tài:** 2.2 — Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt
- **Nhóm:** 14
- **Mô hình chính:** XAI/Grad-CAM

## Mục đích của ví dụ này

Đây **không phải** đáp án hay bài nộp mẫu để chép lại. Mục đích là minh
họa:
1. Cấu trúc thư mục và định dạng báo cáo mà đề tài 3.4 mong đợi.
2. Một pipeline chạy được đầu-cuối thật sự (không phải số liệu dựng sẵn) —
   từ EDA, baseline, SMOTE+ablation, đến XAI với SHAP.
3. Cách đối chiếu kết quả của nhóm với y văn gốc **một cách trung thực**
   — kể cả khi kết quả không khớp con số công bố (xem báo cáo, đặc biệt
   Mục 4.2–4.3), thay vì chỉnh sửa để "trông giống" bài báo.

Số liệu, tham số mô hình, và kết luận của nhóm bạn hoàn toàn có thể (và
nên) khác với ví dụ này.

## Cấu trúc thư mục

- `report/BAO_CAO_VI_DU.md` — báo cáo đầy đủ 6 phần + 2 biểu đồ SHAP
- `src/` — mã nguồn chạy được (`run_analysis.py`, `run_ablation.py`)
- `data/` — bộ dữ liệu đã dùng (bản sao công khai từ UCI)

## Cách chạy lại (reproducibility)

```
cd src
pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib
python run_analysis.py     # EDA + baseline + SMOTE + Extra Trees + SHAP
python run_ablation.py     # So sánh cấu hình mô hình/tập đặc trưng
```

Dữ liệu đầu vào: `data/heart_failure_clinical_records.csv` (đặt cùng thư
mục với script khi chạy, hoặc sửa đường dẫn trong script).
