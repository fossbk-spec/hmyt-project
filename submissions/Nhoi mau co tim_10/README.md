# Tóm tắt bài làm

- **Đề tài:** 1.2 — Phân loại đa nhãn rối loạn nhịp tim từ ECG 12 chuyển đạo (PTB-XL)
- **Nhóm:** nhom10
- **Mô hình chính:** <vd: baseline đặc trưng + Random Forest/Logistic Regression; CNN 1D ResNet + Grad-CAM>
- **Kết quả nổi bật:** <vd: AUROC = x.xxx (95% CI x.xxx–x.xxx), F1 = x.xxx trên fold 10; CI tính bằng cluster bootstrap theo bệnh nhân>

## Cấu trúc thư mục

- `report/` — báo cáo (`report.md`, `refs.bib`, `van_dap.md`)
- `src/` — mã nguồn (`common.py`, `features.py`, `train_*.py`, `analyze.py`, `gradcam.py`, `make_report.py`)
- `slides/` — slide thuyết trình (`build_slides.py`, `slides.pptx` nếu < 10 MB)
- `results/` — kết quả JSON và hình sinh từ dữ liệu thật
- `docs/` — `ai_disclosure_log.md` (nhật ký sử dụng AI)
- `run_all.sh`, `requirements.txt`, `submission.json`

## Cách chạy lại (reproducibility)

**1. Môi trường**

- Python <phiên bản, vd 3.11>
- Cài thư viện (đã ghim phiên bản):

```bash
pip install -r requirements.txt
```

**2. Dữ liệu**

- PTB-XL v1.0.3 (PhysioNet, CC BY 4.0): https://physionet.org/content/ptb-xl/1.0.3/
- Tải về và đặt tại `data/ptb-xl/` (thư mục `data/` không được commit).
- Dữ liệu công khai, không chứa thông tin định danh bệnh nhân.

**3. Chạy**

```bash
bash run_all.sh
```

Lệnh này huấn luyện, đánh giá, sinh hình và báo cáo vào `results/` và `report/`.
Không dùng cờ `--synthetic` (chỉ để kiểm tra nhanh, dữ liệu mô phỏng).

**4. Thiết lập để tái lập kết quả**

- Chia tập theo `strat_fold`: fold 1–8 train, fold 9 validation, fold 10 test; có `assert` không trùng bệnh nhân giữa các tập.
- Scaler/Imputer chỉ fit trên Train; ngưỡng phân loại chọn trên validation, không chọn trên test.
- Seed cố định: <SEED>.
- Khoảng tin cậy 95%: cluster bootstrap theo `patient_id`, 1000 lần.
- Thời gian chạy ước tính: <vd: baseline ~x phút (CPU); CNN ~y giờ (GPU ...)>.

**5. File lớn** (nếu có): <link Google Drive>

## Hạn chế

<điền trung thực: mất cân bằng lớp, nhãn nhiễu, chỉ một bộ dữ liệu, tần số lấy mẫu dùng, ...>