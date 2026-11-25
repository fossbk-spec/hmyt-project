# Tóm tắt bài làm

- **Đề tài:** Dự đoán nguy cơ đột quỵ (stroke) từ dữ liệu y tế bằng các mô hình học máy cổ điển, có xử lý mất cân bằng lớp và giải thích mô hình.
- **Nhóm:** nhom22
- **Mô hình chính:** Logistic Regression (đã tinh chỉnh, `C=0.1`, `penalty='l2'`), so sánh với Random Forest và SVM; dùng SMOTE để cân bằng lớp trên tập Train và SHAP để giải thích Random Forest.
- **Kết quả nổi bật:** (điền sau khi chạy lại notebook) Logistic Regression trên tập Test: F1 = 0.231, Recall = 0.800, ROC AUC = 0.842

## Cấu trúc thư mục

- `report/` — báo cáo (PDF hoặc Markdown)
- `src/` — mã nguồn (notebook `Stroke_detection.ipynb`)
- `slides/` — slide thuyết trình

## Cách chạy lại (reproducibility)

**Dữ liệu.** Bộ *Stroke Prediction Dataset* (Kaggle, tác giả fedesoriano), tệp `healthcare-dataset-stroke-data.csv`: 5.110 dòng, 12 cột, mục tiêu là `stroke` (249 ca dương tính, khoảng 4,9%). Mỗi dòng là một bệnh nhân (`id` không trùng lặp). Tải tệp từ Kaggle và đặt cùng thư mục với notebook (`src/healthcare-dataset-stroke-data.csv`).

**Môi trường.** Python 3.14 (đã chạy thử). Cài thư viện từ `requirements.txt` ở thư mục gốc bài nộp:

```bash
pip install -r requirements.txt
```

**Chạy.** Mở `src/Stroke_detection.ipynb` và chạy tuần tự từ trên xuống dưới (Run All). Chạy đúng thứ tự vì bước điền `bmi` nằm sau bước chia tập.

**Quy trình xử lý (tránh rò rỉ dữ liệu).**
1. Chia Train/Test 80/20 bằng `train_test_split(train_size=0.8, random_state=42)` **trước** mọi bước điền giá trị thiếu. Notebook kiểm tra bằng `assert` rằng Train và Test không có bệnh nhân (`id`) chung.
2. Điền `bmi` thiếu (201 dòng) bằng `StandardScaler + DecisionTreeRegressor` dựa trên `age` và `gender`. Bộ điền được fit chỉ trên Train, sau đó áp dụng cho cả Train và Test.
3. SMOTE (`random_state=42`) chỉ áp dụng trên tập Train; tập Test giữ nguyên phân phối gốc.
4. Huấn luyện Random Forest, SVM, Logistic Regression; đánh giá trên Test bằng F1, Accuracy, Recall, Precision, ROC AUC.

**Tính tái lập.** Mọi bước ngẫu nhiên (chia tập, SMOTE, Random Forest, SVM, Logistic Regression, Decision Tree) dùng `random_state=42`, nên chạy lại trên cùng phiên bản thư viện sẽ cho cùng kết quả.
