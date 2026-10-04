<!--
Mẫu PR nộp bài đồ án ET4248. Điền đầy đủ trước khi nhờ giảng viên/TA review.
CI "Review Submission (rubric)" đọc checklist bên dưới — đánh dấu [x] trung thực.
-->

## Thông tin bài nộp

- Thư mục: `submissions/<topic_slug>_<ma_nhom>/`
- Đề tài: <mã đề tài> — <tên đề tài>
- Mốc nộp: M1 / M2 / M3 / M4 / Chung kết (ghi thêm `"milestone"` trong `submission.json`)
- Repo nhóm (nhánh gd1–gd7, nếu có):

## Tự kiểm tra trước khi nộp

- [ ] `python scripts/validate_submission.py submissions/<topic_slug>_<ma_nhom>` → `[PASSED]`
- [ ] `python scripts/review_project.py submissions/<topic_slug>_<ma_nhom>` → đã đọc báo cáo và xử lý các mục ❌ (hoặc giải trình bên dưới)

## Biomedical Reproducibility Checklist

- [ ] Dữ liệu công khai/được cấp phép; ghi rõ nguồn, phiên bản, link tải; không có dữ liệu bệnh nhân thật/định danh
- [ ] Chia tập ở mức bệnh nhân, có `assert` chứng minh Train ∩ Test = ∅ (theo mã bệnh nhân)
- [ ] Scaler / Imputer / chọn đặc trưng / SMOTE chỉ fit trên Train (hoặc nằm trong Pipeline)
- [ ] Cố định random seed
- [ ] `requirements.txt` cố định phiên bản; README có lệnh tái lập kết quả
- [ ] Báo cáo Sensitivity, Specificity, F1, ROC-AUC, PR-AUC (+ 95% CI) — không chỉ Accuracy
- [ ] Có XAI (SHAP/Grad-CAM) và phân tích lỗi sai
- [ ] Mọi số liệu trong báo cáo truy vết được tới output (notebook đã chạy / file kết quả)
- [ ] Mọi trích dẫn có trong `.bib` với DOI đã xác minh — không dùng trích dẫn do AI tự sinh
- [ ] Có mục AI Disclosure trong báo cáo + `docs/ai_disclosure_log.md`

## Giải trình (nếu có)

Các mục ❌/⚠️ trong báo cáo review mà nhóm cho là dương tính giả, hoặc hạn chế đã biết của bài làm.
