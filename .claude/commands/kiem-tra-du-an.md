---
description: Kiểm tra 1 bài nộp đồ án ET4248 theo rubric và đề xuất phương án nâng cao chất lượng
argument-hint: <submissions/<topic_slug>_<ma_nhom> | số PR | all> [m1|m2|m3|m4|final] [hạn nộp YYYY-MM-DD]
---

Thực hiện đúng quy trình trong `.agents/workflows/kiem-tra-du-an.md` (đọc toàn bộ file đó trước) cho: $ARGUMENTS

- Tuân thủ mục "Nguyên tắc bắt buộc" của workflow: không chấm điểm chính thức, không merge/approve, mọi nhận xét có bằng chứng `file:dòng`, không chạy code sinh viên ngoài môi trường cô lập, không chép dữ liệu bệnh nhân vào nhận xét.
- Nếu đối số là `all`: chạy `python scripts/review_project.py --all --output-dir review_reports`, tóm tắt bảng tổng hợp và liệt kê các nhóm có vấn đề nghiêm trọng — không soạn nhận xét chi tiết cho từng nhóm trừ khi được yêu cầu.
- Kết thúc bằng bản nhận xét theo mẫu ở Bước 5, lưu thành file, và hỏi giảng viên trước khi đăng bất cứ gì lên PR.
