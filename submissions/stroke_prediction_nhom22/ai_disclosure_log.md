# Nhật ký sử dụng AI (AI Disclosure Log)

- **Nhóm:** nhom22
- **Đề tài:** Dự đoán nguy cơ đột quỵ từ các biến lâm sàng thường quy
- **Cập nhật lần cuối:** 16/11/2026

## 1. Công cụ AI đã sử dụng

| Công cụ | Mục đích | Mức độ |
| --- | --- | --- |
| Claude (Anthropic), giao diện web claude.ai | Soạn thảo báo cáo; hỗ trợ viết và sửa code; soạn tài liệu nộp bài | Chính |
| Consensus | Tìm kiếm tài liệu ban đầu | Phụ (theo mục AI usage của báo cáo) |
| NotebookLM | Tóm tắt các bài báo | Phụ (theo mục AI usage của báo cáo) |

## 2. Chi tiết sử dụng Claude

| # | Hạng mục | Claude làm gì | Phần nhóm tự làm / kiểm tra |
| --- | --- | --- | --- |
| 1 | Báo cáo (`report/`) | Soạn thảo và cô đọng nội dung báo cáo từ slide và notebook của nhóm | Nhóm cung cấp nội dung gốc (slide, notebook, số liệu EDA), đọc lại và chịu trách nhiệm về nội dung cuối cùng |
| 2 | Notebook `src/Stroke_detection.ipynb`: sửa rò rỉ dữ liệu | Theo nhận xét review: chuyển bước điền `bmi` ra sau bước chia Train/Test, fit bộ điền trên Train rồi áp dụng cho Test; thêm `assert` kiểm tra điền xong không còn NaN | Nhóm chạy lại toàn bộ notebook và cập nhật số liệu |
| 3 | Notebook: kiểm tra chia tập theo bệnh nhân | Kiểm tra trên dữ liệu: 5.110 `id` duy nhất, không có dòng trùng; thêm `assert` Train ∩ Test = ∅ theo `id` | Nhóm xác nhận kết quả khi chạy |
| 4 | Notebook: tính tái lập | Thêm `random_state=42` cho SMOTE và Random Forest; đổi đường dẫn dữ liệu thành đường dẫn tương đối; thay các lệnh `pip install` không ghim phiên bản bằng ghi chú dùng `requirements.txt` | Nhóm chạy lại để kiểm tra |
| 5 | `requirements.txt` | Dựng file từ danh sách phiên bản `pip freeze` do nhóm cung cấp | Nhóm tạo danh sách phiên bản trên máy của mình |
| 6 | `README.md` | Soạn nội dung theo mẫu của giảng viên | Nhóm điền kết quả (F1, Recall, ROC AUC) từ lần chạy của mình |
| 7 | `references.bib` | Tra DOI của 6 tài liệu trong báo cáo và đối chiếu với trang nhà xuất bản; sắp xếp thành định dạng BibTeX | Nhóm chọn danh sách tài liệu |
| 8 | Tài liệu này | Soạn bản nháp từ mô tả của nhóm | Nhóm xem lại và xác nhận |

## 3. Giới hạn của phần việc do Claude thực hiện

- Claude **không** chạy toàn bộ notebook trên dữ liệu của nhóm (môi trường không có `imbalanced-learn` và `shap`). Mọi chỉ số mô hình (F1, Recall, ROC AUC, ma trận nhầm lẫn, SHAP) là kết quả từ lần chạy của nhóm.
- Việc chọn đề tài, bộ dữ liệu, mô hình so sánh và các kết luận trong báo cáo là quyết định của nhóm.

## 4. Cam kết

Nhóm đã đọc lại, kiểm tra và chịu trách nhiệm về mọi nội dung có sự hỗ trợ của AI trong bài nộp này.
