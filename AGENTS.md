# AGENTS.md — repo nộp bài `hmyt-project`

Chính sách cho mọi agent (Claude Code, Antigravity, Codex...) làm việc trong repo
nộp bài đồ án môn **Học Máy trong Y Tế (ET4248)**. Chính sách soạn sách nằm ở
repo `fossbk-spec/HMYT`, không áp dụng ở đây.

## Vai trò của agent trong repo này

- **Được làm:** chạy công cụ kiểm tra, đọc bài nộp, soạn nhận xét và phương án
  nâng cao chất lượng theo `.agents/workflows/kiem-tra-du-an.md`
  (Claude Code: lệnh `/kiem-tra-du-an`).
- **Không được làm:** chấm điểm chính thức, approve/merge PR, sửa thư mục
  `submissions/<...>/` của sinh viên, đăng nhận xét lên PR khi giảng viên
  chưa đồng ý.

## Bất biến

1. **Zero-Hallucination:** mọi nhận xét kèm bằng chứng `file:dòng`; chỉ dùng
   rubric/trang đề tài trong ngân hàng đề tài làm căn cứ.
2. **An toàn:** không chạy code sinh viên ngoài môi trường cô lập không có
   credential. `scripts/review_project.py` chỉ phân tích tĩnh — an toàn để
   chạy trên mọi PR.
3. **Quyền riêng tư:** không đưa dữ liệu bệnh nhân vào prompt hay nhận xét;
   phát hiện PHI → dừng và báo giảng viên.
4. **Thay đổi hạ tầng** (`scripts/`, `rubric/`, `.github/`, `.agents/`) đi
   qua nhánh + PR do giảng viên duyệt (CODEOWNERS). Sửa `rubric/` → chạy
   `python scripts/review_project.py --self-test` và
   `python -m unittest discover -s scripts/tests` trước khi mở PR.
