# HMYT Project — Repo Nộp Bài Chung Của Lớp

Repo này là nơi **sinh viên môn Học Máy trong Y Tế (ET4248)** nộp bài đồ án
theo đề tài đã chọn trong [Ngân hàng đề tài](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc).

- **Đề bài, y văn, workflow chi tiết:** xem tại
  [hmyt-book](https://fossbk-spec.github.io/hmyt-book/) — repo này **không**
  chứa lại đề bài, chỉ chứa bài làm của sinh viên.
- **Mỗi nhóm = 1 thư mục riêng** dưới `submissions/`, đặt tên theo đúng quy
  ước ở [CONTRIBUTING.md](CONTRIBUTING.md).
- **Nộp bài qua Pull Request** — không push trực tiếp vào `main`. Xem quy
  trình đầy đủ trong [CONTRIBUTING.md](CONTRIBUTING.md).

## Vì sao dùng Pull Request thay vì push trực tiếp?

1. **Cách ly giữa các nhóm** — mỗi PR chỉ được phép đổi file trong đúng 1
   thư mục `submissions/<đề-tài>_<mã-nhóm>/` của nhóm đó. CI tự động chặn
   PR nếu đụng vào thư mục của nhóm khác (xem
   `.github/workflows/validate-submission.yml`).
2. **Không mất bài** — lịch sử Git giữ nguyên toàn bộ commit của từng nhóm;
   `main` chỉ nhận bản đã qua kiểm tra tự động + giảng viên duyệt.
3. **Không nộp nhầm/nộp thiếu** — CI kiểm tra cấu trúc thư mục, file bắt
   buộc (`submission.json`, `README.md`), và chặn các mẫu hình dữ liệu
   nhạy cảm phổ biến (khóa API, thông tin bệnh nhân thật) trước khi merge.

## Kiểm tra bài nộp: 3 lớp

| Lớp | Công cụ | Vai trò |
|---|---|---|
| 1. Cổng bắt buộc | `scripts/validate_submission.py` (CI `validate-submission.yml`) | Cách ly thư mục, schema, secret, file nặng — ❌ thì không merge |
| 2. Kiểm tra theo rubric | `scripts/review_project.py` (CI `review-submission.yml`) | Đối chiếu bài nộp với [rubric 100 điểm](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc) + yêu cầu riêng của từng đề tài: nghi vấn rò rỉ dữ liệu, độ đo, XAI, tái lập, liêm chính (PHI, trích dẫn, số liệu, sao chép) và **3 tầng phương án nâng cao chất lượng**. Tư vấn — không chặn merge, không phải điểm |
| 3. Review chuyên sâu | `.agents/workflows/kiem-tra-du-an.md` (Claude Code: `/kiem-tra-du-an`) | Giảng viên/TA (có AI hỗ trợ) xác minh kết quả tự động, chấm các mục cần con người, soạn nhận xét theo mẫu |

Sinh viên tự chạy lớp 2 trước khi nộp:

```
python scripts/review_project.py submissions/<topic_slug>_<ma_nhom>            # mốc theo submission.json
python scripts/review_project.py submissions/<topic_slug>_<ma_nhom> --milestone m2 --online
```

Giảng viên review cả lớp: `python scripts/review_project.py --all --output-dir review_reports`
hoặc chạy workflow **Review Submission (rubric)** (Actions → Run workflow, để trống `folder`).

## Cấu trúc repo

```
hmyt-project/
├── submissions/
│   ├── _TEMPLATE/                  ← Copy thư mục này để bắt đầu nộp bài
│   │   ├── submission.json         ← Thông tin nhóm + đề tài (bắt buộc)
│   │   ├── README.md               ← Tóm tắt bài làm (bắt buộc)
│   │   ├── report/                 ← Báo cáo (PDF/Markdown)
│   │   ├── src/                    ← Mã nguồn
│   │   └── slides/                 ← Slide thuyết trình (tuỳ chọn)
│   │
│   └── <topic_slug>_<ma_nhom>/     ← 1 thư mục nộp bài của 1 nhóm
│       └── ... (giống cấu trúc _TEMPLATE)
│
├── scripts/
│   ├── validate_submission.py      ← Cổng bắt buộc của CI
│   ├── review_project.py           ← Kiểm tra theo rubric + gợi ý nâng cao
│   └── tests/                      ← Test của công cụ review
├── rubric/                         ← Cấu hình rubric, hồ sơ 15 đề tài, gợi ý nâng cao
├── .github/
│   ├── workflows/validate-submission.yml
│   ├── workflows/review-submission.yml
│   ├── workflows/review-tool-selftest.yml
│   └── PULL_REQUEST_TEMPLATE.md    ← Mẫu PR + Biomedical Reproducibility Checklist
├── .agents/workflows/kiem-tra-du-an.md  ← Quy trình review chuyên sâu (GV/TA + AI)
├── AGENTS.md / CLAUDE.md           ← Chính sách cho agent AI trong repo này
├── CONTRIBUTING.md                 ← Hướng dẫn nộp bài chi tiết từng bước
└── CODEOWNERS                      ← Định tuyến review PR tới giảng viên/TA
```
