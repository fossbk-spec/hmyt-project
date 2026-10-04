---
name: kiem-tra-du-an
description: Kiểm tra 1 bài nộp đồ án môn học ET4248 (hmyt-project) theo requirements/rubric của project và đề xuất phương án nâng cao chất lượng
---

# Workflow: Kiểm tra dự án môn học của sinh viên

**Đầu vào:** thư mục `submissions/<topic_slug>_<ma_nhom>/` hoặc số PR; mốc cần
nghiệm thu (`m1`, `m2`, `m3`, `m4`, `final`); hạn nộp (tùy chọn).
**Đầu ra:** 1 bản nhận xét theo mẫu ở Bước 5 — giảng viên đọc, chỉnh, quyết định.

**Căn cứ (chỉ dùng các nguồn này, không tự đặt yêu cầu mới):**

- [Khung Quản Lý & Rubric 100 điểm](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc)
  — Phần II (4 milestone), III (rubric), IV (chính sách trừ điểm), VII (AI Disclosure).
- Trang đề tài trong ngân hàng (`https://fossbk-spec.github.io/hmyt-book/du_an/<topic_slug>`)
  — yêu cầu đặc thù đã mã hóa tại `rubric/topics/<topic_slug>.json`.
- Đề tài 3.4 dùng rubric lab riêng (`/du_an/suy_tim_risk_dxai_lab`, 5 tiêu chí × 20 điểm).

## Ba lớp kiểm tra

| Lớp | Công cụ | Khi nào | Kết quả |
|---|---|---|---|
| 1. Cổng bắt buộc | `scripts/validate_submission.py` (CI `validate-submission.yml`) | Tự động, mọi PR | ❌ = không được merge |
| 2. Kiểm tra theo rubric | `scripts/review_project.py` (CI `review-submission.yml`) | Tự động mọi PR + SV tự chạy trước khi nộp | Báo cáo ✅ ⚠️ ❌ 👤 + 3 tầng nâng cao + câu hỏi vấn đáp |
| 3. Review chuyên sâu | Workflow này (Claude Code/Antigravity hỗ trợ GV/TA) | Khi nghiệm thu mốc | Nhận xét theo mẫu, đề xuất mức rubric — GV quyết định |

Lớp 2 chỉ phát hiện **có/không có dấu hiệu** (phân tích tĩnh). Lớp 3 xác minh
dấu hiệu đó là thật, đánh giá **chất lượng**, và chấm các mục 👤 mà máy không
chấm được (tính sáng tạo, độ sâu y văn, khớp lâm sàng của XAI, vấn đáp).

## Nguyên tắc bắt buộc

1. **AI không chấm điểm chính thức**, không approve/merge, không "request
   changes" thay giảng viên — chỉ soạn nhận xét đề xuất.
2. **Zero-Hallucination:** mỗi nhận xét phải kèm bằng chứng `file:dòng` (hoặc
   ghi rõ "không tìm thấy trong bài nộp"). Không suy diễn nội dung chưa đọc;
   không tự thêm tác giả/năm/số liệu ngoài bài nộp và tài liệu căn cứ.
3. **An toàn:** không chạy code của sinh viên trên máy có token/SSH key/dữ
   liệu thật. Chỉ chạy lại (Bước 4.2) trong môi trường cô lập dùng xong bỏ.
4. **Quyền riêng tư:** không đưa dữ liệu bệnh nhân vào prompt. Nếu phát hiện
   PHI (`I.PHI` ❌): dừng, báo GV ngay, **không trích lại nội dung dữ liệu**
   trong nhận xét — chỉ ghi tên file và tên cột.
5. **Công bằng:** cùng rubric, cùng mốc, cùng phiên bản công cụ cho mọi nhóm
   (ghi version `review_project.py` vào nhận xét).
6. **Không sửa bài nộp** của sinh viên; mọi đề xuất sửa nằm trong nhận xét.

## Bước 0 — Lấy bài nộp (chỉ đọc)

```bash
git fetch origin pull/<N>/head:review/pr-<N>   # PR chưa merge
git switch review/pr-<N>
# hoặc bài đã merge: git switch main && git pull --ff-only
```

Xác định mốc: tham số GV đưa ra > trường `milestone` trong `submission.json` > `final`.

## Bước 1 — Cổng bắt buộc

```bash
python scripts/validate_submission.py submissions/<topic_slug>_<ma_nhom>
```

`[FAILED]` → dừng, chuyển thẳng Bước 5 với danh sách lỗi cần sửa (không review tiếp).

## Bước 2 — Kiểm tra tự động theo rubric

```bash
python scripts/review_project.py submissions/<topic_slug>_<ma_nhom> \
  --milestone <m1|m2|m3|m4|final> --online [--deadline YYYY-MM-DD] \
  --output review_<ma_nhom>.md --json review_<ma_nhom>.json
```

- `--online`: xác minh DOI qua Crossref, nhánh `gd1`–`gd7` của repo nhóm
  (`repo_link_optional`), so khớp với code minh họa công khai của đề tài.
- Cả lớp: `python scripts/review_project.py --all --output-dir review_reports`
  (hoặc chạy workflow **Review Submission (rubric)** bằng *Run workflow* trên GitHub).
- Đọc **Mục 2 (Vấn đề nghiêm trọng)** trước tiên.

## Bước 3 — Xác minh các mục ❌/⚠️ (loại dương tính giả, tìm âm tính giả)

Công cụ dùng heuristic, nên mọi ❌/⚠️ nghiêm trọng phải được người/AI mở đúng
`file:dòng` để xác nhận trước khi đưa vào nhận xét.

| Check | Cách xác minh | Âm tính giả cần tự tìm thêm |
|---|---|---|
| `M2.LEAK_*`, `T.STRAT_FOLD`, `T.NO_RANDOM_SPLIT` | Lần theo dữ liệu: nạp → chia tập → fit → đánh giá. Biến được fit là toàn bộ dữ liệu hay chỉ Train? | Chuẩn hóa thủ công `(X - X.mean()) / X.std()` trước khi chia; chọn đặc trưng theo tương quan với nhãn trên toàn bộ dữ liệu; augmentation/oversampling trước khi chia; cùng bệnh nhân ở 2 tập (2 mắt, nhiều lần khám); biến "tương lai" (vd `time` ở đề tài 3.4) |
| `M2.SPLIT_PATIENT` | Dataset có mã bệnh nhân không? 1 dòng/ảnh = 1 bệnh nhân? | Chia theo ảnh khi 1 bệnh nhân có nhiều ảnh |
| `I.PHI`, `I.DATA_LICENSE` | Mở header file (không chép dữ liệu ra nhận xét); kiểm tra giấy phép bộ dữ liệu | Ảnh có tên/ID bệnh nhân trong pixel hoặc tên file; header DICOM |
| `I.SIMILARITY` | So 2 nguồn cạnh nhau; tách boilerplate (nạp dữ liệu, vẽ hình) khỏi logic cốt lõi | Đổi tên biến nhưng giữ nguyên cấu trúc |
| `I.CITATION_INTEGRITY` | Tra DOI trên doi.org/Crossref: tiêu đề, tác giả, năm khớp `.bib`; số liệu trích dẫn khớp abstract | Trích dẫn đúng DOI nhưng sai số liệu/kết luận |
| `I.RESULT_TRACE` | Đối chiếu từng số trong bảng kết quả với output | Số trong hình/biểu đồ, số trong phần Tóm tắt |

Kết luận cho từng mục: **Xác nhận** / **Dương tính giả** (ghi lý do) / **Cần SV giải trình**.

## Bước 4 — Review chuyên sâu (mục 👤 và chất lượng)

1. **Bài toán & y văn (M1):** Input/Output và chi phí FP/FN có nhất quán với
   độ đo, ngưỡng được chọn ở M4 không? Bảng y văn có phân tích hạn chế và chỉ
   ra khoảng trống mà nhóm giải quyết không? Đối chiếu ≥ 3 bài ngẫu nhiên.
2. **Tái lập (FINAL.repro — khi nghi vấn số liệu hoặc ở mốc Chung kết):**
   chạy trong môi trường cô lập (container/VM/Codespace dùng xong xóa, không
   có credential), có giới hạn thời gian:
   ```bash
   python -m venv .venv-review && . .venv-review/bin/activate
   pip install -r submissions/<...>/requirements.txt
   timeout 1800 <lệnh tái lập trong README>
   ```
   So số liệu chạy lại với báo cáo; chênh lệch vượt mức làm tròn → ghi vào
   nhận xét kèm cả 2 giá trị.
3. **Mô hình đề xuất & ablation (M3):** cơ chế cải tiến có căn cứ không? Mỗi
   dòng ablation có đúng chỉ thay 1 thành phần? Có dấu hiệu quá khớp (Train ≫ Val)?
4. **Đánh giá & XAI (M4):** độ đo có phù hợp mất cân bằng nhãn và chi phí
   FN/FP? CI tính trên Test, lấy mẫu lại theo bệnh nhân? XAI: đủ ≥ 3 ca TP và
   ≥ 2 ca FP/FN, vùng nổi bật có khớp giải phẫu/sinh lý không?
5. **Bài báo (FINAL):** logic IMRAD, bảng/hình chuẩn, hạn chế trung thực,
   mục AI Disclosure + `docs/ai_disclosure_log.md`, CRediT.
6. **Câu hỏi vấn đáp:** lấy từ Mục 6 của báo cáo tự động, thêm 2–3 câu nhắm
   vào điểm yếu đã xác nhận ở Bước 3–4 (mỗi thành viên ít nhất 1 câu).

## Bước 5 — Soạn nhận xét theo mẫu

```markdown
## Nhận xét bài nộp `submissions/<topic_slug>_<ma_nhom>` — Mốc <M?>
Người soạn: <GV/TA/AI — ghi rõ> · Ngày: <YYYY-MM-DD> · Commit: <sha> · review_project.py v<x.y.z>

### 1. Kết luận nhanh
<2–3 câu: điểm mạnh nổi bật nhất, vấn đề lớn nhất, có dính chính sách trừ điểm (Phần IV) không>

### 2. Bảng phát hiện
| # | Mức độ | Tiêu chí rubric | Phát hiện | Bằng chứng (file:dòng) | Đề xuất sửa |
|---|---|---|---|---|---|
| 1 | Nghiêm trọng / Quan trọng / Nhỏ | M2 — Zero-Leakage (8 đ) | ... | `src/a.py:42` | ... |

### 3. Kết quả xác minh mục tự động
| Check | Kết luận (Xác nhận / Dương tính giả / Cần giải trình) | Ghi chú |

### 4. Phương án nâng cao chất lượng
- **Tầng 1 — Sửa ngay:** ...
- **Tầng 2 — Lên mức Xuất sắc của rubric:** ...
- **Tầng 3 — Vượt rubric (chọn 2–3):** ...

### 5. Câu hỏi vấn đáp gợi ý
1. ...

### 6. Đề xuất mức rubric (tham khảo — giảng viên quyết định)
| Tiêu chí | Mức đề xuất (Xuất sắc / Đạt / Cần cải thiện) | Lý do ngắn |
```

## Bước 6 — Bàn giao

- Lưu nhận xét thành file, gửi GV. GV chỉnh sửa và quyết định điểm.
- Chỉ đăng lên PR khi GV đồng ý: 1 comment dạng *review COMMENT* (không
  APPROVE / REQUEST_CHANGES), ghi rõ "Nhận xét do AI soạn, giảng viên đã duyệt".
- Không merge. Bài đã merge cần nộp lại → trao đổi trực tiếp với GV (CONTRIBUTING Mục 4).

## Phương án nâng cao chất lượng

### Cho từng dự án (tự động trong Mục 5 của báo cáo `review_project.py`)

| Tầng | Nội dung | Nguồn |
|---|---|---|
| 1 — Sửa ngay | Mọi tiêu chí ❌, xếp theo mức nghiêm trọng (rò rỉ dữ liệu, PHI, sao chép trước) | Rubric mức "Cần cải thiện" + Phần IV |
| 2 — Lên mức Xuất sắc | Mọi tiêu chí ⚠️, kèm hành động cụ thể | Cột "Xuất sắc (90–100%)" của rubric |
| 3 — Vượt rubric | Hướng nâng cao chưa thấy trong bài (CV lặp, kiểm định thống kê, ngưỡng theo chi phí lâm sàng, phân tích nhóm con, external validation, bất định, checklist TRIPOD+AI/CLAIM, Model Card...) + gợi ý riêng của trang đề tài | `rubric/improvements.json`, `rubric/topics/*.json` |

### Cho cả lớp (quy trình)

1. **Kiểm tra sớm, liên tục:** SV ghi `"milestone": "m1"`…`"final"` trong
   `submission.json` và chạy `review_project.py` trước mỗi mốc — chỉ tiêu chí
   đến mốc đó được đánh giá, lỗi rò rỉ dữ liệu bị bắt từ M2 thay vì phát hiện
   ở buổi bảo vệ.
2. **Bảng tổng hợp lớp:** sau mỗi hạn nộp, chạy workflow *Review Submission
   (rubric)* với `folder` để trống → bảng xếp các nhóm theo số vấn đề nghiêm
   trọng; lỗi lặp lại ở nhiều nhóm → bổ sung 15 phút giảng lại đúng chủ đề.
3. **Review chéo giữa các nhóm:** mỗi nhóm dùng báo cáo tự động + mẫu nhận
   xét ở Bước 5 để review PR của 1 nhóm khác (kết quả đưa vào đánh giá đồng
   đẳng, Phần V).
4. **Vấn đáp dựa trên bằng chứng:** dùng câu hỏi gợi ý (Mục 6 báo cáo) để
   kiểm tra mọi thành viên hiểu đúng phần mình làm (Zero Freerider).
5. **Hiệu chỉnh công cụ mỗi học kỳ:** ghi lại các dương tính/âm tính giả gặp
   ở Bước 3 thành Issue; cập nhật `rubric/` (xem `rubric/README.md`) và chạy
   `python scripts/review_project.py --self-test` trước khi merge.
