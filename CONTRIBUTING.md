# Hướng Dẫn Nộp Bài (dành cho sinh viên)

## 0. Quy ước đặt tên thư mục nộp bài

```
submissions/<topic_slug>_<ma_nhom>/
```

- `topic_slug` — đúng tên đề tài trong
  [ngân hàng đề tài](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc)
  (vd `suy_tim`, `day_mat`, `suy_tim_risk_dxai`...). Xem cột "Đề tài" ở đó
  hoặc hỏi giảng viên nếu không chắc slug chính xác.
- `ma_nhom` — mã nhóm giảng viên cấp (vd `nhom01`) — **không dùng họ tên
  thật hay MSSV** trong tên thư mục (xem Mục 3 về quyền riêng tư).

Ví dụ: `submissions/suy_tim_risk_dxai_nhom07/`.

**Quan trọng:** 1 Pull Request chỉ được đụng tới ĐÚNG 1 thư mục
`submissions/<topic_slug>_<ma_nhom>/` của nhóm bạn. CI sẽ tự động FAIL nếu
PR chứa thay đổi ở bất kỳ đường dẫn nào khác (kể cả sửa file gốc của repo
này) — đây là cơ chế bắt buộc để nộp bài của nhóm A không bao giờ vô tình
ghi đè hay xung đột với nhóm B.

## 1. Các bước nộp bài

1. **Fork** repo này về tài khoản GitHub cá nhân (hoặc dùng chung 1 fork
   cho cả nhóm — 1 thành viên đại diện fork, thêm thành viên còn lại làm
   collaborator trên fork đó).
2. **Tạo branch riêng** từ `main`, đặt tên `submit/<topic_slug>-<ma_nhom>`:
   ```
   git checkout -b submit/suy_tim_risk_dxai-nhom07
   ```
3. **Copy thư mục mẫu:**
   ```
   cp -r submissions/_TEMPLATE submissions/suy_tim_risk_dxai_nhom07
   ```
4. **Điền `submission.json`** (xem schema Mục 2) và các file trong `report/`,
   `src/`, `slides/`.
5. **Chạy kiểm tra cục bộ trước khi nộp** (bắt buộc — CI sẽ chạy lại đúng
   script này, nộp trước khi tự kiểm tra chỉ tốn thời gian chờ CI fail):
   ```
   python scripts/validate_submission.py submissions/suy_tim_risk_dxai_nhom07
   ```
   Sau đó **tự kiểm tra theo rubric** (khuyến nghị mạnh — chỉ cần Python 3,
   không cài thêm gì; xem Mục 5):
   ```
   python scripts/review_project.py submissions/suy_tim_risk_dxai_nhom07
   ```
6. **Commit + push** lên fork của bạn, rồi **mở Pull Request** vào
   `main` của `fossbk-spec/hmyt-project` — điền mẫu PR (có Biomedical
   Reproducibility Checklist).
7. Chờ CI chạy xanh (✅) — nếu đỏ (❌), đọc log lỗi, sửa, push tiếp lên
   cùng branch (PR tự cập nhật, không cần mở PR mới).
8. Giảng viên/TA review nội dung và merge. Sau khi merge, bài nộp của nhóm
   bạn chính thức nằm trong lịch sử `main` — **không sửa được nữa qua PR
   thường** (mọi cập nhật sau merge coi như nộp lại, cần trao đổi trực tiếp
   với giảng viên).

## 2. Schema `submission.json`

```json
{
  "topic_slug": "suy_tim_risk_dxai",
  "topic_id": "3.4",
  "group_code": "nhom07",
  "members": ["MSSV1", "MSSV2", "MSSV3"],
  "submitted_at": "2026-12-15",
  "milestone": "final",
  "repo_link_optional": "https://github.com/<fork>/hmyt-project (nếu code chính nằm ở fork riêng, không copy hết vào đây)"
}
```

Trường bắt buộc: `topic_slug`, `topic_id`, `group_code`, `members`.
`members` dùng **MSSV**, không dùng họ tên đầy đủ (xem Mục 3).
`milestone` (tùy chọn: `m1`, `m2`, `m3`, `m4`, `final` — mặc định `final`) cho
công cụ review biết chỉ đánh giá các tiêu chí đến mốc đó.

## 3. Quy tắc bắt buộc về quyền riêng tư (CI sẽ chặn nếu vi phạm)

1. **Không đưa dữ liệu bệnh nhân thật** vào bất kỳ đâu trong PR — dùng dữ
   liệu công khai (MIMIC/PTB-XL/PhysioNet...) hoặc dữ liệu tự mô phỏng
   (synthetic). Đây là nguyên tắc xuyên suốt của toàn bộ ngân hàng đề tài.
2. **Không commit khóa API, token, mật khẩu, file `.env`.**
3. **Không dùng họ tên thật đầy đủ** của thành viên trong code/README/tên
   file — dùng MSSV. Tránh rủi ro khi repo này công khai vĩnh viễn trên
   GitHub.
4. **Không commit file dữ liệu lớn** (>10MB) — dùng link Google
   Drive/HuggingFace/Kaggle trong `README.md` thay vì đẩy thẳng vào Git.
5. `scripts/validate_submission.py` chạy vài kiểm tra mẫu hình cơ bản
   (regex cho khóa API dạng phổ biến, file `.env`, MSSV/tên trong
   `submission.json` không phải placeholder) — đây là **lưới an toàn cuối**,
   không thay thế trách nhiệm tự rà soát của nhóm trước khi nộp.

## 4. Câu hỏi thường gặp

**Check "Review Submission (rubric)" báo nhiều ❌/⚠️, có bị trừ điểm không?**
Không trực tiếp — đó là kiểm tra tự động mang tính tư vấn (xem Mục 5). Điểm
do giảng viên chấm theo rubric. Nhưng các mục ❌ thường trùng đúng chỗ bị
trừ điểm, nên sửa trước khi nghiệm thu; nếu cho là dương tính giả, giải
trình trong mô tả PR.

**Nhóm em có thể sửa bài sau khi đã merge không?**
Không tự sửa qua PR thường (branch protection chặn ghi đè thư mục nhóm
khác nhưng cũp chặn cả việc bạn tự ý sửa thư mục của chính mình sau khi đã
merge, để giữ tính toàn vẹn lịch sử nộp bài) — trao đổi trực tiếp với
giảng viên nếu cần nộp bổ sung/nộp lại.

**2 nhóm có thể chọn cùng 1 đề tài không?**
Có — mỗi nhóm vẫn có thư mục riêng (`ma_nhom` khác nhau), không xung đột.

**Sao không cho push thẳng vào `main`?**
Vì `main` được bảo vệ (branch protection) — chỉ nhận thay đổi qua PR đã
qua CI xanh, tránh 1 nhóm vô tình (hoặc cố ý) ghi đè bài của nhóm khác.

## 5. Tự kiểm tra theo rubric và nâng cao chất lượng

`scripts/review_project.py` đối chiếu bài nộp với
[Khung Quản Lý & Rubric](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc)
và yêu cầu riêng của đề tài (`rubric/topics/<topic_slug>.json`). Công cụ chỉ
**đọc** file (không chạy code của nhóm) và in báo cáo gồm:

1. **Tóm tắt theo tiêu chí rubric** — 🟢/🟡/🔴 là *dự báo* dựa trên dấu hiệu
   có/không, không phải điểm.
2. **Vấn đề nghiêm trọng** — nghi vấn rò rỉ dữ liệu (scaler/SMOTE fit trước
   khi chia tập, chia tập không theo bệnh nhân), dữ liệu định danh, trùng lặp
   cao với bài khác/code minh họa.
3. **Chi tiết** — từng tiêu chí kèm `file:dòng` làm bằng chứng.
4. **Liêm chính** — PHI, giấy phép dữ liệu, trích dẫn truy vết được (citekey ∈
   `.bib`, DOI có thật với `--online`), số liệu báo cáo khớp output của code.
5. **Phương án nâng cao chất lượng** — Tầng 1 sửa ngay (❌), Tầng 2 lên mức
   Xuất sắc (⚠️), Tầng 3 vượt rubric (hướng tới bài báo) + gợi ý riêng của đề tài.
6. **Câu hỏi vấn đáp gợi ý** — luyện trước buổi bảo vệ.

Tùy chọn hữu ích: `--milestone m1|m2|m3|m4|final`, `--online` (xác minh DOI
qua Crossref, nhánh gd1–gd7 trên repo nhóm), `--output bao_cao_review.md`.
**Không commit file báo cáo review vào thư mục nộp bài.**
