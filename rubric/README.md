# `rubric/` — cấu hình kiểm tra bài nộp theo rubric

Dữ liệu cho `scripts/review_project.py`. Sửa JSON ở đây để thay đổi **cái gì
được kiểm tra**; chỉ sửa Python khi cần một *loại* phân tích mới.

| File | Nội dung | Căn cứ |
|---|---|---|
| `rubric.json` | Các rubric (`standard` 100 điểm theo 4 milestone + chung kết; `lab_3_4` cho đề tài 3.4 cấp tốc): tiêu chí, điểm, mô tả mức Xuất sắc, và danh sách check tự động thuộc mỗi tiêu chí | [Khung Quản Lý & Rubric](https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc), [Lab 3.4](https://fossbk-spec.github.io/hmyt-book/du_an/suy_tim_risk_dxai_lab) |
| `topics/<topic_slug>.json` | Hồ sơ từng đề tài: loại dữ liệu, loại bài toán (bộ độ đo), cách chia tập, XAI phù hợp, mốc y văn, file bàn giao, yêu cầu đặc thù, gợi ý nâng cao, câu hỏi vấn đáp riêng | Trang đề tài trong ngân hàng (`docs/du_an/<slug>.md` của repo HMYT) |
| `improvements.json` | Tầng 3 "Vượt rubric" (chỉ gợi ý khi chưa thấy dấu hiệu trong bài) + câu hỏi vấn đáp theo kỹ thuật nhóm đã dùng | Thực hành chuẩn ML y tế; tham chiếu chương sách HMYT |

> Mọi yêu cầu trong `topics/*.json` phải truy được về trang đề tài (trường
> `source`). Không thêm yêu cầu mới chỉ ở đây — cập nhật trang đề tài trước.

## Thêm/sửa 1 yêu cầu đặc thù của đề tài

Thêm 1 phần tử vào `checks` của `topics/<slug>.json`:

```json
{
  "id": "T.STRAT_FOLD",
  "title": "Chia tập theo cột `strat_fold` có sẵn của PTB-XL",
  "milestone": "M2",
  "criterion": "M2.leakage",
  "where": "code",
  "any_of": ["strat_fold"],
  "severity": "critical",
  "suggestion": "Train = strat_fold 1–8, Val = 9, Test = 10 ...",
  "source": "HMYT docs/du_an/roi_loan_nhip_tim.md"
}
```

| Trường | Ý nghĩa |
|---|---|
| `id` | Duy nhất trong hồ sơ, bắt đầu bằng `T.` |
| `milestone` | `M0`…`M4`, `FINAL` — chỉ đánh giá khi bài nộp đã tới mốc này (rubric `standard`) |
| `criterion` | Id tiêu chí trong `rubric.json` để gộp vào; dạng `{"lab_3_4": "L3", "standard": "M3.proposed"}` nếu khác nhau theo rubric. Không khớp → hiện ở mục "Yêu cầu đặc thù của đề tài" |
| `where` | `code` (.py + cell code notebook, bỏ comment và dòng import), `report` (thư mục `report/`…), `text` (báo cáo + README + markdown notebook), `any` (code + báo cáo), `files` (đường dẫn file) |
| `any_of` | Danh sách regex (mặc định không phân biệt hoa thường; `"case_sensitive": true` để phân biệt) |
| `min_distinct` | Số regex khác nhau phải khớp (mặc định 1) |
| `type` | `require` (mặc định) hoặc `forbid` — khớp là vi phạm, trừ khi `unless_any` cũng khớp (khi đó trạng thái = `unless_status`, mặc định PASS) |
| `severity` | `critical` / `major` → ❌; `minor` → ⚠️ (đổi bằng `fail_status`) |
| `suggestion`, `penalty`, `fail_detail` | Văn bản hiển thị trong báo cáo |

Các trường cấp hồ sơ: `modalities` (`signal`/`image`/`tabular`/`nlp`),
`metric_set` (`binary`/`multiclass`/`multilabel`/`segmentation`/`ner`/`regression`
hoặc danh sách), `split.one_row_per_patient`, `xai_expected` (khóa trong
`XAI_METHODS`), `xai_mandatory`, `expected_files`, `reference_code_urls`
(code minh họa công khai để so trùng lặp khi chạy `--online`),
`overrides.min_references`, `overrides.tracking_optional`, `overrides.report_sections`.

## Sau khi sửa

```bash
python scripts/review_project.py --self-test            # id/tiêu chí/regex/tổng điểm hợp lệ
python -m unittest discover -s scripts/tests            # bộ test
python scripts/review_project.py --all --output-dir /tmp/review   # chạy thử trên bài nộp hiện có
```

CI `review-tool-selftest.yml` chạy lại 3 lệnh này trên mọi PR sửa `rubric/` hoặc `scripts/`.
