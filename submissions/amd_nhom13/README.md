# Đề tài 2.4: Phân loại Thoái hóa hoàng điểm tuổi già (AMD) qua ảnh đáy mắt

Nhóm: nhom13 | Mốc nộp: m1 (kế hoạch đề tài)

## Mục tiêu
Xây dựng mô hình học sâu phân loại ảnh đáy mắt theo các mức:
bình thường / trung gian / teo địa / AMD ướt.

## Dữ liệu dự kiến
- iChallenge-AMD (công khai): <điền link tải>
- AREDS: cần xin quyền truy cập, đang tìm hiểu
Không đưa ảnh vào repo, chỉ để link.

## Phương pháp dự kiến
- Baseline: ResNet-18.
- Chia tập train/val/test theo bệnh nhân (không để 2 mắt của cùng một
  người nằm ở hai tập khác nhau).
- Độ đo: accuracy, F1, AUC từng lớp.
- Giải thích mô hình: Grad-CAM.

## Tiến độ hiện tại
Mới hoàn thành kế hoạch, chưa có kết quả thực nghiệm.