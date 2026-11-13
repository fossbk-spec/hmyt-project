# Báo cáo mốc m1: Kế hoạch đề tài phân loại AMD qua ảnh đáy mắt

Nhóm nhom13 | MSSV: 20233811

## 1. Bối cảnh y khoa
Thoái hóa hoàng điểm tuổi già (AMD) là bệnh của vùng hoàng điểm ở võng mạc,
gây giảm thị lực trung tâm ở người lớn tuổi. Bệnh thường khởi đầu ở thể khô
với các đám drusen, có thể tiến triển thành teo địa hoặc chuyển sang thể ướt
với tân mạch và xuất huyết. Phát hiện sớm giúp theo dõi và can thiệp kịp thời.

## 2. Vấn đề học máy
Bài toán phân loại ảnh đáy mắt nhiều lớp (bình thường, trung gian, teo địa,
AMD ướt). Thách thức chính: dữ liệu mất cân bằng giữa các lớp, một bệnh nhân có
hai mắt nên dễ rò rỉ dữ liệu nếu chia tập sai, và mô hình cần giải thích được
để bác sĩ tin cậy.

## 3. Dữ liệu
iChallenge-AMD và AREDS (xem chi tiết, tình trạng truy cập ở README).
Nhóm sẽ kiểm tra nhãn, số lượng ảnh và giấy phép sử dụng trước khi huấn luyện.

## 4. Phương pháp
Baseline ResNet-18, sau đó thử transfer learning và tăng cường dữ liệu.
Chia tập theo bệnh nhân. Đánh giá bằng accuracy, F1, AUC và ma trận nhầm lẫn.
Dùng Grad-CAM để kiểm tra mô hình có tập trung vào vùng tổn thương hay không.

## 5. Kế hoạch tiếp theo
- Hoàn tất xin quyền và tải dữ liệu.
- Phân tích dữ liệu và chia tập theo bệnh nhân.
- Huấn luyện baseline ResNet-18 và đánh giá.
- Thêm Grad-CAM và viết phân tích lỗi.

## 6. Tài liệu tham khảo
(Sẽ bổ sung các bài báo thật mà nhóm đã đọc, kèm trích dẫn đầy đủ.)