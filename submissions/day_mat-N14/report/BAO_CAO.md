# Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt (Đề tài 2.2 — day_mat)

## 1. Đặt vấn đề

**Câu hỏi nghiên cứu:** Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt bằng học sâu, có giải thích được (XAI)

**Ý nghĩa lâm sàng:** Tích hợp AI và kỹ thuật giải thích trực quan (Grad-CAM) giúp các cơ sở y tế tuyến dưới tầm soát nhanh với chi phí thấp, đồng thời cung cấp bằng chứng vùng tổn thương để bác sĩ dễ dàng kiểm chứng và đưa ra quyết định điều trị.

**Dataset:** Bộ dữ liệu A Brazilian Multilabel Ophthalmological Dataset (BRSET) 1.0.1 (https://www.kaggle.com/datasets/tanzinabdul/fundus-patientwise-split)
— Tổng quy mô: 16.266 ảnh chụp / 8.524 bệnh nhân độc lập.

---

## 2. Y văn đối chiếu (2 bài trong đề tài 2.2)

| Citekey | Phương pháp | Kết quả công bố |
|---|---|---|
| `dutta2026vgretfound` | RETFound (ViT foundation model) + vessel-graph learning + cross-modal fusion; BRSET chỉ dùng làm external validation độc lập | Accuracy 94,81% · F1 94,25% · MCC 93,09% · Kappa 92,94% |
| `khosravi2024external` | FastVit_SA12 và ResNet18; phân loại nguyên nhân xuất huyết võng mạc (traumatic vs. medical); BRSET là một trong các bộ dữ liệu external validation | ResNet18: Accuracy 94,66% |

### 2.1. Đọc kỹ full-text `dutta2026vgretfound` trước khi so sánh

Đọc lướt abstract dễ hiểu nhầm đây là một thí nghiệm “train trên BRSET rồi test trên BRSET” — thực ra **BRSET chỉ được dùng làm external validation độc lập, hoàn toàn không tham gia huấn luyện, không augmentation, không tinh chỉnh siêu tham số** . Quy trình được mô tả rất rõ:

- **Tập phát triển (development):** RFMiD (3.200 ảnh) + ODIR-5K (~10.000 ảnh), được harmonize về 6 lớp độc quyền: Normal, **Diabetic Retinopathy (DR)**, AMD, Pathological Myopia, Hypertensive Retinopathy, Other Retinal Diseases .

- **Tập external validation:** BRSET (16.266 ảnh), sau harmonize nhãn được lấy 20% stratified subset = 2.236 ảnh, trong đó chỉ có 209 ảnh thuộc lớp DR . Tập này **không hề được dùng để train, augmentation, hay model selection .**

- **Kiến trúc:** RETFound (retinal foundation model dựa trên ViT) học biểu diễn toàn cục; vessel-graph learning học cấu trúc mạch máu; cross-modal fusion kết hợp hai nguồn thông tin này trước khi đưa ra quyết định .

- **Đánh giá:** Báo cáo **mean ± SD** trên cùng một tập BRSET 20% cố định, qua nhiều lần khởi tạo mô hình độc lập — không phải chia lại fold trên BRSET .

**Điểm cần lưu ý:** DR chỉ là một trong 6 lớp, và trong tập external BRSET, DR chỉ chiếm 209/2.236 ảnh (~9,3%) . Vì vậy, con số F1 94,25% là macro/weighted F1 trên toàn bộ 6 lớp, không phải F1 riêng cho lớp DR. Đây là khác biệt căn bản so với một bài toán “phân loại mức độ DR” thuần túy (binary hoặc 5-class severity). Khi đối chiếu với kết quả của nhóm, cần tránh so sánh trực tiếp con số 94,25% với F1 của một mô hình chỉ phân loại DR.


---

## 3. Phương pháp

- **Chia dữ liệu:** Sử dụng giao thức chia tập **Patient-level Split** (đảm bảo không rò rỉ dữ liệu bệnh nhân giữa các tập). Tập dữ liệu BRSET gồm: Train (13.015 ảnh), Val (1.632 ảnh), Test (1.619 ảnh). Kiểm tra chéo xác nhận 0 bệnh nhân trùng lặp giữa 3 tập
- **Tiền xử lý:** Tự động cắt viền đen (Auto-Crop ROI), tăng tương phản cục bộ bằng CLAHE trên không gian màu LAB, đồng bộ kích thước về 384x384 và chuẩn hóa (Normalize) theo chuẩn **ImageNet**.
- **Xử lý mất cân bằng::** Do phân bố nhãn cực kỳ mất cân bằng (Nhãn 0 chiếm ~93%, Nhãn 3 chỉ có 78 ảnh), sử dụng hàm mất mát **Weighted Cross-Entropy** (tính toán trọng số lớp đảo nghịch tần suất) để ép mô hình chú ý vào các lớp bệnh hiếm.
- **Baseline:** Sử dụng 'ResNet-18' (pre-trained ImageNet) làm mô hình cơ sở.
- **Huấn luyện:** Tối ưu hóa AdamW, điều chỉnh Learning Rate (CosineAnnealingLR), kết hợp Mixed Precision (AMP) để tăng tốc và Early Stopping (Patience=5) để chống 'Overfitting'. Sử dụng Data Augmentation (lật, xoay, biến đổi màu sắc).
- **XAI:** Sử dụng Grad-CAM trên lớp tích chập cuối cùng để trực quan hóa vùng ảnh mà mô hình đang tập trung chú ý.
---

## 4. Kết quả

### 4.1. Baseline ResNet-18
- Đạt **Val Accuracy** 94.36%.
- **Đánh giá y tế (Test set):** Mô hình đạt độ nhạy (Sensitivity) rất cao ở nhãn 0 (không bệnh) và nhãn 4 (tăng sinh). Tuy nhiên, do mất cân bằng dữ liệu, mô hình dự đoán kém chính xác ở nhãn 3 (chỉ 1/6 ca đoán đúng).
- **Phân tích FP/FN:** Mô hình có xu hướng dự đoán nhầm nhãn 0 thành nhãn 1 hoặc nhãn 4 (Báo động giả - FP) thay vì bỏ sót bệnh nặng thành không bệnh (Bỏ sót - FN). Điều này phù hợp với nguyên tắc "thà bắt nhầm còn hơn bỏ sót" trong y tế.
---

## 5. Khả năng giải thích (XAI - Grad-CAM)
- **Phát hiện Shortcut Learning:** Bản đồ nhiệt Grad-CAM cho thấy ở nhãn 0 (Không bệnh), AI có xu hướng nhìn vào viền đen/ánh sáng ở rìa ảnh thay vì vùng võng mạc trung tâm.

- **Đánh giá lâm sàng:** Ở các nhãn bệnh nặng (3, 4), vùng đỏ (vùng AI chú ý) tập trung chính xác vào các đốm xuất huyết và vi phình mạch. Điều này chứng minh AI thực sự đã học được đặc trưng bệnh lý y khoa, không chỉ học vẹt hình dáng ảnh.

---

## 6. Hạn chế & Hướng phát triển

1. **Mất cân bằng dữ liệu cực đoan:** Nhãn 3 (Nặng) chỉ có 78 ảnh (chiếm <0.5%), dẫn đến kết quả dự đoán nhãn này trên tập Test rất thấp. Đây là **hạn chế cố hữu của dữ liệu BRSET**, không thể khắc phục hoàn toàn bằng Weighted Loss mà cần thêm dữ liệu thực tế.
2. **Hiện tượng Shortcut Learning (Học đường tắt):** Qua Grad-CAM, mô hình nhãn 0 bị phụ thuộc vào viền ảnh ngoài. Trong tương lai cần áp dụng thêm kỹ thuật **Circular Masking** (tạo mặt nạ đen che viền) để ép AI chỉ nhìn vào trung tâm võng mạc.
3. **Chưa hoàn thành đối chứng đa mô hình:** Do giới hạn về thời gian và tài nguyên tính toán, đề tài hiện tại mới dừng lại ở việc tối ưu mô hình Baseline ResNet-18. Việc thử nghiệm và so sánh với các kiến trúc chuyên sâu hơn như DenseNet-121 và EfficientNet sẽ là hướng phát triển tiếp theo để đánh giá toàn diện hơn.
4. **Dữ liệu đơn nguồn:** Tập dữ liệu BRSET chỉ thu thập tại Brazil với một loại máy chụp (Canon CR-2). Mô hình chưa được đánh giá chéo (external validation) trên các dân số khác hoặc các dòng máy khác.
5. **Chưa triển khai Segmentation:** Do BRSET không cung cấp nhãn Mask điểm ảnh (pixel-level), đề tài chưa thể triển khai mô hình U-Net để khoanh vùng chính xác từng tổn thương, mà chỉ dừng ở mức phân loại ảnh (Classification).
