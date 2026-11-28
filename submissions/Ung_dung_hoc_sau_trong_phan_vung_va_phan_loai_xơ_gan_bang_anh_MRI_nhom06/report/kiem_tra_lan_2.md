# Bài kiểm tra lần 2 — Nhóm 06 — Ứng dụng học sâu trong phân vùng và phân loại xơ gan bằng ảnh MRI

Fork: https://github.com/nguyenthuylinh0903m-lang/hmyt-project · Nhánh: gd5-proposed-model · PR: https://github.com/fossbk-spec/hmyt-project/pull/25

Thành viên (MSSV hoặc alias) và vai trò:

* 20233772 — Đồng thực hiện; tham gia xử lý dữ liệu, huấn luyện và đánh giá U-Net, xây dựng mô hình phân loại và hoàn thiện báo cáo. Phân công chi tiết giữa hai thành viên chưa được ghi chính thức trong tài liệu hiện có.
* 20233735 — Đồng thực hiện; tham gia tổng quan y văn, xử lý dữ liệu, đánh giá mô hình và hoàn thiện báo cáo. Phân công chi tiết giữa hai thành viên chưa được ghi chính thức trong tài liệu hiện có.

**Bài báo nền của đề tài:** CirrGuide (Raju & Ma, 2026), nhóm tái hiện đúng pipeline của bài này (xem mục C2).

---

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm

| Nguồn | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|-|-|-|-|-|
| PubMed | `("liver cirrhosis" OR "liver fibrosis") AND ("MRI" OR "magnetic resonance") AND ("deep learning" OR "convolutional neural network")` | 2018–2026 | [điền từ lần tìm 15/10/2026] | [điền] |
| PubMed | `("liver segmentation") AND ("U-Net") AND ("MRI")` | 2018–2026 | [24.600] | [6] |
| Google Scholar | `CirrMRI600+ liver cirrhosis T2-weighted classification` | 2022–2026 | [12] | [6] |
| arXiv | `liver cirrhosis stage MRI deep learning` | 2022–2026 | [17.000] | [7] |

Ngày tìm kiếm: [15/10/2026]. Bài CirrGuide (arXiv 2609.14010, nộp 12/09/2026) được đưa vào danh sách như bài nền của đề tài.

Tiêu chí chọn / loại bài:

* Chọn nghiên cứu dùng MRI để phát hiện hoặc phân giai đoạn xơ gan / xơ hóa gan, hoặc phân vùng gan trên MRI bằng học sâu.
* Ưu tiên bài dùng chuỗi xung T2W hoặc bộ dữ liệu CirrMRI600+, vì cùng dữ liệu với đề tài của nhóm.
* Ưu tiên bài có báo cáo Dice/IoU (phân vùng) hoặc Accuracy, Sensitivity, Specificity, AUC, F1 (phân loại).
* Ưu tiên nghiên cứu 3–5 năm gần nhất, giữ một số bài cũ có giá trị nền tảng hoặc so sánh trực tiếp.
* Loại nghiên cứu chỉ dùng siêu âm, CT hoặc xét nghiệm huyết thanh, không dùng ảnh MRI.
* Loại bài không có số liệu hiệu năng đủ để đối chiếu, và bản trùng lặp của cùng một nghiên cứu.

### A2. Ma trận y văn

| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? | Hạn chế chính |
|-|-|-|-|-|-|-|-|
| 1 | Raju & Ma, 2026, CirrGuide: A Deep Cascaded Framework for Liver Cirrhosis Segmentation and Severity Classification from T2-Weighted MRI. Workshop MLMI 2026, tổ chức cùng MICCAI 2026 | https://arxiv.org/abs/2609.14010 | CirrMRI600+, tập T2W 2D chia sẵn: 318 bệnh nhân xơ gan, chia 256/31/31 bệnh nhân (5.428 / 674 / 664 lát); nhãn mask gan và mức độ nhẹ/vừa/nặng | Hai giai đoạn: (1) ResNet50 encoder + Attention U-Net decoder cho ra mask mềm; (2) ResNet50 phân loại, kết hợp đặc trưng toàn cục (GAP, F3–F5) và đặc trưng vùng dẫn đường bằng mask (Mask-Guided Attention Pooling, F4–F5), qua MLP, 3 lớp | Phân vùng: Dice 0,8983; mIoU 0,8414. Phân loại theo lát: Acc 0,6958; macro F1 0,6155; QWK 0,7758; MAE 0,3102; F1 từng lớp: nhẹ 0,8608, nặng 0,6877, vừa 0,2979. Theo bệnh nhân (31 BN): Acc 0,7097; macro F1 0,5859 | Không (đánh giá trên tập test chính thức của cùng bộ dữ liệu) | Tập test chỉ 31 bệnh nhân, lớp vừa chỉ 6 bệnh nhân; không có nhóm khỏe; mask dự đoán chưa được thử với nhiễu; xử lý 2D từng lát, chưa dùng quan hệ giữa các lát; nhãn mức độ dựa trên đánh giá hình ảnh |
| 2 | Gupta et al., 2025, Advanced liver fibrosis detection using a two-stage deep learning approach on standard T2-weighted MRI | https://doi.org/10.1007/s00261-025-05160-y | CirrMRI600+ (n = 374: 318 bệnh nhân xơ gan, 56 người khỏe mạnh) và dữ liệu nội bộ (n = 187 bệnh nhân mắc bệnh gan mạn tính) | Hai giai đoạn: nnU-Net phân vùng gan + Masked Attention ResNet phân loại xơ hóa tiến triển (METAVIR ≥ F3) | Dice 0,960 ± 0,009; IoU 0,923 ± 0,016; AUC 0,811; Acc 0,783; Sens 0,778; Spec 0,800 (tập kiểm tra n = 23) | Không (có tập kiểm tra độc lập nội bộ n = 23, chưa chứng minh kiểm định ngoài trung tâm) | Tập kiểm tra nhỏ; hiệu suất phân loại phụ thuộc chất lượng phân đoạn gan; chưa đánh giá trên nhiều trung tâm độc lập |
| 3 | Nowak et al., 2021, Detection of liver cirrhosis in standard T2-weighted MRI using deep transfer learning, European Radiology | https://doi.org/10.1007/s00330-021-07858-1 | 713 bệnh nhân tại một bệnh viện (553 xơ gan, 160 nhóm chứng); MRI T2W, một lát tại mức thùy đuôi | U-Net với encoder ResNet34 phân vùng gan + ResNet50 phân loại xơ gan | Dice > 0,98; Acc test = 0,96 khi có phân vùng và 0,95 khi không phân vùng; Sens = 1,00; Spec = 0,83 | Không (chia train/validation/test nội bộ 70/15/15) | Nghiên cứu một trung tâm; dùng một lát MRI chọn thủ công; chỉ phát hiện xơ gan, chưa phân loại các mức độ |
| 4 | Jha et al., 2025, Large Scale MRI Collection and Segmentation of Cirrhotic Liver (CirrMRI600+), Scientific Data | https://doi.org/10.1038/s41597-025-05201-7 | CirrMRI600+: 628 scan MRI từ 339 bệnh nhân (310 T1W, 318 T2W), gần 40.000 lát có nhãn phân vùng gan | Benchmark 11 mô hình phân vùng 3D và 6 mô hình 2D; gồm nnUNet, Attention U-Net, nnFormer, nnSynergyNet3D, MedSegDiff... | T2W 3D: nnSynergyNet3D đạt Dice 0,8651; T1W 3D: Dice 0,8789; T2W 2D: MedSegDiff đạt Dice 0,7667 | Có (kiểm định chéo tập LiTS, nhưng khác phương thức ảnh: CT) | Chỉ phân vùng gan, chưa có mô hình phân loại; dữ liệu một trung tâm; nhãn phân vùng có thể chưa chính xác hoàn toàn |
| 5 | Zeng et al., 2025, Liver Cirrhosis Stage Estimation from MRI with Deep Learning, arXiv:2502.18225 | https://arxiv.org/abs/2502.18225 | CirrMRI600+: 628 scan MRI từ 339 bệnh nhân (310 T1W, 318 T2W) | Deep learning đa chuỗi MRI, học đặc trưng đa tỷ lệ và attention đặc thù cho từng chuỗi; phân loại 3 giai đoạn (nhẹ, vừa, nặng) | T1W: Acc 72,8%; T2W: Acc 63,8% | Không (tập chia nội bộ của CirrMRI600+) | Chỉ 3 mức; hiệu suất trên T2W thấp; nhãn dựa trên đánh giá hình ảnh, chưa thay thế mô bệnh học |
| 6 | Liu et al., 2025 (bản arXiv 2024), MERIT: Multi-view Evidential Learning for Reliable and Interpretable Liver Fibrosis Staging, Medical Image Analysis | https://doi.org/10.1016/j.media.2025.103507 · https://arxiv.org/abs/2405.02918 | 703 bệnh nhân; MRI pha gan–mật có tiêm Gd-EOB-DTPA; đa trung tâm, nhiều máy MRI; nhãn xơ hóa S1–S4 | Multi-view Evidential Deep Learning; kết hợp 9 góc nhìn cục bộ và góc nhìn toàn cục bằng Subjective Logic, định lượng độ bất định | Cải thiện độ tin cậy và khả năng diễn giải; đánh giá khi dữ liệu dịch chuyển phân phối; chưa ghi chỉ số định lượng vì chưa đối chiếu bảng kết quả gốc | Không (có đánh giá đa trung tâm và OOD, chưa có cohort độc lập bên ngoài) | Phụ thuộc chất lượng và đặc trưng các góc nhìn MRI; cần thêm đánh giá trên nhóm bệnh nhân và điều kiện chụp khác |
| 7 | Lilhore et al., 2025, Advanced Liver Fibrosis Detection and Classification Through Deep Learning-Driven Image Analysis | https://doi.org/10.1007/s44196-025-01065-2 | Liver Fibrosis Dataset (LiverFib): 3.100 ảnh, gồm MRI và CT, 5 giai đoạn F0–F4 | EfficientNet-B7 + Dual Attention (không gian và kênh) + Dilated Convolutions | Accuracy 98,5%; Precision 97,2%; Recall 96,5%; F1 96,8%; AUC 0,98 | Không (chia nội bộ 70/15/15) | Chưa kiểm định đa trung tâm; trộn MRI và CT; chưa rõ tính độc lập theo bệnh nhân giữa các tập |
| 8 | Luetkens et al., 2022, Deep learning supports the differentiation of alcoholic and other-than-alcoholic cirrhosis based on MRI, Scientific Reports | https://doi.org/10.1038/s41598-022-12410-2 | 465 bệnh nhân xơ gan một trung tâm (221 do rượu, 244 không do rượu); MRI T2W, một lát tại mức thùy đuôi | Phân vùng gan tự động + ResNet50 và DenseNet121 pretrained ImageNet để phân biệt nguyên nhân xơ gan | ResNet50: AUC 0,82 (95% CI: 0,71–0,91); Acc 0,75 (95% CI: 0,64–0,85) | Không (85% train với 5-fold CV, 15% test) | Proof-of-concept, một trung tâm; một lát MRI; phân biệt nguyên nhân chứ không phân giai đoạn |
| 9 | Yu et al., 2018, Deep learning enables automated scoring of liver fibrosis stages, Scientific Reports | https://doi.org/10.1038/s41598-018-34300-2 | 25 chuột Wistar; 100 ảnh SHG của mô gan; 5 giai đoạn METAVIR F0–F4, mỗi giai đoạn 20 mẫu | AlexNet-CNN pretrained, transfer learning; so sánh ANN, MLR, SVM, Random Forest | AUROC của CNN: 0,93 (F0 vs F1–F4); 0,85 (F0–F1 vs F2–F4); 0,95 (F0–F2 vs F3–F4); 0,88 (F0–F3 vs F4) | Không (leave-one-out nội bộ) | Mô hình chuột gây xơ hóa bằng TAA, chưa kiểm chứng trên người; mẫu nhỏ; không phải ảnh MRI |
| 10 | Yasaka et al., 2018, Liver Fibrosis: Deep Convolutional Neural Network for Staging by Using Gadoxetic Acid-enhanced Hepatobiliary Phase MR Images, Radiology | https://doi.org/10.1148/radiol.2017171928 | 634 bệnh nhân một trung tâm; 534 train, 100 test; MRI pha gan–mật có tiêm Gd-EOB-DTPA; F0–F4 xác nhận bằng sinh thiết hoặc mẫu phẫu thuật | Deep Convolutional Neural Network, xây dựng điểm xơ hóa FDL từ ảnh MRI | Spearman r = 0,63 (p < 0,001); AUC: 0,84 (F4); 0,84 (F3); 0,85 (F2) | Không (tập test nội bộ) | Hồi cứu một trung tâm; cần thuốc tương phản; hiệu quả giữa các giai đoạn chưa đồng đều |
| 11 | Hectors et al., 2021, Fully automated prediction of liver fibrosis using deep learning analysis of gadoxetic acid-enhanced MRI, European Radiology | https://doi.org/10.1007/s00330-020-07475-4 | 355 bệnh nhân một trung tâm; train 178, validation 123, test 54; MRI gadoxetic acid pha gan–mật và mô bệnh học | Phân vùng gan tự động + VGG16 pretrained ImageNet dự đoán các ngưỡng F1–F4, F2–F4, F3–F4, F4 | AUC test: 0,77 (F1–F4); 0,91 (F2–F4); 0,90 (F3–F4); 0,85 (F4) | Không (tập test nội bộ) | Hồi cứu một trung tâm; test nhỏ (54 BN); AUC validation thấp hơn đáng kể so với train, khái quát hóa còn hạn chế |

Ghi chú: Dice valid (gộp pixel theo batch) và Dice test (trung bình từng lát) tính khác nhau nên không so trực tiếp. Trên test có 83 lát không có gan (238 lát nhóm khỏe trừ 155 lát có gan), mô hình dự đoán nhầm có gan ở 6 lát. Slide báo cáo hiện chưa có kết quả phân loại.

### A3. Research gaps

| Mã | Khoảng trống | Bằng chứng | Dự án của nhóm giải quyết bằng cách nào |
|---|---|---|---|
| G1 | Khả năng tổng quát hóa: phần lớn nghiên cứu dùng MRI một trung tâm và đánh giá trên tập chia nội bộ, chưa chứng minh trên nguồn dữ liệu độc lập | #1, #2, #3, #4, #5, #7, #8, #10, #11 | Dự án **không giải quyết trực tiếp** khoảng trống này vì cũng chỉ dùng CirrMRI600+ (một trung tâm). Nhóm chỉ giảm rủi ro rò rỉ nội bộ: dùng đúng split chính thức theo bệnh nhân, kiểm tra giao các tập = 0, và nêu rõ đây là hạn chế |
| G2 | Phân loại chi tiết mức độ xơ gan trên MRI T2W: các nghiên cứu T2W chủ yếu phát hiện xơ gan hoặc xơ hóa tiến triển; bài phân loại mức độ trên T2W (CirrGuide, Zeng) cho hiệu năng còn thấp, đặc biệt lớp mức vừa (F1 = 0,2979 ở CirrGuide) | #1, #2, #3, #5, #8 | Nhóm ưu tiên tái hiện CirrGuide với bài toán phân loại 3 mức độ nhẹ, vừa và nặng trên nhóm bệnh nhân xơ gan. Theo góp ý của giảng viên, nhóm sẽ bổ sung đánh giá phân loại nhị phân và 4 lớp theo phạm vi yêu cầu của môn học nếu được xác nhận. Kết quả tái hiện 3 lớp được dùng làm mốc đối chiếu trực tiếp với CirrGuide. |
| G3 | Ảnh hưởng của phân vùng gan đến phân loại: sai số phân vùng có thể ảnh hưởng đặc trưng đầu vào, nhưng chưa được khảo sát đầy đủ | #1, #2, #3, #4, #8, #11 | Dùng mask mềm của giai đoạn 1 làm tiền đề cho giai đoạn 2 đúng như CirrGuide; làm ablation theo Bảng 4 của bài (không mask, mask cứng, mask mềm, chỉ đặc trưng toàn cục); thêm phân tích theo chất lượng Dice của từng bệnh nhân (CirrGuide chưa thử nhiễu mask) |
| G4 | Độ tin cậy và tính nhất quán đánh giá: một số bài báo cáo độ chính xác rất cao nhưng trộn nhiều phương thức ảnh, mẫu nhỏ hoặc chưa rõ tính độc lập giữa các tập | #7, #9, #10, #11 | Dữ liệu có nguồn gốc xác định, chia theo bệnh nhân, báo cáo Accuracy, macro F1, F1 từng lớp, QWK, MAE (theo CirrGuide), thêm AUC, calibration, Brier và 95% CI bootstrap theo bệnh nhân |
| G5 | Khai thác MRI T2W tiêu chuẩn: các phương pháp tốt trên MRI có tiêm thuốc tương phản chưa chắc duy trì được trên T2W | #1, #2, #3, #5, #6, #10, #11 | Tập trung vào T2W 2D của CirrMRI600+ giống CirrGuide |

### A4. Thẩm định bài quan trọng nhất trong A2 bằng PROBAST+AI

Bài được chọn: **#1 CirrGuide (Raju & Ma, 2026)**, vì đây là bài nền mà nhóm tái hiện, cùng dữ liệu (CirrMRI600+ T2W 2D) và cùng split.

| Miền thẩm định | Mức độ nguy cơ sai lệch | Cơ sở đánh giá | Nhận xét và hạn chế |
|---|---|---|---|
| Người tham gia và nguồn dữ liệu | Cao | Chỉ gồm bệnh nhân đã xơ gan, một trung tâm (Đại học Istanbul), không có nhóm khỏe | Bài chỉ phân biệt 3 mức trong nhóm đã xơ gan nên không đánh giá được khả năng phân biệt khỏe / xơ gan; kết quả khó suy ra cho quần thể thực tế |
| Biến dự báo | Thấp | Đầu vào là ảnh T2W 2D; mask dùng ở giai đoạn 2 là mask do giai đoạn 1 dự đoán, nên lúc suy luận không cần mask thật | Bài không ghi cách crop, cách chuẩn hóa, danh sách augmentation, ngưỡng τ nên khó tái hiện chính xác |
| Kết cục | Cao | Nhãn nhẹ/vừa/nặng lấy từ đánh giá hình ảnh học (radiological evaluation), không phải mô bệnh học | Nhãn và đầu vào cùng là ảnh, có nguy cơ nhãn không độc lập với dự báo; ranh giới giữa các mức liền kề mơ hồ (lớp mức vừa F1 chỉ 0,2979) |
| Phân tích | Cao | Tập test 31 bệnh nhân, lớp mức vừa chỉ 6 bệnh nhân (đúng 1/6); không có khoảng tin cậy, không có calibration, không kiểm định ngoài; so sánh với Zeng chỉ tham khảo vì khác cách chia | Chọn mô hình theo macro F1 trên validation là hợp lý, nhưng số liệu theo bệnh nhân dao động rất mạnh với cỡ mẫu này |

Tính áp dụng cho bài toán của nhóm: **cao** về phương pháp, vì nhóm dùng cùng dữ liệu, cùng split, cùng pipeline nên có thể đối chiếu trực tiếp. Hạn chế cần lưu ý: bài không có nhóm khỏe, nên kết quả không suy ra được cho bài toán phân biệt khỏe / xơ gan; nhóm cũng không làm bài toán này.

---

## B. Kết quả và timeline

### B1. Kết quả hiện có

Hiện tại, nhóm đã hoàn thành đánh giá mô hình Baseline 1 (U-Net phân vùng gan) trên tập test CirrMRI600+ theo cách chia ở mức bệnh nhân. Các kết quả này được thực hiện với pipeline tiền xử lý phiên bản cũ, **chưa phải kết quả của pipeline tái hiện CirrGuide**. Nhóm đang thực hiện lại các bước tiền xử lý theo phương pháp của CirrGuide, bao gồm crop, resize về 256 × 256, chuẩn hóa và augmentation đồng bộ giữa ảnh MRI và mask. Vì vậy, kết quả của pipeline mới sẽ được cập nhật sau khi hoàn thành huấn luyện và đánh giá.

**Thông tin tập test của Baseline 1:** Tổng cộng 902 lát từ 37 bệnh nhân, gồm 664 lát của 31 bệnh nhân xơ gan và 238 lát của 6 bệnh nhân khỏe mạnh. Trong 238 lát khỏe mạnh, có 155 lát chứa gan và 83 lát không có gan. Như vậy, tập test có tổng cộng 819 lát chứa gan và 83 lát không có gan. Mô hình Baseline 1 dự đoán nhầm có gan ở 6/83 lát không có gan.

| Mô hình | Tập đánh giá (cách chia, n) | Metric chính (giá trị, 95% CI nếu có) | Metric phụ | Nguồn số liệu |
|---|---|---|---|---|
| **Baseline 1 (Mốc 2): U-Net phân vùng gan — pipeline tiền xử lý cũ** | Tập test CirrMRI600+ chia theo bệnh nhân: 902 lát từ 37 bệnh nhân (31 xơ gan, 6 khỏe mạnh); gồm 819 lát có gan và 83 lát không có gan | Dice trung bình từng lát trên toàn bộ tập test: **0,9025 ± 0,1760**; trên 819 lát có gan: **0,8994**; Dice theo khối bệnh nhân: **0,9513** (37 BN). Chưa tính 95% CI | IoU toàn bộ test: 0,8522 ± 0,1895; Precision: 0,9066 ± 0,1722; Recall: 0,9136 ± 0,1834. Trên 819 lát có gan: IoU 0,8476; Precision 0,9256; Recall 0,9116. Riêng nhóm xơ gan: Dice từng lát 0,9006; Dice theo khối bệnh nhân 0,9482 (31 BN). Dự đoán nhầm có gan ở 6/83 lát không có gan | Notebook đánh giá mô hình phân vùng gan và bảng tổng hợp kết quả tập test |
| **CirrGuide-Seg (ResNet50 + Attention U-Net, Mốc 3)** | Cùng tập test; đánh giá trên dữ liệu sau khi hoàn thành tiền xử lý theo CirrGuide | Chưa có kết quả chính thức | Chưa có | Notebook tiền xử lý và huấn luyện phân vùng giai đoạn 1; đang thực hiện |
| **CirrGuide-Cls 3 lớp (Mốc 3)** | Tập test chính thức của nhóm bệnh nhân xơ gan: 31 BN, 664 lát; sử dụng dữ liệu sau tiền xử lý theo CirrGuide | Chưa có Accuracy và macro F1 | Chưa có F1 từng lớp, QWK, MAE, ma trận nhầm lẫn và kết quả theo bệnh nhân | Notebook huấn luyện và đánh giá mô hình phân loại; chưa có kết quả |

**Lưu ý:** Baseline 1 được giữ lại làm mốc so sánh với mô hình tái hiện CirrGuide. Không so sánh trực tiếp kết quả của hai pipeline để kết luận mô hình nào tốt hơn nếu chưa kiểm soát sự khác biệt về tiền xử lý và cấu hình huấn luyện. Các chỉ số của CirrGuide-Seg và CirrGuide-Cls sẽ được cập nhật sau khi nhóm hoàn thành thực nghiệm trên pipeline mới.

### B2. Ablation (Mốc 3)

Các cấu hình lấy đúng theo Bảng 4 của CirrGuide. Cột "Số của bài" chỉ để tham chiếu, **không phải kết quả của nhóm**.

| Cấu hình | Metric chính (Acc / macro F1) của nhóm | Chênh lệch so với mô hình đầy đủ | Số của bài (Acc / macro F1) | Nguồn |
|-|-|-|-|-|
| Chỉ đặc trưng toàn cục (GAP), không mask — tương ứng "không phân vùng" | Chưa có kết quả | Chưa có | 0,5828 / 0,5535 | `cls_results/` (chưa có) |
| Không mask + Attention | Chưa thực hiện | Chưa có | 0,5587 / 0,4784 | Chưa có |
| Mask cứng + Attention | Chưa thực hiện | Chưa có | 0,6039 / 0,5608 | Chưa có |
| Mask mềm + Average pooling | Chưa thực hiện | Chưa có | 0,5783 / 0,5401 | Chưa có |
| CirrGuide đầy đủ (mask mềm + Attention pooling) | Chưa có kết quả | — | 0,6958 / 0,6155 | `cls_results/` (chưa có) |

Ưu tiên: hai dòng đầu và dòng cuối chạy trước (trả lời câu hỏi "có phân vùng tốt hơn không phân vùng không"); ba dòng còn lại chạy nếu đủ thời gian.

### B3. Đối chiếu với y văn


* So với **#1 CirrGuide** (cùng dataset, cùng tập 31 bệnh nhân xơ gan ở test nếu nhóm dùng đúng split chính thức): Dice nhóm xơ gan 0,9006 so với 0,8983 của CirrGuide và 0,8967 của U-Net trong bài đó. Chênh lệch dưới 0,005, với 31 bệnh nhân và không có CI nên **chưa kết luận hơn kém**. Cũng cần lưu ý bài không nêu rõ cách tính Dice (theo lát hay theo khối), và mIoU của bài (0,8414) chưa chắc cùng định nghĩa với IoU của nhóm.
* So với **#4 (CirrMRI600+)**: Dice của các mô hình 2D trên T2W là 0,69–0,77 và 3D tốt nhất là 0,8651. Dice của nhóm cao hơn, nhưng phép so sánh *không công bằng* vì bài #4 có thể tính theo khối bệnh nhân và khác chiến lược huấn luyện (số epoch, augmentation), còn nhóm tính theo từng lát.
* So với **#3 (Nowak 2021)**: Dice 0,983. Phép so sánh *không công bằng* vì bài #3 chỉ dùng một lát ở thùy đuôi, nơi gan lớn và rõ, còn nhóm dùng cả các lát ở rìa gan nơi gan rất nhỏ (có lát Dice bằng 0).

**Phân loại.** Chưa có kết quả nên chưa đối chiếu được. Khi có kết quả:

* Đối chiếu trực tiếp nhất là **#1 CirrGuide**: cùng dữ liệu, cùng split, cùng 3 lớp (Acc 0,6958; macro F1 0,6155 theo lát; Acc 0,7097, macro F1 0,5859 theo bệnh nhân; F1 lớp vừa 0,2979). Nhóm cần báo cáo cả F1 từng lớp vì lớp mức vừa là điểm yếu đã biết.
* **#5 Zeng**: Acc 0,638 cho 3 lớp T2W, khác cách chia, chỉ tham khảo (chính CirrGuide cũng ghi như vậy).
* **#3 Nowak** (Acc 0,96, nhị phân khỏe / xơ gan, một lát mỗi bệnh nhân): không đối chiếu số vì nhóm không làm bài toán nhị phân; chỉ dùng làm bối cảnh y văn.

**Kết luận đối chiếu.** Phần phân vùng của nhóm có kết quả ổn định giữa các nhóm bệnh (Dice 0,894–0,903) nhưng bị kéo xuống bởi một số ít lát gan rất nhỏ. Điểm mạnh của dự án là chia dữ liệu theo bệnh nhân, kiểm tra không rò rỉ, và nhận diện sớm nguy cơ mô hình học lối tắt (lát không có gan).

### B4. Timeline đến khi hoàn thành

Mốc: M3 tuần 10 · M4 tuần 12 · Final tuần 15. Cột "Ngày dự kiến" ghi theo hạn của mốc; nhóm thay bằng ngày cụ thể nếu có.

| Việc | Phụ trách | Mốc | Ngày dự kiến | Trạng thái (Chưa / Đang / Xong + bằng chứng) |
|-|-|-|-|-|
| Chia dữ liệu theo bệnh nhân, kiểm tra không rò rỉ | 20233772 + 20233735 | M3 tuần 10 | Xong | Xong — train 300 BN, valid 36 BN, test 37 BN; kết quả giao các tập = 0 trong notebook |
| Huấn luyện và đánh giá U-Net phân vùng (Baseline 1) | 20233772 + 20233735 | M3 tuần 10 | Xong | Xong — Dice valid 0,9561 (epoch 9); Dice test 0,8994 (với tiền xử lý cũ) |
| Slide báo cáo | 20233772 + 20233735 | M3 tuần 10 | Xong | Xong — đã đưa lên Google Drive: [dán link]; còn phải đưa file slide vào repo/PR |
| Tiền xử lý lại dữ liệu theo CirrGuide (crop, resize 256 × 256, chuẩn hóa, augmentation giữ căn chỉnh ảnh–mask) | 20233772 + 20233735 | M3 tuần 10 | Hạn M3 | Đang — nhóm đang xử lý lại; chưa có kết quả |
| Phân vùng giai đoạn 1 theo CirrGuide: ResNet50 encoder + Attention U-Net, loss 0,5 Dice + 0,5 BCE (trọng số dương 5,0), đóng băng sau khi huấn luyện | 20233772 + 20233735 | M3 tuần 10 | Hạn M3 | Đang — viết code; chưa có kết quả |
| Phân loại giai đoạn 2 theo CirrGuide, 3 lớp nhẹ/vừa/nặng (ResNet50, GAP + Mask-Guided Attention Pooling, MLP, weighted sampling) trên split chính thức | 20233772 + 20233735 | M3 tuần 10 | Hạn M3 | Đang — viết code; chưa có kết quả |
| Điền SHA-256, checklist TRIPOD+AI, xác nhận giấy phép trên OSF | 20233772 + 20233735 | M3 tuần 10 | [điền ngày] | Chưa — lệnh: `sha256sum Cirrhosis_T2_2D.zip` |
| Bổ sung `submission.json`, README, `src/`, slide; chạy `validate_submission.py` và `review_project.py`; cập nhật PR #25 (hoặc đóng PR và nộp Test 2 qua comment SHA ở issue #19/#20) | 20233772 + 20233735 | M3 tuần 10 | [điền ngày] | Chưa |
| Tổng hợp theo bệnh nhân (trung bình softmax các lát); Dice theo khối bệnh nhân | 20233772 + 20233735 | M4 tuần 12 | Hạn M4 | Chưa |
| Ablation theo Bảng 4 của CirrGuide (ưu tiên: chỉ GAP, không mask + attention, đầy đủ) | 20233772 + 20233735 | M4 tuần 12 | Hạn M4 | Chưa |
| Calibration, Brier, 95% CI bootstrap theo bệnh nhân | 20233772 + 20233735 | M4 tuần 12 | Hạn M4 | Chưa |
| Grad-CAM kiểm tra mô hình nhìn vào vùng nào | 20233772 + 20233735 | M4 tuần 12 | Hạn M4 | Chưa |
| Hoàn thiện FAIR (datasheet, checksum), checklist TRIPOD+AI | 20233772 + 20233735 | M4 tuần 12 | Hạn M4 | Chưa |
| Nộp final, báo cáo và Pull Request | 20233772 + 20233735 | Final tuần 15 | Hạn Final | Chưa |

### B5. Rủi ro / đang vướng / câu hỏi cho GV

* Phần phân loại chỉ dùng 318 bệnh nhân xơ gan (giống CirrGuide) nên không còn rủi ro "không có gan thì là khỏe". Nhóm khỏe (55 BN, 6 BN ở test) chỉ còn xuất hiện ở Baseline 1 phân vùng; khoảng 38% lát nhóm khỏe không có gan (trên tập test là 83/238 lát, khoảng 34,9%), làm Dice trung bình bị kéo dao động.
* Ảnh nhóm khỏe do nhóm tự tách từ NIfTI và chuẩn hóa min–max từng lát; chưa chắc cùng điều kiện với ảnh PNG có sẵn của nhóm xơ gan. Với phần phân loại, vấn đề này không ảnh hưởng.
* Tập test phân loại chỉ có 31 bệnh nhân, lớp mức vừa chỉ 6 bệnh nhân (như CirrGuide), nên số theo bệnh nhân của lớp này rất dao động.
* CirrGuide không công bố cách crop, cách chuẩn hóa, danh sách augmentation, ngưỡng τ và kích thước lớp ẩn của MLP; nhóm sẽ tự chọn, ghi rõ trong README và không nhận đó là chi tiết của tác giả. Kết quả tái hiện có thể không trùng 0,6958 / 0,6155.
* Mô hình mới được kiểm định nội bộ trên một bộ dữ liệu, chưa có kiểm định ngoài.
* Câu hỏi cho GV: thầy gợi ý làm nhị phân trước rồi 4 lớp. Nhóm đang ưu tiên tái hiện CirrGuide với bài toán 3 mức độ nhẹ/vừa/nặng để đối chiếu trực tiếp với bài báo. Đồng thời, nhóm sẽ triển khai các bài toán phân loại nhị phân và 4 lớp theo góp ý của giảng viên. Nhóm cần thống nhất cách định nghĩa nhãn và phân chia dữ liệu cho các bài toán mở rộng để bảo đảm đánh giá đúng theo bệnh nhân.
---

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu

| Mục | Trả lời | Bằng chứng |
|-|-|-|
| Nguồn, phiên bản, ngày tải; checksum file dùng để huấn luyện | CirrMRI600+ (Jha et al., Scientific Data 2025;12:896, công bố 28/05/2025), tải từ OSF https://osf.io/cuk24/. Ngày tải: [20/9/2026]. SHA-256 của file zip dùng để huấn luyện: | https://www.nature.com/articles/s41597-025-05201-7 |
| Giấy phép / DUA, căn cứ được phép dùng | Bản tóm tắt của bài dữ liệu ghi CC BY-SA 4.0. **Chưa khớp hoàn toàn với OSF:** trang dự án OSF (kiểm tra 28/11/2026) hiển thị trường License là "Other". Nhóm mở mục Files/Wiki của OSF để đọc điều khoản thực tế. Trong lúc chờ xác nhận: dữ liệu công khai, dùng cho học tập, ghi nguồn, không đưa dữ liệu ảnh vào repo | https://osf.io/cuk24/ |
| Datasheet: quần thể, tiêu chí chọn/loại, định nghĩa và đơn vị biến, giới hạn | Quần thể: 373 BN, ảnh T2W (318 xơ gan có mask gan và mức độ 1–3 từ "Radiological Evaluation", gồm 131 nhẹ, 114 vừa, 73 nặng; 55 khỏe mạnh có mask gan). Ảnh xơ gan là PNG 2D chia sẵn train/valid/test (6.766 lát = 5.428 + 674 + 664); ảnh khỏe là NIfTI 3D nhóm tự tách thành 2.143 lát (thang 0–255, mask nhị phân); tổng 8.909 lát. Dữ liệu gốc thu tại Đại học Istanbul (một trung tâm, nhiều hãng máy). Giới hạn: nhãn mức độ từ đọc ảnh, không phải sinh thiết; nhóm khỏe nhỏ; một nguồn dữ liệu | `CirrMRI600_full_dataset.csv` |
| Không có định danh bệnh nhân | Nhóm chỉ dùng mã dạng cirr_xxx, healthy_xx; không đưa ảnh hay họ tên vào repo. Theo bài dữ liệu, dữ liệu gốc đã được ẩn danh theo yêu cầu IRB, loại bỏ thông tin sức khỏe cá nhân (PHI) và chuyển từ DICOM sang NIfTI | https://www.nature.com/articles/s41597-025-05201-7 |

### C2. Hướng dẫn báo cáo

* **Câu hỏi nghiên cứu của nhóm:** Với ảnh MRI T2W 2D, pipeline hai giai đoạn của CirrGuide (ResNet50 + Attention U-Net tạo mask mềm; ResNet50 phân loại có Mask-Guided Attention Pooling) có tái hiện được kết quả phân loại 3 mức độ xơ gan (nhẹ, vừa, nặng) trên split chính thức của CirrMRI600+ không, và mask mềm có cải thiện so với chỉ dùng đặc trưng toàn cục không?
* **Hướng dẫn báo cáo phù hợp:** *TRIPOD+AI*, vì đề tài phát triển và đánh giá một mô hình dự đoán chẩn đoán (Phụ lục §3.2). Không dùng STARD-AI vì nhóm không đánh giá một xét nghiệm AI có sẵn so với chuẩn tham chiếu; không dùng DECIDE-AI vì chưa đánh giá lâm sàng giai đoạn sớm.
* **Số mục checklist đã đáp ứng / tổng số mục; các mục còn thiếu:** 
* **Kết cục, nhãn, T0, biến có sẵn tại T0:** mức độ xơ gan 3 lớp (0 nhẹ, 1 vừa, 2 nặng). T0 là lúc có ảnh MRI T2W; biến dự báo chỉ gồm ảnh MRI (không dùng tuổi, giới, xét nghiệm).

**Thiết kế mô hình và đối chiếu với CirrGuide.** Nhóm chỉ dùng 318 bệnh nhân xơ gan, split chính thức 256/31/31, bám đúng bài gốc.

| Thành phần | CirrGuide (theo bài) | Nhóm thực hiện | Ghi chú |
|-|-|-|-|
| Dữ liệu và chia tập | CirrMRI600+ T2W 2D chính thức; 256/31/31 BN; 5.428/674/664 lát | Giống hệt (chỉ bệnh nhân xơ gan) | Split hiện có 300/36/37 trừ 44/5/6 BN khỏe thì bằng 256/31/31; nhóm kiểm tra danh sách mã BN có trùng split chính thức không |
| Tiền xử lý | Crop, resize 256 × 256, chuẩn hóa, augmentation giữ căn chỉnh ảnh–mask (chỉ train) | Cùng các bước | Bài không ghi cách crop, min–max hay z-score, phép augmentation: nhóm tự chọn [điền] và ghi lại |
| Giai đoạn 1: phân vùng | ResNet50 encoder + Attention U-Net decoder (cổng chú ý ở các đường nối, upsample bilinear ×2); sigmoid → mask mềm M | Giống | Baseline 1 (U-Net) giữ làm mốc so sánh, tương ứng dòng U-Net trong Bảng 1 của bài |
| Loss phân vùng | 0,5 Dice + 0,5 BCE, trọng số pixel dương 5,0 | Giống | Bài không nêu ngưỡng τ để lấy mask cứng |
| Sau giai đoạn 1 | Đóng băng toàn bộ trọng số | Giống | Giai đoạn 2 dùng mask mềm do mạng này dự đoán |
| Giai đoạn 2: phân loại | ResNet50 riêng; dùng F3, F4, F5; GAP trên F3–F5 → g3, g4, g5; MAP trên F4, F5 với M4, M5 (Ã = σ(conv1×1(ReLU(conv1×1(F))) + M); r = ΣFÃ / (ΣÃ + ε)); z = [g3, g4, g5, r4, r5] → MLP → softmax 3 lớp | Giống (thay ResNet18 cũ bằng ResNet50) | Bài không ghi số nơ-ron lớp ẩn của MLP: nhóm tự chọn [điền] |
| Loss và cân bằng lớp | Cross-entropy; lấy mẫu có trọng số (train: 2.215 nhẹ, 2.097 vừa, 1.116 nặng) | Giống. Thay "class weight" cũ bằng weighted sampling | |
| Tối ưu | AdamW, lr 1×10⁻⁴, weight decay 1×10⁻⁴, batch 12, tối đa 40 epoch, early stopping, chọn mô hình theo macro F1 trên validation | Giống | |
| Từ lát lên bệnh nhân | Trung bình softmax các lát của một bệnh nhân, chọn lớp cao nhất | Giống | |
| Metric | Dice, mIoU; Accuracy, macro F1, F1 từng lớp, QWK, MAE; theo lát và theo bệnh nhân | Giống, thêm AUC, calibration, Brier, 95% CI bootstrap theo bệnh nhân | Phần thêm là đóng góp của nhóm, không có trong bài gốc |

### C3. PROBAST+AI — tự thẩm định dự án của nhóm

| Miền | Nguy cơ sai lệch | Căn cứ trong repo (file, dòng) | Việc sẽ làm để giảm nguy cơ |
|-|-|-|-|
| Người tham gia và nguồn dữ liệu | Cao | Phần phân loại chỉ có bệnh nhân xơ gan, một trung tâm, không có nhóm khỏe nên không phản ánh phổ bệnh thực tế; nhãn mức độ đến từ đọc ảnh. | Grad-CAM; nêu rõ hạn chế về một nguồn dữ liệu và nhãn từ đọc ảnh |
| Phân tích (chia theo bệnh nhân, fit chỉ trên train, cỡ mẫu, dữ liệu thiếu) | Không rõ | Có: chia theo bệnh nhân (train 300 / valid 36 / test 37 BN) và kiểm tra giao các tập = 0; chuẩn hóa không fit từ dữ liệu (min–max từng lát, thống kê ImageNet cố định); weighted sampling tính chỉ từ train. Thiếu: calibration, khoảng tin cậy, chứng minh cỡ mẫu; test chỉ 31 BN, lớp mức vừa chỉ 6 BN. | Bootstrap theo bệnh nhân lấy 95% CI; calibration và Brier; báo cáo theo bệnh nhân |

Tính áp dụng (quần thể, bối cảnh dùng mô hình): thấp. Dữ liệu một nguồn, nhóm khỏe nhỏ, chưa có kiểm định ngoài nên chưa đủ để dùng trong lâm sàng.

### C4. Bảng kiểm nhanh (Phụ lục §7)

| Mục | Đạt / Chưa | Bằng chứng |
|-|-|-|
| Datasheet | Một phần | Mục C1 (bản tóm tắt); chưa có file datasheet riêng |
| Phiên bản + checksum dữ liệu | Chưa | Có nguồn và bài công bố (mục C1); SHA-256:  |
| Không định danh; căn cứ được phép dùng dữ liệu | Đạt một phần | Dữ liệu công khai, đã ẩn danh theo bài dữ liệu; giấy phép cần đối chiếu lại với OSF (mục C1) |
| Chọn đúng hướng dẫn báo cáo + checklist đã điền | Một phần | Đã chọn TRIPOD+AI (mục C2); checklist chưa điền |
| Định nghĩa kết cục, nhãn, T0 | Đạt | Mục C2 |
| Chia theo bệnh nhân; tiền xử lý fit chỉ trên train | Đạt | Output kiểm tra giao các tập = 0 trong notebook; mục C3 |
| Mô hình mốc | Một phần | Baseline phân vùng: U-Net (Dice test 0,8994). Baseline phân loại: cấu hình "chỉ đặc trưng toàn cục" (ResNet50 + GAP, không mask) trong B2; chưa có kết quả. Hồi quy logistic trên pixel không phù hợp với ảnh |
| AUC + calibration + Brier + 95% CI | Chưa | Hiện mới có Dice cho phân vùng; phần phân loại đang thực hiện, chưa có kết quả |
| Cỡ mẫu và giới hạn về tính áp dụng | Chưa | Mới nêu giới hạn (mục B5, C3) |

---

## D. Khai báo dùng AI

| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
|-|-|-|
| Claude (Anthropic) | Hỗ trợ viết code Colab (tải dữ liệu, đánh giá, phân loại); giải thích khái niệm và pipeline CirrGuide; hướng dẫn đọc và tóm tắt các bài báo| Đối chiếu số liệu phân vùng với slide báo cáo và notebook; đối chiếu mô tả pipeline và số liệu CirrGuide với bản PDF và trang arXiv 2609.14010; đọc lại abstract các bài #3, #4, #5, #8, #10, #11 |
