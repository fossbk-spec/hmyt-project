# Bài kiểm tra lần 2 — Nhóm 06 — <ung_dung_hoc_sau_trong_phan_vung_va_phan_loai_xo_gan_bang_anh_MRI>
Fork: https://github.com/nguyenthuylinh0903m-lang/hmyt-project · Nhánh: gd5-proposed-model
Thành viên (MSSV hoặc alias) và vai trò:
* 20233772 — Đồng thực hiện; tham gia xử lý dữ liệu, huấn luyện và đánh giá U-Net, xây dựng mô hình phân loại và hoàn thiện báo cáo. Phân công chi tiết giữa hai thành viên chưa được ghi chính thức trong tài liệu hiện có.
* 20233735 — Đồng thực hiện; tham gia tổng quan y văn, xử lý dữ liệu, đánh giá mô hình và hoàn thiện báo cáo. Phân công chi tiết giữa hai thành viên chưa được ghi chính thức trong tài liệu hiện có.

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm


|Nguồn (PubMed, Scopus, IEEE Xplore, Google Scholar…)|Từ khóa / truy vấn|Khoảng năm|Số kết quả|Số bài giữ lại|
|-|-|-|-|-|
|PubMed|`("liver cirrhosis" OR "liver fibrosis") AND ("MRI" OR "magnetic resonance") AND ("deep learning" OR "convolutional neural network")`|2018–2026|
|PubMed|`("liver segmentation") AND ("U-Net") AND ("MRI")`|2018–2026|[24.600]|[6]|
|Google Scholar|`CirrMRI600+ liver cirrhosis T2-weighted classification`|2022–2026|[12]|[6]|
|arXiv|`liver cirrhosis stage MRI deep learning`|2022–2026|[17.000]|[7]|

Ngày tìm kiếm: [15/10/2026].

Tiêu chí chọn / loại bài:

* Chọn nghiên cứu dùng MRI để phát hiện hoặc phân giai đoạn xơ gan / xơ hóa gan, hoặc phân vùng gan trên MRI bằng học sâu.
* Ưu tiên bài dùng chuỗi xung T2W hoặc bộ dữ liệu CirrMRI600+, vì cùng dữ liệu với đề tài của nhóm.
* Ưu tiên bài có báo cáo Dice/IoU (phân vùng) hoặc Accuracy, Sensitivity, Specificity, AUC (phân loại).
* Ưu tiên nghiên cứu 3–5 năm gần nhất, giữ một số bài cũ có giá trị nền tảng hoặc so sánh trực tiếp.
* Loại nghiên cứu chỉ dùng siêu âm, CT hoặc xét nghiệm huyết thanh, không dùng ảnh MRI.
* Loại bài không có số liệu hiệu năng đủ để đối chiếu, và bản trùng lặp của cùng một nghiên cứu.

### A2. Ma trận y văn (≥ 10 bài, ưu tiên 3–5 năm gần nhất)

|#|Bài (tác giả, năm)|DOI / link|Dataset (n, nguồn)|Phương pháp|Kết quả chính (metric, giá trị)|Kiểm định ngoài? (có / không)|Hạn chế chính|
|-|-|-|-|-|-|-|-|
1|Gupta et al., 2025, Advanced liver fibrosis detection using a two-stage deep learning approach on standard T2-weighted MRI|https://doi.org/10.1007/s00261-025-05160-y|CirrMRI600+ (n = 374: 318 bệnh nhân xơ gan, 56 người khỏe mạnh) và dữ liệu nội bộ (n = 187 bệnh nhân mắc bệnh gan mạn tính)|Hai giai đoạn: nnU-Net phân vùng gan + Masked Attention ResNet phân loại xơ hóa tiến triển (METAVIR ≥ F3)|Dice 0,960 ± 0,009; IoU 0,923 ± 0,016; AUC 0,811; Acc 0,783; Sens 0,778; Spec 0,800 (tập kiểm tra n = 23)|Không (có tập kiểm tra độc lập nội bộ n = 23, chưa chứng minh kiểm định ngoài trung tâm)|Tập kiểm tra nhỏ; hiệu suất phân loại phụ thuộc chất lượng phân đoạn gan; chưa đánh giá trên nhiều trung tâm độc lập.|
|2|Nowak et al., 2021, Detection of liver cirrhosis in standard T2-weighted MRI using deep transfer learning, European Radiology|https://doi.org/10.1007/s00330-021-07858-1|713 bệnh nhân tại một bệnh viện (553 xơ gan, 160 nhóm chứng); MRI T2W, một lát tại mức thùy đuôi|U-Net với encoder ResNet34 phân vùng gan + ResNet50 phân loại xơ gan|Dice > 0,98; Acc test = 0,96 khi có phân vùng và 0,95 khi không phân vùng; Sens = 1,00; Spec = 0,83|Không (chia train/validation/test nội bộ 70/15/15)|Nghiên cứu một trung tâm; sử dụng một lát MRI được chọn thủ công; chỉ phát hiện xơ gan, chưa phân loại các giai đoạn F1–F4.|
|3|Jha et al., 2025, Large Scale MRI Collection and Segmentation of Cirrhotic Liver, Scientific Data|https://doi.org/10.1038/s41597-025-05201-7|CirrMRI600+: 628 scan MRI từ 339 bệnh nhân (310 T1W, 318 T2W), gần 40.000 lát ảnh có nhãn phân vùng gan|Benchmark 11 mô hình phân vùng 3D và 6 mô hình phân vùng 2D; gồm nnUNet, Attention U-Net, nnFormer, nnSynergyNet3D, MedSegDiff...|T2W 3D: nnSynergyNet3D đạt Dice 0,8651; T1W 3D: Dice 0,8789; T2W 2D: MedSegDiff đạt Dice 0,7667|Có (kiểm định chéo tập LiTS, nhưng khác phương thức ảnh: CT)|Chỉ tập trung vào phân vùng gan, chưa xây dựng mô hình phân loại giai đoạn xơ hóa; dữ liệu thu thập tại một trung tâm; nhãn phân vùng có thể chưa chính xác hoàn toàn.|
|4|Zeng et al., 2025, Liver Cirrhosis Stage Estimation from MRI with Deep Learning, arXiv:2502.18225|https://arxiv.org/abs/2502.18225|CirrMRI600+: 628 scan MRI từ 339 bệnh nhân (310 T1W, 318 T2W)|Deep learning đa chuỗi MRI, kết hợp học đặc trưng đa tỷ lệ và cơ chế attention đặc thù cho từng chuỗi ảnh; phân loại 3 giai đoạn xơ gan (nhẹ, vừa, nặng)|T1W: Accuracy 72,8%; T2W: Accuracy 63,8%|Không (đánh giá trên các tập chia nội bộ của CirrMRI600+)|Chỉ phân loại 3 mức độ xơ gan; hiệu suất trên T2W còn thấp; nhãn phân giai đoạn dựa trên đánh giá hình ảnh học, chưa thay thế xác nhận mô bệnh học.|
|5|Liu et al., 2025, MERIT: Multi-view Evidential Learning for Reliable and Interpretable Liver Fibrosis Staging, Medical Image Analysis|https://doi.org/10.1016/j.media.2025.103507|703 bệnh nhân; MRI pha gan–mật có tiêm Gd-EOB-DTPA; dữ liệu đa trung tâm, nhiều máy MRI; nhãn xơ hóa S1–S4|Multi-view Evidential Deep Learning; kết hợp 9 góc nhìn cục bộ và góc nhìn toàn cục bằng Subjective Logic, định lượng độ bất định|Cải thiện độ tin cậy và khả năng diễn giải; đánh giá khả năng hoạt động khi có dịch chuyển phân phối dữ liệu; không ghi chỉ số định lượng khi chưa đối chiếu bảng kết quả gốc|Không (có đánh giá đa trung tâm và OOD, nhưng chưa xác nhận cohort độc lập bên ngoài)|Phụ thuộc chất lượng và đặc trưng các góc nhìn MRI; cần đánh giá thêm trên các nhóm bệnh nhân và điều kiện chụp khác nhau.|
|6|Lilhore et al., 2025, Advanced Liver Fibrosis Detection and Classification Through Deep Learning-Driven Image Analysis|https://doi.org/10.1007/s44196-025-01065-2|Liver Fibrosis Dataset (LiverFib): 3.100 ảnh, gồm MRI và CT, chia thành 5 giai đoạn F0–F4|EfficientNet-B7 + Dual Attention (Spatial Attention và Channel Attention) + Dilated Convolutions|Accuracy 98,5%; Precision 97,2%; Recall 96,5%; F1-score 96,8%; AUC 0,98|Không (chia train/validation/test nội bộ 70/15/15)|Chưa kiểm định trên dữ liệu đa trung tâm độc lập; kết hợp MRI và CT; thông tin nguồn gốc và tính độc lập theo bệnh nhân của dữ liệu cần được làm rõ.|
|7|Luetkens et al., 2022, Deep learning supports the differentiation of alcoholic and other-than-alcoholic cirrhosis based on MRI, Scientific Reports|https://doi.org/10.1038/s41598-022-12410-2|465 bệnh nhân xơ gan tại một trung tâm (221 do rượu, 244 không do rượu); MRI T2W, một lát tại mức thùy đuôi|Phân vùng gan tự động + ResNet50 và DenseNet121 pretrained ImageNet để phân biệt nguyên nhân xơ gan|ResNet50: AUC 0,82 (95% CI: 0,71–0,91); Accuracy 0,75 (95% CI: 0,64–0,85)|Không (chia dữ liệu nội bộ, 85% train với 5-fold CV và 15% test)|Nghiên cứu proof-of-concept, một trung tâm; chỉ sử dụng một lát MRI; phân biệt nguyên nhân xơ gan chứ không trực tiếp phân giai đoạn xơ hóa.|
|8|Yu et al., 2018, Deep learning enables automated scoring of liver fibrosis stages, Scientific Reports|https://doi.org/10.1038/s41598-018-34300-2|25 chuột Wistar; 100 mẫu mô gan với 100 ảnh SHG; chia 5 giai đoạn METAVIR F0–F4, mỗi giai đoạn 20 mẫu|AlexNet-CNN pretrained và transfer learning; so sánh với ANN, MLR, SVM và Random Forest|AUROC của CNN: 0,93 (F0 vs F1–F4); 0,85 (F0–F1 vs F2–F4); 0,95 (F0–F2 vs F3–F4); 0,88 (F0–F3 vs F4)|Không (đánh giá nội bộ bằng leave-one-out)|Dữ liệu từ mô hình chuột gây xơ hóa bằng TAA, chưa kiểm chứng trên mô gan người; số lượng mẫu nhỏ; chưa sử dụng ảnh MRI.|
|9|Yasaka et al., 2018, Liver Fibrosis: Deep Convolutional Neural Network for Staging by Using Gadoxetic Acid-enhanced Hepatobiliary Phase MR Images, Radiology|https://doi.org/10.1148/radiol.2017171928|634 bệnh nhân tại một trung tâm; 534 train và 100 test; MRI pha gan–mật có tiêm Gd-EOB-DTPA; giai đoạn F0–F4 được xác nhận bằng sinh thiết hoặc mẫu phẫu thuật|Deep Convolutional Neural Network (DCNN), xây dựng điểm xơ hóa FDL từ ảnh MRI|Spearman r = 0,63 (p < 0,001); AUC: 0,84 (F4); 0,84 (F3); 0,85 (F2)|Không (tập test nội bộ)|Nghiên cứu hồi cứu một trung tâm; phụ thuộc MRI có tiêm thuốc tương phản; hiệu quả ở từng giai đoạn chưa đồng đều.|
|10|Hectors et al., 2021, Fully automated prediction of liver fibrosis using deep learning analysis of gadoxetic acid-enhanced MRI, European Radiology|https://doi.org/10.1007/s00330-020-07475-4|355 bệnh nhân tại một trung tâm; train 178, validation 123, test 54; MRI gadoxetic acid pha gan–mật và kết quả mô bệnh học|Phân vùng gan tự động + VGG16 pretrained ImageNet để dự đoán các ngưỡng xơ hóa F1–F4, F2–F4, F3–F4 và F4|AUC test: 0,77 (F1–F4); 0,91 (F2–F4); 0,90 (F3–F4); 0,85 (F4)|Không (tập test nội bộ)|Nghiên cứu hồi cứu một trung tâm; tập test nhỏ (54 bệnh nhân); AUC validation thấp hơn đáng kể so với train, cho thấy khả năng khái quát hóa còn hạn chế.|

Ghi chú: Dice valid (gộp pixel theo batch) và Dice test (trung bình từng lát) tính khác nhau nên không so trực tiếp. Trên test có 83 lát không có gan (238 lát nhóm khỏe trừ 155 lát có gan), mô hình dự đoán nhầm có gan ở 6 lát. Slide báo cáo hiện chưa có kết quả phân loại.

### B2. Ablation (Mốc 3)

|Cấu hình (bỏ / đổi thành phần nào)|Metric chính|Chênh lệch so với mô hình đầy đủ|Nguồn|
|-|-|-|-|
|Có phân vùng (B) so với không phân vùng (A)|Chưa có kết quả|Chưa có|`cls_results/` (chưa có kết quả)|
|Bỏ class weight; bỏ augmentation|Chưa thực hiện|Chưa có|Chưa có|

### B3. Đối chiếu với y văn

Phân vùng: mô hình của nhóm đạt Dice trung bình từng lát 0,8994 trên 819 lát có gan của tập test (nhóm xơ gan 0,9006); IoU 0,8476; Precision (0,9256) cao hơn Recall (0,9116), nghĩa là mô hình ít tô thừa hơn là bỏ sót vùng gan.

So với bài #3 (CirrMRI600+), Dice của các mô hình 2D trên T2W là 0,69–0,77 và 3D tốt nhất là 0,8651. Dice của nhóm cao hơn, nhưng phép so sánh *không công bằng* vì bài #3 có thể tính Dice theo khối bệnh nhân và khác chiến lược huấn luyện (số epoch, augmentation), còn nhóm tính theo từng lát.

So với bài #2 (Nowak 2021), Dice 0,983 cao hơn Dice của nhóm. Phép so sánh *không công bằng* vì bài #2 chỉ dùng một lát ở thùy đuôi, nơi gan lớn và rõ, còn nhóm dùng cả các lát ở rìa gan nơi gan rất nhỏ (có lát Dice bằng 0).

Phân loại: chưa có kết quả nên chưa đối chiếu được. Khi có kết quả cần đối chiếu với bài #2 (Acc 0,96, AUC 0,99, nhị phân, một lát mỗi bệnh nhân) và, nếu làm 4 lớp, với bài #4 (Acc 0,638 cho 3 lớp T2W), kèm nêu rõ khác dataset, khác số lát và khác cách chia.

Kết luận đối chiếu: phần phân vùng của nhóm có kết quả ổn định giữa các nhóm bệnh (Dice 0,894–0,903) nhưng bị kéo xuống bởi một số ít lát gan rất nhỏ. Điểm mạnh của dự án là chia dữ liệu theo bệnh nhân, kiểm tra không rò rỉ, và nhận diện sớm nguy cơ mô hình học lối tắt.

### B4. Timeline đến khi hoàn thành

Mốc: M3 tuần 10 · M4 tuần 12 · Final tuần 15. Cột "Ngày dự kiến" ghi theo hạn của mốc; nhóm thay bằng ngày cụ thể nếu có.

|Việc|Phụ trách (MSSV / vai trò)|Mốc|Ngày dự kiến|Trạng thái (Chưa / Đang / Xong + bằng chứng)|
|-|-|-|-|-|
|Chia dữ liệu theo bệnh nhân, kiểm tra không rò rỉ|20233772 + 20233735 — đồng thực hiện|M3 tuần 10|Xong|Xong — bảng chia dữ liệu (train 300 BN, valid 36 BN, test 37 BN) và kết quả giao các tập = 0 trong notebook|
|Huấn luyện và đánh giá U-Net phân vùng gan|20233772 + 20233735 — đồng thực hiện|M3 tuần 10|Xong|Xong — Dice valid 0,9561 (epoch 9); Dice test 0,8994|
|Lọc lát không có gan, tạo CSV cho phân loại|20233772 + 20233735 — đồng thực hiện|M3 tuần 10|09/11/2026|Đang — đã xác định trên test có 83 lát không có gan (slide Bảng 3); file CSV phân loại [CẦN BẠN XÁC NHẬN đã tạo chưa]|
|Phân loại nhị phân ResNet18 cấu hình A và B|20233772 + 20233735 — đồng thực hiện|M3 tuần 10|Hạn M3 (tuần 10)|Đang — đã thiết kế quy trình (slide mục 4); chưa có kết quả|
|Calibration, Brier, 95% CI bằng bootstrap theo bệnh nhân|20233772 + 20233735 — đồng thực hiện|M4 tuần 12|Hạn M4 (tuần 12)|Chưa|
|Grad-CAM kiểm tra lối tắt|20233772 + 20233735 — đồng thực hiện|M4 tuần 12|Hạn M4 (tuần 12)|Chưa|
|Phân loại 4 lớp (khỏe / nhẹ / vừa / nặng)|20233772 + 20233735 — đồng thực hiện|M4 tuần 12|Hạn M4 (tuần 12)|Chưa|
|Dice theo khối từng bệnh nhân để so với bài CirrMRI600+|20233772 + 20233735 — đồng thực hiện|M4 tuần 12|Hạn M4 (tuần 12)|Chưa|
|Hoàn thiện FAIR (datasheet, checksum), checklist TRIPOD+AI|20233772 + 20233735 — đồng thực hiện|M4 tuần 12|Hạn M4 (tuần 12)|Chưa|
|Nộp final, báo cáo và Pull Request|20233772 + 20233735 — đồng thực hiện|Final tuần 15|Hạn Final (tuần 15)|Chưa|

### B5. Rủi ro / đang vướng / câu hỏi cho GV

* Nhóm khỏe mạnh chỉ có 55 bệnh nhân (6 bệnh nhân ở tập test), nên Specificity ở mức bệnh nhân dao động mạnh.
* Khoảng 38% lát nhóm khỏe không có gan (thống kê trên mẫu 400 ảnh mỗi nhóm; trên tập test là 83/238 lát, khoảng 34,9%), trong khi nhóm xơ gan không có lát nào như vậy; nguy cơ mô hình học "không có gan thì là khỏe". Nhóm sẽ loại các lát này khỏi bước phân loại.
* Ảnh nhóm khỏe do nhóm tự tách từ NIfTI và chuẩn hóa min–max từng lát; chưa chắc cùng điều kiện với ảnh PNG có sẵn của nhóm xơ gan.
* Mô hình mới được kiểm định nội bộ trên một bộ dữ liệu, chưa có kiểm định ngoài.
* Câu hỏi cho GV: nhóm làm phân loại nhị phân trước rồi mở rộng sang 4 lớp trong Mốc 4, có phù hợp không?

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu

|Mục|Trả lời|Bằng chứng|
|-|-|-|
|Nguồn, phiên bản, ngày tải; checksum file dùng để huấn luyện|CirrMRI600+ (Jha et al., Scientific Data 2025;12:896, công bố 28/05/2025), tải từ OSF https://osf.io/cuk24/. Ngày tải: [20/9/2026]. SHA-256 của file zip dùng để huấn luyện:[chạy sha256sum <Cirrhosis_T2_2D>.zip ]|https://www.nature.com/articles/s41597-025-05201-7|
|Giấy phép / DUA, căn cứ được phép dùng|CC BY-SA 4.0 theo bài dữ liệu (cần xác nhận lại trên trang OSF khi nộp); dữ liệu công khai nên dùng cho nghiên cứu, phải ghi nguồn và giữ cùng giấy phép khi chia sẻ lại sản phẩm phái sinh|https://osf.io/cuk24/|
|Datasheet: quần thể, tiêu chí chọn/loại, định nghĩa và đơn vị biến, giới hạn|Quần thể: 373 BN, ảnh T2W (318 xơ gan có mask gan và mức độ 1–3 từ "Radiological Evaluation", gồm 131 nhẹ, 114 vừa, 73 nặng; 55 khỏe mạnh có mask gan). Ảnh xơ gan là PNG 2D chia sẵn train/valid/test (6.766 lát); ảnh khỏe là NIfTI 3D nhóm tự tách thành 2.143 lát (thang 0–255, mask nhị phân); tổng 8.909 lát. Dữ liệu gốc thu tại Đại học Istanbul (một trung tâm, nhiều hãng máy). Giới hạn: nhãn mức độ từ đọc ảnh, không phải sinh thiết; nhóm khỏe nhỏ; một nguồn dữ liệu|`CirrMRI600_full_dataset.csv` 
|Không có định danh bệnh nhân|Nhóm chỉ dùng mã dạng cirr_xxx, healthy_xx; không đưa ảnh hay họ tên vào repo. Theo bài dữ liệu, dữ liệu gốc đã được ẩn danh theo yêu cầu IRB, loại bỏ thông tin sức khỏe cá nhân (PHI) và chuyển từ DICOM sang NIfTI|https://www.nature.com/articles/s41597-025-05201-7|

### C2. Hướng dẫn báo cáo

* Câu hỏi nghiên cứu của nhóm: Với ảnh MRI T2W vùng bụng, mô hình học sâu hai giai đoạn (U-Net phân vùng gan, ResNet18 phân loại) có phân biệt được gan khỏe và gan xơ (về sau là 4 mức) không, và bước phân vùng có cải thiện so với dùng ảnh gốc không?
* Hướng dẫn báo cáo phù hợp: *TRIPOD+AI*, vì đề tài phát triển và đánh giá một mô hình dự đoán chẩn đoán (Phụ lục §3.2). Không dùng STARD-AI vì nhóm không đánh giá một xét nghiệm AI có sẵn so với chuẩn tham chiếu; không dùng DECIDE-AI vì chưa đánh giá lâm sàng giai đoạn sớm.
* Số mục checklist đã đáp ứng / tổng số mục; các mục còn thiếu: [CẦN BẠN ĐIỀN: tải checklist TRIPOD+AI, đếm thật; hiện checklist chưa điền].
* Kết cục, nhãn, thời điểm dự đoán T0, biến có sẵn tại T0: kết cục là tình trạng xơ gan (0/1) và mức độ (0 khỏe, 1 nhẹ, 2 vừa, 3 nặng); T0 là lúc có ảnh MRI T2W; biến dự báo chỉ gồm ảnh MRI (không dùng tuổi, giới, xét nghiệm).

### C3. PROBAST+AI — tự thẩm định dự án của nhóm

|Miền|Nguy cơ sai lệch (thấp / cao / không rõ)|Căn cứ trong repo (file, dòng)|Việc sẽ làm để giảm nguy cơ|
|-|-|-|-|
|Người tham gia và nguồn dữ liệu|Cao|Hai nhóm xử lý khác nhau (xơ gan: PNG có sẵn; khỏe: nhóm tự tách từ NIfTI); tỉ lệ khỏe/xơ gan (15%/85% theo bệnh nhân) do bộ dữ liệu quyết định, không phản ánh tỉ lệ thực. [file code]|Thống kê độ sáng, kích thước, tỉ lệ gan theo nhóm; Grad-CAM; nêu rõ hạn chế|
|Phân tích (chia theo bệnh nhân, fit chỉ trên train, cỡ mẫu, dữ liệu thiếu)|Không rõ|Có: chia theo bệnh nhân (train 300 / valid 36 / test 37 BN) và kiểm tra giao các tập = 0; chuẩn hóa không fit từ dữ liệu (min–max từng lát, thống kê ImageNet cố định); class weight tính chỉ từ train. Thiếu: calibration, khoảng tin cậy, chứng minh cỡ mẫu; test nhóm khỏe chỉ 6 BN. [file code]|Bootstrap theo bệnh nhân lấy 95% CI; calibration và Brier; báo cáo theo bệnh nhân|

Tính áp dụng (quần thể, bối cảnh dùng mô hình): thấp. Dữ liệu một nguồn, nhóm khỏe nhỏ, chưa có kiểm định ngoài nên chưa đủ để dùng trong lâm sàng.

### C4. Bảng kiểm nhanh (Phụ lục §7)

|Mục|Đạt / Chưa|Bằng chứng|
|-|-|-|
|Datasheet|Một phần|Mục C1 (bản tóm tắt); chưa có file datasheet riêng|
|Phiên bản + checksum dữ liệu|Chưa|Có nguồn và bài công bố (mục C1); SHA-256: [CẦN BẠN ĐIỀN sau khi chạy `sha256sum`]|
|Không định danh; căn cứ được phép dùng dữ liệu|Đạt một phần|Dữ liệu công khai, đã ẩn danh theo bài dữ liệu; giấy phép CC BY-SA 4.0 (mục C1)|
|Chọn đúng hướng dẫn báo cáo + checklist đã điền|Một phần|Đã chọn TRIPOD+AI (mục C2); checklist chưa điền|
|Định nghĩa kết cục, nhãn, T0|Đạt|Mục C2|
|Chia theo bệnh nhân; tiền xử lý fit chỉ trên train|Đạt|Output kiểm tra giao các tập = 0 trong notebook; mục C3|
|Mô hình mốc|Chưa|Chưa có kết quả; dự kiến dùng cấu hình A (ResNet18 trên ảnh gốc) làm mốc cho bài toán ảnh, vì hồi quy logistic trên pixel không phù hợp|
|AUC + calibration + Brier + 95% CI|Chưa|Hiện mới có Dice cho phân vùng; phần phân loại đang thực hiện, chưa có kết quả|
|Cỡ mẫu và giới hạn về tính áp dụng|Chưa|Mới nêu giới hạn (mục B5, C3)|

## D. Khai báo dùng AI

|Công cụ|Dùng cho phần nào|Nhóm đã kiểm chứng lại bằng cách nào|
|-|-|-|
|Claude (Anthropic)|Hỗ trợ viết code Colab (tải dữ liệu, đánh giá, phân loại), giải thích khái niệm, hướng đẫn đọc và tóm tát các bài báo|Đối chiếu số liệu phân vùng với slide báo cáo; đọc lại abstract các bài #2, #3, #4, #7, #9, #10|

