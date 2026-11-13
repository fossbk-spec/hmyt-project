# Bài kiểm tra lần 2 — N01 — edge_ai_da_benh_ly

- Fork:(https://github.com/dotrang7825-prog/hmyt-project.git) · Nhánh: gd5-proposed-model
- Thành viên (MSSV hoặc alias) và vai trò:
+ 20233786: Phụ trách EDA, Evaluation & Edge
+ 20233814: Phụ trách Data Pipeline & Model PyTorch

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
| Nguồn (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|arXiv preprint, Google Scholar, IEEE Xplore, PubMed |ảnh đáy mắt, edge AI đa bệnh lý, ODIR-5K|5 năm gần đây|~80|15|

Tiêu chí chọn / loại bài:
- 5 năm gần đây
- cùng bộ dataset
- cùng bài toán 

### A2. Ma trận y văn (≥ 10 bài, ưu tiên 3–5 năm gần nhất)
| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? (có / không) | Hạn chế chính |
|1|Leveraging fundus images for on device eye disease diagnosis with AI powered lightweight software hardware framework (Prajwal Thaware, A. A. Khurshid, Chetan Sonkusare, Yash Agrawal, 2025)|10.1038/s41598-025-20990-y|13.300 ảnh (EyePACS, Messidor-2, IDRiD, REFUGE)|EfficientNet-B0 + Spatial Attention, CLAHE, Grad-CAM, lượng tử hóa TFLite on-device|Accuracy = 96,2% (8 nhãn); F1 = 0,958; Latency ~5s/ảnh trên Raspberry Pi 5|Có|Phải nhỏ thuốc giãn đồng tử; FOV hẹp (56°) dễ bỏ sót tổn thương ngoại vi|
|2|Multi-label Retinal Disease Classification Using Transformers (Bản thảo Preprint arXiv) (M. A. Rodríguez, H. AlMarzouqi, P. Liatsis) (2022)|https://doi.org/10.48550/arXiv.2207.02335|Tập dữ liệu MuReD (2.208 ảnh, 20 nhóm nhãn bệnh, hợp nhất từ ARIA, STARE, RFMiD)|C-Tran (DenseNet161 + Transformer Encoder), LMT, tái lấy mẫu LP ROS (10%), PolyLoss|ML-AUC = 0,962; Model Score = 0,900; ML-mAP = 0,685|Không|Lớp cực hiếm (HTR, ASR) có F1 = 0 do thiếu mẫu; chỉ thử nghiệm ảnh tĩnh|
| 3 | Multi-Label Retinal Disease Classification Using Transformers (Công bố chính thức IEEE JBHI) (Manuel Alejandro Rodríguez, Hasan AlMarzouqi, Panos Liatsis) (2023) | 10.1109/JBHI.2022.3214086 | 2.208 ảnh MuReD (lọc mờ Canny) | C-Tran (DenseNet161 + Transformer), LMT, PolyLoss, Grad-CAM | ML-AUC = 0,962; Model Score = 0,824; Bin-F1 = 0,824 | Không | Chưa thử nghiệm thực tế trên phần cứng nhúng biên; phụ thuộc lớn vào CNN backbone |
| 4 | Federated Learning for Diabetic Retinopathy Detection in a Multi-center Fundus Screening Network (Sarah Matta, Mariem Ben Hassine, Clément Lecat, Laurent Borderie, Alexandre Le Guilcher, Pascale Massin, Béatrice Cochener, Mathieu Lamard, Gwenolé Quellec) (2023) | 10.1109/embc40787.2023.10340772 | 697.275 ảnh (96.023 bệnh nhân, OPHDIAT) | EfficientNet-B5 kết hợp Federated Averaging (FedAvg), phân tách cross-center & cross-grader | Cross-grader FL đạt AUC = 0,9522 (vượt Centralized CL: 0,9482) | Có (43 trung tâm sàng lọc) | Mô phỏng FL trên dữ liệu tập trung sẵn; thời gian huấn luyện mỗi vòng lớn (~78h) |
| 5 | A Federated Machine Learning Approach for the Detection and Visualisation of Eye Diseases Using Activation Maps (Filomena Niro, Miriam Di Renzo, Patrizia Agnello, Marta Petyx, Fabio Martinelli, Maurizio Maddalena, Mario Cesarelli, Antonella Santone, Francesco Mercaldo) (2026| 10.3390/s26165288 | 1.293 ảnh (Kaggle: Normal, Cataract, Glaucoma) | Vision Transformer (ViT-tiny đóng băng backbone), FedProx/FedAvg, Grad-CAM cho ViT | FedProx (IID): Acc = 0,8480, AUC = 0,9641; Non-IID: Acc = 0,8000 | Không | Quy mô dữ liệu nhỏ; chưa đánh giá định lượng cấu trúc bản đồ nhiệt |
| 6 | Federated learning for privacy-preserving ophthalmic artificial intelligence: clinical applications and translational challenges (Yuan Wei, Kaikai Zhao, Andrzej Grzybowski, Kai Jin), (2026) | 10.3389/fmed.2026.1904004 | Tổng quan y văn đa nguồn (Fundus, OCT, OCTA) | Khảo sát hệ thống FL (Horizontal/Vertical), Differential Privacy, Homomorphic Encryption | Glaucoma OCT FL đạt AUC = 0,92; xác nhận tính khả thi lâm sàng của FL | Có (Tổng hợp đa nghiên cứu) | Bài báo dạng Narrative Review, thiếu đánh giá định lượng meta-analysis |
| 7 | Federated Learning for Diabetic Retinopathy Diagnosis: Enhancing Accuracy and Generalizability in Under-Resourced Regions (Gajan Mohan Raj, Michael G. Morley, Mohammad Eslami) (2024)| 10.48550/arXiv.2411.00869 | 27.286 ảnh (DDR, EyePACS, APTOS) | EfficientNet-B0 + FedAvg (TensorFlow Federated); đóng gói Flask & iOS app | Accuracy = 93,21% trên tập test độc lập; đạt 91,05% trên ảnh kém chất lượng ở H3 | Có (3 cơ sở y tế độc lập) | Chưa tích hợp mã hóa nâng cao chống tấn công gradient inversion; chưa đo trễ mạng |
| 8 | Federated Learning for Computer-Aided Diagnosis of Glaucoma Using Retinal Fundus Images (Telmo Baptista, Carlos Soares, Tiago Oliveira, Filipe Soares) (2024) | 10.3390/app132111620 | ORIGA, REFUGE, GAMMA, Drishti, AIROGS | Flower framework, khảo sát FedAvg, FedProx, FedBN, lọc chất lượng ảnh, Jetson AGX Xavier | FedProx và FedSGD tổng quát hóa tốt nhất trên dữ liệu phân tán; Acc đạt 0,873 | Có | Chỉ khảo sát trên 1 kiến trúc CNN nền tảng; API chưa có chứng chỉ SSL |
| 9 | FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings (Jean Ogier du Terrail, Samy-Safwan Ayed, Edwige Cyffers, et al.) (2023)| 10.48550/arXiv.2210.04620 | 7 tập dữ liệu y tế phân chia tự nhiên (FLamby) | Benchmark FLamby (PyTorch), tích hợp FedAvg, FedProx, Scaffold, FedOpt | Thuật toán FedOpt chiếm ưu thế vượt trội so với mô hình cục bộ trên toàn bộ bài toán y tế | Có (Đa trung tâm y tế) | Hiện chỉ hỗ trợ PyTorch; chưa tối ưu hóa ràng buộc băng thông viễn thông khắc nghiệt |
| 10 | Pushing the Limits of Machine Design: Automated CPU Design with AI (Shuyao Cheng, Pengwei Jin, Qi Guo, Zidong Du, Rui Zhang, Yunji Chen, et al.) (2023)| 10.48550/arXiv.2306.12456 | Tập lệnh RISC-V 32IA (< 2^40 cặp mẫu I/O) | Binary Speculation Diagram (BSD), Monte Carlo, rút gọn nút theo Boolean distance | Tự thiết kế chip CPU trong < 5 giờ (nhanh hơn ~1000 lần); độ chính xác > 99,99% | Không | Hiệu năng đơn nhân còn khoảng cách so với chip hiện đại; phụ thuộc tập kiểm thử |
| 11 | Privacy-Preserving Federated Learning With Domain Adaptation for Multi-Disease Ocular Disease Recognition (Z. Tang, H. S. Wong, Z. Yu) (2024)| 10.1109/JBHI.2023.3305685 | ODIR-5K và dữ liệu đáy mắt đa nguồn | Federated Learning kết hợp Thích ứng miền (Domain Adaptation) xử lý domain shift | Hiệu năng AUC và F1 vượt trội FedAvg cơ bản, tiệm cận mô hình học tập trung (Centralized) | Có (Đa nguồn thiết bị chụp) | Độ phức tạp tính toán của mô-đun thích ứng miền cao khi chạy trên thiết bị biên yếu |
| 12 | Highly Efficient Homomorphic Encryption-Based Federated Learning for Diabetic Retinopathy Classification (Christopher Nielsen, Matthias Wilms, Nils D. Forkert) (2025)| 10.1117/1.JMI.12.3.034504 | ODIR-5K (1.399 train, 465 val, 931 test) và APTOS-2019 | RETFound trích xuất đặc trưng biên + FL tích hợp Mã hóa đồng cấu (Homomorphic Encryption) | AUROC ~0,78 trên ODIR-5K, ~0,93 trên APTOS; kháng hoàn toàn tấn công gradient | Có | Chủ yếu đánh giá trên bài toán DR trích xuất từ ODIR, chưa mở rộng đủ 8 nhãn |
| 13 | CD-FL: Cataract Images Based Disease Detection Using Federated Learning (Uzma Ghulam Mohammad, Saqib Imtiaz, et al.) (2023)| 10.32604/csse.2023.039296 | ODIR (Ocular Disease Recognition - 6.392 ảnh) | CNN (VGG16 tinh gọn) kết hợp Federated Learning phân tán client-server (mô phỏng 2 client) | Accuracy đạt 95,0% (Client 1) và 96,0% (Client 2); ROC-AUC đạt 0,987 | Không | Chỉ tập trung vào phát hiện đục thủy tinh thể (Cataract); số client mô phỏng ít (2 clients) |
| 14 | Enhanced Privacy-Preserving Architecture for Fundus Disease Diagnosis with Federated Learning (Raymond Jiang, Yulia Kumar, Dov Kruger) (2025) | 10.3390/app15063004 | ODIR (8 nhóm bệnh lý đáy mắt) | Kiến trúc Federated Learning tối ưu hóa cho nút biên (Edge clients), phân cụm tổng hợp, giảm băng thông | Duy trì độ chính xác cao trên client phân tán, giảm thiểu chi phí băng thông truyền thông | Không | Chưa đo kiểm trực tiếp trên phần cứng nhúng vật lý; đánh giá hoàn toàn bằng mô phỏng |
| 15 | FEDEYE: A Scalable and Flexible End-to-End Federated Learning Platform for Ophthalmology (B. Yan, D. Cao, X. Jiang, Y. Chen, W. Dai, F. Dong, et al.), (2024) | 10.1016/j.patter.2024.100928 | Dữ liệu nhãn khoa đa nguồn gồm ODIR, Fundus và OCT đa trung tâm | Nền tảng FL end-to-end chuyên biệt cho nhãn khoa; hỗ trợ đa kiến trúc mạng và quản trị phân tán | Triển khai thành công nền tảng phân tán ổn định trên mạng lưới phòng khám thực tế | Có (Mạng lưới đa trung tâm) | Tập trung nhiều vào kiến trúc hệ thống phần mềm hơn là tối ưu hóa toán học hàm mất mát |
### A3. Research gaps
| Mã | Khoảng trống (1–2 câu) | Bằng chứng (số # trong A2) | Dự án của nhóm giải quyết bằng cách nào |
| G1 | Phần lớn các nghiên cứu phân loại đa nhãn trên tập dữ liệu ảnh đáy mắt ODIR-5K hiện nay chủ yếu tập trung vào mô hình học tập trung (Centralized Learning) với chi phí tính toán lớn, chưa tối ưu hóa trọng số mô hình để triển khai suy luận thời gian thực trên thiết bị biên. | #2, #3, #11 | Nhóm xây dựng hệ thống Edge AI sử dụng các kiến trúc backbone gọn nhẹ (MobileNetV3 hoặc EfficientNet kết hợp Transformer) và áp dụng kỹ thuật lượng tử hóa (Quantization) từ FP32 xuống INT8 để tối ưu tốc độ suy luận (FPS/Latency) trên thiết bị biên. |
| G2 | Các hệ thống học liên kết (Federated Learning) xử lý bệnh lý đáy mắt thường giả định môi trường dữ liệu đồng nhất (IID) và chưa giải quyết triệt để vấn đề lệch phân phối (domain shift) do sự khác biệt về thiết bị chụp và điều kiện ánh sáng giữa các cơ sở y tế. | #4, #7, #13 | Nhóm áp dụng các bước tiền xử lý chuẩn hóa cường độ sáng (CLAHE), xử lý mất cân bằng nhãn bằng Weighted Random Sampling và thiết lập kịch bản phân chia dữ liệu Non-IID để tăng khả năng tổng quát hóa của mô hình phân tán. |

### A4. Thẩm định 1 bài quan trọng nhất trong A2 bằng PROBAST+AI
Bài số # : Bài 15 (Yan et al., 2024 - Patterns / Cell Press) · Lý do chọn: Là nghiên cứu tiên tiến và toàn diện nhất về nền tảng học liên kết (Federated Learning) end-to-end chuyên biệt cho nhãn khoa, cung cấp kiến trúc hệ thống phân tán thực tế phù hợp nhất với định hướng kỹ thuật của nhóm.

| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ (trích dẫn hoặc số trang trong bài) |
| Người tham gia và nguồn dữ liệu | Thấp | Sử dụng dữ liệu nhãn khoa đa nguồn bao gồm ODIR, Fundus và OCT thu thập từ mạng lưới đa trung tâm y tế thực tế, đảm bảo tính đại diện cao cho dữ liệu lâm sàng phân tán. |
| Biến dự báo | Thấp | Hệ thống trích xuất và chuẩn hóa các đặc trưng hình ảnh đáy mắt và OCT qua đa kiến trúc mạng nơ-ron sâu được cấu hình đồng bộ trên toàn bộ các client tham gia. |
| Kết cục | Thấp | Đánh giá toàn diện các chỉ số hiệu năng hệ thống phân tán, độ ổn định mô hình và tốc độ đồng bộ trọng số giữa các trung tâm y tế. |
| Phân tích | Thấp | Xây dựng nền tảng quản trị mô hình phân tán hoàn chỉnh, tối ưu hóa quá trình huấn luyện bảo mật giữa các nút biên mà không làm rò rỉ dữ liệu gốc của bệnh nhân. |
Tính áp dụng cho bài toán của nhóm: Rất cao. Cung cấp mô hình kiến trúc hệ thống và luồng dữ liệu chuẩn (end-to-end framework) để nhóm tham khảo khi thiết kế hệ thống phân tán, quản trị client và tối ưu hóa quy trình học liên kết cho bài toán phân loại đa bệnh lý ảnh đáy mắt.

## B. Kết quả và timeline

### B1. Kết quả hiện có
| Mô hình | Tập đánh giá (cách chia, n) | Metric chính (giá trị, 95% CI nếu có) | Metric phụ | Nguồn số liệu (file / notebook / W&B) |
| EfficientNet | Validation(15% trong tổng số ảnh)| | | File: train.py (Phiên bản Mốc 2) - Log lịch sử chạy huấn luyện trên W&B Dashboard (Run: efficientnet_baseline) |
| EfficientNet-B0 + Focal Loss nâng cao + Medical-grade Transforms | Tập Test độc lập (15% còn lại từ tổng dataset, n = 1,117 ảnh, 60 batches) | | | File trọng số: checkpoints/best_model.pth - Script xuất báo cáo: evaluate.py - Bảng kết quả Classification Report chi tiết cho 8 nhãn bệnh. |

### B2. Ablation (Mốc 3)
| Cấu hình (bỏ / đổi thành phần nào) | Metric chính | Chênh lệch so với mô hình đầy đủ | Nguồn |
|---|---|---|---|

### B3. Đối chiếu với y văn
So sánh kết quả B1 với 2–3 bài trong A2. Nêu rõ những điểm làm phép so sánh **không công bằng**, nếu có: khác dataset, khác cách chia, khác metric.

### B4. Timeline đến khi hoàn thành
| Việc | Phụ trách (MSSV / vai trò) | Mốc (M3 tuần 10 · M4 tuần 12 · Final tuần 15) | Ngày dự kiến | Trạng thái (Chưa / Đang / Xong + bằng chứng) |
|---|---|---|---|---|

### B5. Rủi ro / đang vướng / câu hỏi cho GV
- 

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu
| Mục | Trả lời | Bằng chứng |
|---|---|---|
| Nguồn, phiên bản, ngày tải; checksum file dùng để huấn luyện | | |
| Giấy phép / DUA, căn cứ được phép dùng | | |
| Datasheet: quần thể, tiêu chí chọn/loại, định nghĩa và đơn vị biến, giới hạn | | |
| Không có định danh bệnh nhân | | |

### C2. Hướng dẫn báo cáo
- Câu hỏi nghiên cứu của nhóm:
- Hướng dẫn báo cáo phù hợp (TRIPOD+AI / STARD-AI / DECIDE-AI) và lý do (Phụ lục §3.2):
- Số mục checklist đã đáp ứng / tổng số mục; các mục còn thiếu:
- Kết cục, nhãn, thời điểm dự đoán T0, các biến có sẵn tại T0:

### C3. PROBAST+AI — tự thẩm định dự án của nhóm
| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ trong repo (file, dòng) | Việc sẽ làm để giảm nguy cơ |
|---|---|---|---|
| Người tham gia và nguồn dữ liệu | | | |
| Biến dự báo | | | |
| Kết cục (chú ý incorporation bias) | | | |
| Phân tích (chia theo bệnh nhân, fit chỉ trên train, cỡ mẫu, dữ liệu thiếu) | | | |
Tính áp dụng (quần thể, bối cảnh dùng mô hình):

### C4. Bảng kiểm nhanh (Phụ lục §7)
| Mục | Đạt / Chưa | Bằng chứng |
|---|---|---|
| Datasheet | | |
| Phiên bản + checksum dữ liệu | | |
| Không định danh; căn cứ được phép dùng dữ liệu | | |
| Chọn đúng hướng dẫn báo cáo + checklist đã điền | | |
| Định nghĩa kết cục, nhãn, T0 | | |
| Chia theo bệnh nhân; tiền xử lý fit chỉ trên train | | |
| Mô hình mốc hồi quy logistic (hoặc mốc phù hợp, nêu lý do) | | |
| AUC + calibration (đường cong, in-the-large, slope) + Brier, kèm 95% CI | | |
| Cỡ mẫu có đủ không; giới hạn về tính áp dụng | | |

## D. Khai báo dùng AI
| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
|---|---|---|