# Bài kiểm tra lần 2 — N33 — Predicting Breast Cancer Outcomes (DxAI)

- Fork: . <URL: 'https://github.com/imissh3r-tester/hmyt-project/tree/Breast_Cancer_N33'> · Nhánh: Breast_Cancer_N33
- Thành viên (MSSV) và vai trò:
  - 202414830 (Trần Hải Đăng) — Pipeline tiền xử lý (ColumnTransformer, khử trùng lặp, assertion chống leakage, SMOTETomek), viết phần EDA, Preprocessing và đánh giá mô hình
  - 202414846 (Đặng Gia Khánh) — EDA và trực quan hóa, tổng hợp ma trận y văn, kiểm tra trích dẫn và định dạng báo cáo IEEE



---

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
| Nguồn | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|---|---|---|---|---|
| PubMed |  (vd: breast cancer AND survival AND machine learning AND SEER) | 2021–2026 | . | . |
| Scopus |  | 2021–2026 |  |  |
| Google Scholar |  | 2021–2026 |  |  |
| Tổng sau loại trùng | | | | 20 bài trong Bảng I của báo cáo IEEE (A2 dưới đây chọn 14 bài trọng tâm về mô hình tiên lượng/phân loại) |

Tiêu chí chọn / loại bài:
- Chọn: nghiên cứu dự đoán sống còn hoặc kết cục ung thư vú bằng ML/survival model, ưu tiên dùng SEER hoặc registry/bệnh viện có mô tả rõ cách chia dữ liệu; xuất bản 2021–2026.
- Loại khỏi A2 (vẫn có trong Bảng I của báo cáo): Soleimani et al. 2025 [16] (dữ liệu ung thư đại trực tràng, tham khảo phương pháp resampling); Siegel et al. 2026 [18] (báo cáo thống kê dịch tễ, không phải mô hình ML); Zhang et al. 2025 [13] (phân tích mô hình di căn bằng Kaplan–Meier/Cox); Massafra et al. 2022 [3] và Shui et al. 2026 [21] (dự đoán biến cố xâm lấn/tim mạch chuyên biệt); Islam et al. 2024 [8] (tập trung vào dữ liệu chẩn đoán hình thái tế bào).

### A2. Ma trận y văn
Số trong ngoặc vuông `[n]` khớp chính xác với danh mục tài liệu tham khảo (References) trong báo cáo IEEE.

| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? | Hạn chế chính |
|---|---|---|---|---|---|---|---|
| 1 | Moncada-Torres et al., 2021 [1] | 10.1038/s41598-021-86327-7 | Netherlands Cancer Registry, n = 36,658 | CPH, RSF, Survival SVM, XGBoost; chia ở mức bệnh nhân, 10-fold CV | XGBoost C-index ≈ 0.73, CPH ≈ 0.72 | Không | Dữ liệu một quốc gia, hồi cứu; dữ liệu thô không chia sẻ được |
| 2 | Sedighi-Maman & Mondello, 2021 [2] | 10.1016/j.ijmedinf.2021.104438 | SEER 2004–2016 | Hai giai đoạn: phân loại (GLM-LASSO, XGB, MLP + SMOTE) rồi hồi quy số tháng sống | Giai đoạn I (1 năm): GLM-LASSO-SMOTE AUC 0.898, Acc 0.821; giai đoạn II: MAE 7.96 tháng | Không | SEER thiếu biến lâm sàng chi tiết; MLP kém GLM ở giai đoạn hồi quy |
| 3 | Manikandan et al., 2023 [5] | 10.1038/s41598-023-32029-1 | SEER Breast, n = 712,319 bản ghi | Chọn đặc trưng (Variance Threshold + PCA) + AdaBoost, XGB, GB, NB, DT; chia theo bản ghi 70:30 | DT: Acc, Precision, Recall, F1 đều 0.98 | Không | Nhiều dữ liệu thiếu; rủi ro rò rỉ dữ liệu nghiêm trọng do chạy PCA toàn cục trước khi chia tập |
| 4 | Gurcan & Soylu, 2024 [7] | 10.3390/cancers16193417 | 5 bộ ung thư; SEER Breast n = 4,024 | 10 bộ phân loại × 19 thuật toán resampling; 5-fold CV phân tầng | SEER Breast, SMOTEENN: Acc 92.87%, F1 92.88%, ROC-AUC 97.71% | Không | Chỉ tập trung resampling; không có XAI và không kiểm tra hiệu chuẩn xác suất |
| 5 | Rajpoot et al., 2024 [9] | 10.1080/08839514.2024.2340386 | UCI ML Repository (Breast Cancer) | RF, DT, SVM, ANN + chọn đặc trưng FS-ML; 10-fold CV theo bản ghi | Mô hình đề xuất: Sens 85.5%, Spec 79.3%, Precision 87% | Không | Dataset benchmark tế bào học UCI, không phải cohort sống còn dọc (EHR/SEER); không XAI |
| 6 | Manir & Deshpande, 2024 [10] | 10.3390/diagnostics14100984 | BCSC n = 527,712; WBCD n = 569; SEER Breast n = 4,024 | Extra Trees, RF, XGB, LR, DT, KNN + PCA + SMOTE/Tomek; 70/30 + 10-fold CV | BCSC Extra Trees Acc 87.54%; WBCD KNN+PCA AUC 0.9796 | Không | Mất cân bằng nặng ở BCSC; PCA phá hủy khả năng diễn giải lâm sàng của biến gốc |
| 7 | Alshamrani, 2025 [11] | 10.3390/electronics14163173 | Dữ liệu tổng hợp n = 1,000; SEER n = 4,024 | Box–Cox + LR, SVM, RF, XGB, voting, stacking; 80/20 + SMOTE trên train | SEER: Stacking + Box-Cox + SMOTE Acc 94.53% | Không | Box-Cox đòi hỏi giá trị dương nghiêm ngặt; không XAI và thiếu kiểm toán rò rỉ hậu kết cục |
| 8 | Baidoo & Rodrigo, 2025 [12] | 10.1371/journal.pone.0318167 | SEER Breast n = 4,024 bệnh nhân nữ | CPH, RSF, DeepSurv so với RF, XGB, SVM-RBF, LightGBM + SMOTE | RSF C-index 0.72 ± 0.0117; RF AUC 0.74 ± 0.0021, Acc 0.82; XGB AUC 0.69 | Không | Mô hình ML phân loại chuẩn không xử lý tự nhiên dữ liệu bị cắt cụt phải (right-censoring) |
| 9 | Noman et al., 2025 [15] | 10.1038/s41598-025-87622-3 | METABRIC, MSKCC, Duke, SEER (n = 272,252) + Ai Cập | RFS + LightGBM, XGB, RF dự đoán tái phát/di căn xa | C-index 0.837; LightGBM tái phát AUC 92% | Có (cohort Ai Cập) | Tỷ lệ phân loại sai còn ≈ 13.55% đối với di căn xa |
| 10 | Raz et al., 2026 [17] | 10.1186/s12885-026-16283-w | Bệnh viện Hazrat Rasool, n = 3,476, 43 biến | XGB, LR, RF, DT, AdaBoost, MLP + SMOTE; nested 5-fold CV | XGB/DT/AdaBoost AUC 0.97, Acc > 90% | Không | Dữ liệu hồi