# Phân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt (Đề tài 2.2 — day_mat)

## 1. Đặt vấn đề

**Câu hỏi nghiên cứu:** MPhân loại mức độ bệnh võng mạc đái tháo đường (Diabetic Retinopathy) từ ảnh chụp đáy mắt bằng học sâu, có giải thích được (XAI)
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

- **Chia dữ liệu:** `train_test_split` 80/20, `stratify=death_event`,
  `random_state=42` → Train 239 ca (32,22% tử vong), Test 60 ca (31,67%
  tử vong) — tỷ lệ nhãn giữ đồng đều 2 tập.
- **Chuẩn hóa:** `StandardScaler` — `fit` **chỉ trên Train**, `transform`
  cho cả Train/Test (tránh data leakage).
- **Baseline:** Logistic Regression + Random Forest (100 cây), không
  SMOTE, đúng thiết kế gốc của `chicco2020machine`.
- **Cải tiến:** `SMOTE` (`fit_resample` **chỉ trên Train** — Train sau
  SMOTE: 324 mẫu, cân bằng 50/50) + Extra Trees (300 cây), theo
  `ishaq2021improving`.
- **Ablation bổ sung** (Buổi 3, nhiệm vụ 3): thử thêm GradientBoosting,
  Random Forest đã tinh chỉnh (500 cây/max_depth=8), và so sánh 3 tập đặc
  trưng (đủ 12 biến / bỏ `time` / chỉ 2 biến creatinine+EF) để hiểu rõ vai
  trò của biến `time`.
- **XAI:** `shap.TreeExplainer` trên mô hình Extra Trees+SMOTE, tính SHAP
  values cho lớp "tử vong" trên tập Test.

---

## 4. Kết quả

### 4.1. Baseline (không SMOTE) — đối chiếu `chicco2020machine`

| Mô hình | Accuracy | MCC | F1 | Precision | Recall | AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 81,7% | 0,556 | 66,7% | 78,6% | 57,9% | 85,9% |
| **Random Forest (baseline)** | **83,3%** | **0,600** | 70,6% | 80,0% | 63,2% | 89,1% |
| *Chicco 2020 (công bố)* | *74,0%* | *0,384* | — | — | — | — |

→ Random Forest baseline của ví dụ này **vượt** con số công bố gốc (83,3%
vs 74,0%; MCC 0,600 vs 0,384). **2 lý do chính:**

1. **Baseline ở đây dùng đủ 12 biến (gồm `time`)**, còn Bảng 4 của bài
   báo gốc **loại `time`** có chủ đích (xem Mục 2.1) — `time` là biến
   mạnh nhất (Mục 4.4), nên có nó gần như chắc chắn đẩy accuracy lên.
2. **N=299, test set chỉ 60 ca, ví dụ này chỉ chạy 1 split**
   (`random_state=42`) — trong khi bài báo gốc lấy **trung bình của 100
   lần chạy** với các split ngẫu nhiên khác nhau. 1 con số đơn lẻ trên tập
   nhỏ dao động rất mạnh theo may rủi của lần chia tập; trung bình 100 lần
   mới phản ánh đúng hiệu năng kỳ vọng.

Đây là điểm đáng đưa vào phần "Hạn chế" của báo cáo: **một con số accuracy
đơn lẻ trên tập nhỏ không đáng tin cậy nếu không có khoảng tin cậy (CI)
hoặc lặp lại nhiều lần (như chính bài báo gốc đã làm — 100 executions).**

### 4.2. Cải tiến với SMOTE — đối chiếu `ishaq2021improving`

| Mô hình | Accuracy | MCC | F1 |
|---|:---:|:---:|:---:|
| Extra Trees + SMOTE (300 cây, như lab yêu cầu) | 76,7% | 0,418 | 53,3% |
| Random Forest + SMOTE (100 cây) | 80,0% | 0,538 | 68,4% |
| **Random Forest + SMOTE (500 cây, max_depth=8 — tinh chỉnh)** | **85,0%** | **0,649** | **75,7%** |
| *Ishaq 2021 (công bố)* | *92,6%* | — | — |

→ **Không tiệm cận được 92,6%** dù đã tinh chỉnh (mốc gần nhất đạt 85,0%
test-set thật). Điều này **không có nghĩa pipeline sai** — xem phát hiện ở
Mục 4.3.

### 4.3. Phát hiện đáng chú ý: bẫy "CV sau SMOTE"

Khi đánh giá bằng 5-fold cross-validation **trên chính tập đã SMOTE**
(thay vì tập Test giữ nguyên gốc), Random Forest 500 cây cho
**accuracy 90,1% ± 4,4%** — rất gần 92,6% của Ishaq. Nhưng con số này
**lạc quan giả tạo**: khi SMOTE sinh mẫu tổng hợp trước rồi mới chia
CV-fold, một số mẫu tổng hợp ở fold validation được nội suy từ hàng xóm
gần nằm ở fold train → rò rỉ thông tin giữa các fold. Test set thật (chưa
từng bị SMOTE đụng vào) chỉ cho 85,0%.

**Giả thuyết hợp lý:** nhiều bài báo trong dòng nghiên cứu này (có thể
gồm cả `ishaq2021improving`) áp dụng SMOTE **trước** khi chia CV thay vì
chỉ trên fold train của từng vòng — đúng loại lỗi mà bản lab đã cảnh báo
đỏ ở Buổi 3 ("SMOTE tuyệt đối chỉ được áp dụng trên tập Train"). Nếu nhóm
bạn gặp tình huống tương tự (không tiệm cận được con số công bố dù đã làm
đúng quy trình), đây là một hướng giải thích hợp lý đáng đưa vào báo cáo
thay vì chỉ cố tinh chỉnh siêu tham số để "đuổi theo con số".

### 4.4. Ablation vai trò biến `time`

| Tập đặc trưng | Random Forest (không SMOTE) | Δ so với đủ 12 biến |
|---|:---:|:---:|
| Đủ 12 biến (gồm `time`) | Accuracy 83,3% · MCC 0,600 | — |
| Bỏ `time` (11 biến) | Accuracy 70,0% · MCC 0,251 | **−13,3 điểm % / −0,349 MCC** |
| Chỉ `serum_creatinine` + `ejection_fraction` (2 biến, đúng tiêu đề bài báo gốc) | Accuracy 71,7% · MCC 0,287 | −11,6 điểm % / −0,313 MCC |

→ `time` (số ngày theo dõi) một mình đóng góp phần lớn sức mạnh dự báo
của mô hình đủ đặc trưng — khớp với vị trí #1 tuyệt đối trong SHAP (Mục
5), và **khớp trực tiếp với phát hiện của chính bài báo gốc** (Bảng 11:
thêm `time` đẩy MCC từ 0,384 → 0,616, xem Mục 2.1).

Về mặt lâm sàng, `time` **không phải là biến biết trước tại thời điểm
nhập viện** — bản chất giống thời gian theo dõi/censoring hơn là một yếu
tố nguy cơ đo được lúc khám. Dùng nó làm đặc trưng dự báo có thể bị xem
là **rò rỉ thông tin tương lai (temporal leakage)** tùy theo cách đặt bài
toán — **điểm này bài báo gốc không bàn tới** (họ chỉ khẳng định `time`
"quan trọng", không đặt câu hỏi liệu dùng nó có hợp lý về mặt lâm sàng hay
không). Đây là một góc nhìn phản biện các nhóm có thể tự phát triển thêm
trong báo cáo của mình.

---

## 5. Khả năng giải thích (SHAP)

Mô hình: Extra Trees + SMOTE (đúng yêu cầu Buổi 4). File ảnh:
`shap_summary_bar.png`, `shap_summary_beeswarm.png` (thư mục `report/`).

| Hạng | Đặc trưng | Mean |SHAP value|| Ghi chú |
|:---:|---|:---:|---|
| 1 | `time` | 0,1559 | Chi phối mạnh nhất — xem thảo luận Mục 4.4 |
| 2 | `ejection_fraction` | 0,0759 | **Khớp y văn** — 1 trong 2 biến "quyết định" theo Chicco 2020 |
| 3 | `serum_creatinine` | 0,0689 | **Khớp y văn** — biến còn lại theo Chicco 2020 |
| 4 | `age` | 0,0303 | |
| 5 | `serum_sodium` | 0,0288 | |
| 6-12 | còn lại | <0,03 | high_blood_pressure, anaemia, diabetes, creatinine_phosphokinase, sex, platelets, smoking |

`ejection_fraction` và `serum_creatinine` đứng **#2 và #3** — đúng như y
văn khẳng định là 2 chỉ số quyết định nhất trong mô hình **loại `time`**
(Bảng 8 của bài báo gốc: Random Forest feature ranking không có `time`,
`serum_creatinine` #1, `ejection_fraction` #2 — khớp gần như tuyệt đối với
kết quả ở đây khi bỏ `time`, xem cột "Bỏ `time`" ở Mục 4.4). Ở SHAP đầy đủ
(model CÓ `time`), 2 biến này tụt xuống #2/#3 vì bị `time` vượt qua — đây
chính là kết luận Bảng 11 của bài báo gốc, không phải sai lệch do SMOTE
(đã kiểm chứng: correlation Pearson thô trên dữ liệu gốc — chưa SMOTE —
cũng cho `time` đứng #1 với hệ số −0,527, mạnh hơn hẳn serum_creatinine
+0,294 và ejection_fraction −0,269).

---

## 6. Hạn chế

1. **Cỡ mẫu nhỏ (N=299), test set chỉ 60 ca** — mọi con số accuracy/MCC
   dao động mạnh theo `random_state`; cần CV lặp lại nhiều lần + báo cáo
   khoảng tin cậy (Bootstrap 95% CI) mới đáng tin, một con số đơn lẻ không
   đủ.
2. **Biến `time` mang nguy cơ rò rỉ thời gian (temporal leakage)** — xem
   Mục 4.4. Một pipeline dự báo nguy cơ thực tế (tại thời điểm nhập viện)
   nên cân nhắc bỏ `time` hoặc dùng nó theo cách khác (biến mục tiêu trong
   bài toán survival analysis, không phải đặc trưng đầu vào của bài toán
   phân loại nhị phân).
3. **Dữ liệu 1 trung tâm, năm 2015, dân số Pakistan** — chưa có external
   validation trên dân số khác; không nên khái quát hóa trực tiếp. Đây
   cũng là hạn chế chính bài báo gốc tự nêu ("small size of the dataset",
   thiếu thông tin thể trạng, không có validation cohort khác khu vực) —
   nếu chỉ liệt kê lại đúng những điểm này thì báo cáo còn thiếu chiều
   sâu; nên tự nêu thêm góc nhìn riêng (vd điểm 2 ở trên, bài báo gốc
   không bàn tới).
4. **SMOTE không cải thiện chắc chắn** — trong ví dụ này, SMOTE + Extra
   Trees (đúng cấu hình lab yêu cầu) cho kết quả **kém hơn** cả Random
   Forest baseline không SMOTE (MCC 0,418 vs 0,600) trên tập Test thật.
   SMOTE giúp cân bằng lớp nhưng không đảm bảo luôn tăng hiệu năng trên
   tập test giữ nguyên phân bố gốc — nên thử nghiệm có đối chứng (có/không
   SMOTE) thay vì mặc định SMOTE luôn tốt hơn.
5. **Chưa có đánh giá lâm sàng thực địa (HITL)** — mô hình chỉ dừng ở mức
   nghiên cứu, không thay thế đánh giá bác sĩ.
