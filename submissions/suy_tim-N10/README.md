\# Heart Failure 30-Day Readmission Prediction



\## Project



Dự báo nguy cơ tái nhập viện trong vòng 30 ngày ở bệnh nhân suy tim từ dữ liệu bệnh án điện tử MIMIC-IV v3.1.



\## Dataset



Project sử dụng MIMIC-IV Clinical Database v3.1 từ PhysioNet.



Dữ liệu MIMIC-IV gốc và các dataset dẫn xuất theo từng bệnh nhân không được đưa lên repository công khai.



\## Current Pipeline



\### 1. 01\_build\_hf\_preprocessing.py



\- Lọc cohort bệnh nhân suy tim.

\- ICD-9: 428\*

\- ICD-10: I50\*

\- Lọc bệnh nhân từ 18 tuổi trở lên.

\- Loại trường hợp tử vong trong viện.

\- Chỉ giữ thời gian nằm viện từ 1 ngày trở lên.

\- Tạo nhãn tái nhập viện trong vòng 30 ngày.

\- Tạo các đặc trưng về tiền sử nhập viện, thủ thuật và xét nghiệm.

\- Tạo patient-level folds để hạn chế data leakage.



\### 2. 02\_add\_clinical\_features.py



\- Bổ sung các bệnh đồng mắc.

\- Bổ sung nhóm thuốc.

\- Bổ sung đặc trưng OMR.

\- Kiểm tra chất lượng dữ liệu.

\- Tạo bộ dữ liệu cuối trước mô hình.



\### 3. MIMIC\_IV\_v3\_1\_HF\_EDA.ipynb



EDA bao gồm:



\- Tổng quan cohort.

\- Phân bố lớp.

\- Phân bố tuổi và giới tính.

\- So sánh Readmission và Non-readmission.

\- Missing data.

\- Correlation analysis.

\- Data quality checks.



\## Current Dataset Summary



\- HF patients: 28,376

\- HF admissions: 70,066

\- Readmission within 30 days: 17,522

\- Readmission rate: 25.01%

\- Initial predictor columns: 46



Một số biến OMR có tỷ lệ missing lớn hơn 60% được loại khỏi mô hình đầu tiên.



\## Leakage Control



Dữ liệu được chia theo `subject\_id`.



Tất cả admissions của cùng một bệnh nhân phải nằm trong cùng một tập.



Các bước phụ thuộc dữ liệu như:



\- Imputation

\- StandardScaler

\- One-Hot Encoding



chỉ được fit trên tập Train.



Validation và Test chỉ sử dụng các phép biến đổi đã học từ Train.



\## Current Status



Đã hoàn thành:



\- MIMIC-IV access

\- Heart failure cohort construction

\- 30-day readmission labeling

\- Feature engineering

\- EDA

\- Pre-model preprocessing

\- Patient-level split



Bước tiếp theo:



\- Model training

\- Logistic Regression

\- XGBoost / LightGBM

\- Evaluation

\- SHAP explainability



\## Data Privacy



Không upload dữ liệu MIMIC-IV gốc hoặc dataset dẫn xuất theo từng bệnh nhân lên GitHub công khai.



File `hf\_model\_dataset\_final.csv.gz` chỉ được lưu cục bộ.

