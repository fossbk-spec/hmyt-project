# Phân loại nguy cơ bệnh võng mạc tiểu đường bằng ảnh OCT

Nhóm 14. Midterm.

## Mục tiêu
Phân loại nhị phân bệnh võng mạc tiểu đường đe dọa thị lực
(vision-threatening DR) trên bộ dữ liệu Harvard-FairVision (thư mục DR),
và so sánh với bài báo gốc (FairVision) về AUC và các chỉ số công bằng
theo chủng tộc, giới, dân tộc. OLIVES chưa được dùng ở mốc này.

## Dữ liệu
- Harvard-FairVision, thư mục DR: 10.000 bệnh nhân, mỗi người một lần chụp
  (OCT 3D 128x200x200 và ảnh SLO 200x200). Split có sẵn
  6.000/1.000/3.000. Giấy phép CC BY-NC-ND 4.0.
- Repo này không chứa dữ liệu. Tải dữ liệu từ trang chính thức của dataset
  (https://ophai.hms.harvard.edu/datasets/harvard-fairvision30k).

## Mã nguồn
Thư mục `src/` là phiên bản v5 của pipeline (PyTorch).
- `cache_fv_zip.py`: tạo cache uint8 từ dataset.zip
- `train.py`: huấn luyện và đánh giá
- `tune_threshold.py`, `latency.py`, `report.py`: ngưỡng, độ trễ, so với bài
Phiên bản v6 (`--resume`) và v7 (FIS) chưa chạy thử trên GPU nên không nộp.

```
python cache_fv_zip.py --zip <dataset.zip> --out <cache_dir> --modes slo:224 3d:64x128
python train.py --task fairvision --root <root> --cache <cache_dir>/slo_224 \
  --mode slo --arch timm:vit_base_patch16_224.mae --pretrained --imagenet_norm \
  --epochs 20 --bs 64 --lr 1.25e-4 --layer_decay 0.55 --drop_path 0.1 \
  --warmup_epochs 2 --wd 0.01 --amp --seeds 0 1 2 3 4 --out <runs_dir>
```

## Tiền xử lý 
- SLO: min-max từng ảnh về [0,1], bilinear 200 sang 224, lặp 3 kênh,
  chuẩn hóa ImageNet. Tăng cường duy nhất: nhân độ sáng U(0,9; 1,1).
- OCT 3D: min-max từng volume, trilinear sang 64x128x128.

## Kết quả chính (test)
| Mô hình | AUC của nhóm | AUC của bài |
|---|---|---|
| ViT-B MAE, SLO (5 seed) | 83,16 ± 0,36 | 85,22 ± 0,21 |
| r3d_18 Kinetics-400, OCT 3D (1 seed) | 95,16 | 92,22 ± 0,53 |

Bất công cùng chiều với bài: nhóm Black thấp nhất, White cao nhất.

## Hạn chế
- Bài báo không nêu tiền xử lý, số epoch thực và nguồn trọng số, nên các
  con số của bài chỉ là mốc tham khảo, không phải mục tiêu phải đạt.
- 3D mới chạy 1 seed, 5 epoch; chưa có độ lệch chuẩn.
- Nhóm chưa xác minh split của nhóm trùng split của bài.
- DEOdds là xấp xỉ, không so tuyệt đối với bài.
- Chưa tách riêng nguyên nhân khoảng cách với bài (tiền huấn luyện,
  chuẩn hóa, số epoch).

## Thành viên
202414858, 202414861, 202414864