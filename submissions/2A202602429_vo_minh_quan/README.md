# Lab Day 2 — DeepWeeds · Võ Minh Quân · 2A202602429

Nộp bài Lab Day 2 (backbone, công thức huấn luyện, suy luận) trên DeepWeeds, dùng fold 0.

- Kết quả chung kết trên test (3 seed): **macro-F1 0,9805 ± 0,0016**, **top-1 0,9845 ± 0,0009**.
- Cấu hình: ConvNeXt-T `in12k`, 20 epoch, test ở độ phân giải 256, temperature scaling.
- Xem chi tiết trong [`report.md`](report.md) và [`results.xlsx`](results.xlsx).

## Chạy lại

| | Link |
|---|---|
| Kaggle (bản đã chạy, có output) | https://www.kaggle.com/code/minhquanvo/notebookf7a4fbbfc0 |
| Colab (mở thẳng từ GitHub) | https://colab.research.google.com/github/vminhquan/K4-Track4-Day2-VoMinhQuan-2A202602429/blob/main/submissions/2A202602429_vo_minh_quan/code/lab_day2.ipynb |
| Notebook trong repo | [`code/lab_day2.ipynb`](code/lab_day2.ipynb) (sạch) · [`code/lab_day2_kaggle_executed.ipynb`](code/lab_day2_kaggle_executed.ipynb) (đã chạy, kèm output) |

### Trên Kaggle (cách đã dùng để tạo kết quả)

1. Vào **New Notebook → File → Import Notebook** và dán link raw sau:
   `https://raw.githubusercontent.com/vminhquan/K4-Track4-Day2-VoMinhQuan-2A202602429/main/submissions/2A202602429_vo_minh_quan/code/lab_day2.ipynb`
2. Ở **Settings**, chọn Accelerator **GPU T4 x2** và bật **Internet On**.
3. Chọn **Save Version → Save & Run All (Commit)**. Toàn bộ mất khoảng 4,6 giờ GPU T4 (đo thật).
4. Khi chạy xong, vào tab **Output** của phiên bản đó và tải `lab_day2_work/submission_outputs.zip`.

### Trên Colab

Chọn **Runtime → Change runtime type → T4 GPU**, rồi **Run all**. Kết quả được lưu ở `MyDrive/lab_day2_work`. Nếu bị ngắt, chạy lại **Run all**:
- lần train đã xong được bỏ qua;
- lần đang dở tiếp tục từ epoch cuối;
- test của chung kết không bao giờ chạy lại.

Đặt `SMOKE = True` ở ô đầu để chạy thử nhanh (4 ảnh/lớp, 1 epoch). Chế độ này chỉ để kiểm tra code; kết quả không dùng được.

### Chạy từng bước bằng dòng lệnh

Chạy từ thư mục bài nộp. Các bước phải chạy **theo đúng thứ tự** dưới đây, vì bước sau đọc `results/selection.json` do bước trước ghi:
```bash
python code/step0_checks.py                                   # Bước 0: split, EDA, loss ≈ ln 9, overfit 1 batch
python code/experiments.py --group B --preload                # Bước 1: 6 backbone (seed 0)
python code/experiments.py --select-backbone                  #   đo độ trễ, chọn backbone theo val
python code/experiments.py --group T --preload                # Bước 2: T00 × 3 seed + T01–T14
python code/experiments.py --group C --preload --select-recipe   # kết hợp + chốt công thức theo val
python code/step3_inference.py --preload                      # Bước 3: suy luận + độ trễ, chọn phương pháp theo val
python code/step4_final.py --preload                          # Bước 4: F01 × 3 seed, test 1 lần/seed, eval.py
python code/make_results.py                                   # Bước 5: results.xlsx + bảng
python -m unittest discover -s code -p "test_code.py"         # 18 test tự viết
```
Code tìm `eval.py` gốc bằng cách đi ngược lên các thư mục cha, hoặc đọc biến môi trường `LAB_REPO`. Dữ liệu đặt ở `<repo>/data/images` và `<repo>/data/labels`.

## Môi trường đã chạy

| | |
|---|---|
| Phần cứng | Kaggle Notebook, NVIDIA Tesla T4 (16 GB) |
| Phần mềm | Python 3.13.15 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · timm 1.0.29 · numpy 2.1.3 · fvcore (đếm GMAC) |
| Mixed precision | AMP fp16 khi train |
| Seed | 0 cho Bước 1–2; T00 và F01 chạy seed 0, 1, 2 |
| Dữ liệu | DeepWeeds `images.zip` (MD5 `b7b30f96d466fba86016aa5a26606e0f`), nhãn fold 0 từ github.com/AlexOlsen/DeepWeeds |

**Mức tái lập:** cùng môi trường và cùng seed cho cùng kết quả. B02 và T00 seed 0 có cấu hình giống hệt nhau và đều cho macro-F1 val 0,96704. Chạy trên GPU khác hoặc phiên bản thư viện khác có thể lệch khoảng 0,005, vì cuDNN không tất định hoàn toàn.

## Cấu trúc thư mục

```
├── README.md            # file này
├── report.md            # báo cáo kết luận
├── results.xlsx         # 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency
├── curves/              # 27 ảnh biểu đồ training, mỗi exp_id một ảnh (B01–B06, T00×3, T01–T14, C01, F01×3)
├── predictions/         # <exp_id>_seed<k>_{val,test}.csv đúng định dạng eval.py
│                        #   chung kết F01 / F01_uncal / F01rt và mốc T00: đủ test cho seed 0, 1, 2
├── results/             # selection.json (mọi lựa chọn + lý do), final.json, grade.txt,
│                        #   eval/ (đầu ra eval.py), backbones.csv, inference_val.csv, latency.csv, tables.md
├── figures/             # EDA, ảnh sau augmentation, overfit, đánh đổi độ trễ, ma trận nhầm lẫn, ảnh bị đoán sai
├── logs/                # config.json, history.csv, summary.json của từng lần chạy (truy ngược số liệu)
└── code/                # bộ khung starter/ đã hoàn thiện + các script chạy từng bước + notebook
```

**Kiểm tra lại số liệu chung kết bằng `eval.py` gốc** (chạy từ thư mục bài nộp):
```bash
python ../../eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
  --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \
  --val-csv ../../data/labels/val_subset0.csv --latency-p95-ms 6.5 \
  --test-csv ../../data/labels/test_subset0.csv --labels ../../data/labels/labels.csv
```

Checkpoint (`*.pt`) và dataset không được commit. Có thể tạo lại checkpoint bằng các lệnh ở trên.
