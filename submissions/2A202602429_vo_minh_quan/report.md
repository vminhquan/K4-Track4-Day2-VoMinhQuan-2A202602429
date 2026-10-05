# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Võ Minh Quân · 2A202602429**

Mọi con số dưới đây lấy từ log chạy thật trên Kaggle (Tesla T4). Mỗi số truy ngược được qua `exp_id` tới:
- `logs/<exp_id>/seed<k>/` (cấu hình, log theo epoch, tóm tắt);
- `curves/<exp_id>_*.png`;
- `results.xlsx`;
- `predictions/`.

Số trên test được tính bằng `eval.py` gốc, không sửa. Chênh lệch nhỏ hơn độ lệch chuẩn qua seed được ghi là "không phân biệt được".

---

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp cỏ dại trên DeepWeeds, dùng fold 0 chia sẵn. Dữ liệu mất cân bằng: lớp `Negatives` chiếm khoảng 52%.
- **Thí nghiệm đã chạy:** 6 backbone; 15 cấu hình huấn luyện thuộc 7 trục cộng 1 kết hợp; 18 cấu hình suy luận; chung kết 3 seed. Tổng cộng 30 lần train, khoảng 4,6 giờ GPU.
- **Cấu hình tốt nhất (F01):**
  - backbone ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`);
  - công thức nền, kéo dài lên **20 epoch** (T14);
  - suy luận ở **độ phân giải 256** (I04_256);
  - temperature scaling, T khớp trên val.
- **Kết quả trên test (toàn bộ 3.507 ảnh, mean ± std qua 3 seed):**
  - **top-1 = 0,9845 ± 0,0009**;
  - **macro-F1 = 0,9805 ± 0,0016**;
  - ECE = 0,0037 ± 0,0004;
  - recall Chinee apple 0,960 và Snake weed 0,953.
- **So với mốc** (T00 + I00: công thức nền, 1 view, cùng 3 seed; macro-F1 test 0,9698 ± 0,0023): Δ macro-F1 = **+0,0108**, lớn hơn std (0,0023).
- **Kết luận chính:**
  - **backbone** là yếu tố quyết định lớn nhất (macro-F1 val dao động 0,68–0,97);
  - trên một backbone mạnh, **suy luận ở độ phân giải cao hơn** đóng góp nhiều hơn mọi thay đổi công thức huấn luyện đã thử;
  - cấu hình chung kết chạy được thời gian thực: p95 = 8,5 ms ở batch 1 trên T4.

## 2. Dữ liệu và thiết lập

**Dữ liệu.** 17.509 ảnh RGB 256×256. MD5 của `images.zip` đã kiểm tra (`b7b30f96…`). Dùng fold 0 của tác giả (`train/val/test_subset0.csv`), không sửa và không chia lại.

Kết quả các kiểm tra bắt buộc (`code/step0_checks.py`, `results/step0.json`):
- Số ảnh: train 10.501 · val 3.501 · test 3.507, tỉ lệ 59,97 / 20,00 / 20,03%.
- Giao của từng cặp tập đều rỗng; hợp ba tập đủ 17.509 ảnh; mọi file trong CSV đều tồn tại.

| Lớp | train | val | test | Tổng (khớp Table 1 bài báo) |
|---|---|---|---|---|
| Chinee apple | 675 | 225 | 226 | 1.126 |
| Lantana | 637 | 213 | 213 | 1.063 |
| Parkinsonia | 618 | 206 | 207 | 1.031 |
| Parthenium | 613 | 204 | 205 | 1.022 |
| Prickly acacia | 637 | 212 | 213 | 1.062 |
| Rubber vine | 605 | 202 | 202 | 1.009 |
| Siam weed | 644 | 215 | 215 | 1.074 |
| Snake weed | 609 | 203 | 204 | 1.016 |
| Negatives | 5.463 | 1.821 | 1.822 | 9.106 |

Số ảnh của từng lớp trong `labels.csv` khớp đúng Table 1 của bài báo.

**Phát hiện khi đối chiếu nhãn:** ở bảng trên, tổng của Chinee apple là 1.126 và của Lantana là 1.063, lệch 1 ảnh so với `labels.csv` (1.125 và 1.064). Nguyên nhân là đúng **một ảnh**, `20170714-110407-3.jpg`, có nhãn khác nhau giữa hai file: `train_subset0.csv` ghi Label 0 (Chinee apple), còn `labels.csv` ghi Label 1 (Lantana).
- Theo quy tắc S1, mình giữ nguyên file CSV của fold, không sửa.
- Ảnh này nằm trong **train**, nên không ảnh hưởng tới val, test hay việc chấm điểm. Tác động lên quá trình train là 1/10.501 ảnh, không đáng kể.

Tỉ lệ lớp lớn nhất so với lớp nhỏ nhất là 9,02. Một model luôn đoán `Negatives` đã đạt khoảng 52% top-1, nên **chỉ số chính là macro-F1**.

Biểu đồ và ảnh minh hoạ:
- `figures/eda_class_distribution.png`: phân bố lớp;
- `figures/eda_samples.png`: 4 ảnh mẫu mỗi lớp. Các loài cỏ chụp từ trên xuống, nền đất và cỏ khô rất giống lớp `Negatives`; Chinee apple và Snake weed đều có lá nhỏ, mọc dày;
- thống kê điểm ảnh trên 500 ảnh train: mean RGB (0,341; 0,352; 0,343), std khoảng 0,225. Vì trọng số tiền huấn luyện được chuẩn hoá theo mean/std ImageNet nên mình giữ chuẩn hoá ImageNet.

**Kiểm tra pipeline (GUIDE 1.3).**
- Loss ban đầu của head mới là 2,179, xấp xỉ ln 9 = 2,197.
- Overfit 16 ảnh: loss giảm từ 2,232 xuống 0,018 sau 150 bước, accuracy 100% ở chế độ eval (`figures/overfit_one_batch.png`).
- Ảnh sau augmentation được giải chuẩn hoá và kiểm tra cùng nhãn (`figures/aug_basic.png`, `aug_trivial.png`, `aug_cutmix.png`).
- 18 unit test trong `code/test_code.py` kiểm tra các phần dễ sai:
  - focal γ=0 cho đúng CE (sai số < 1e-6);
  - label smoothing khớp với PyTorch;
  - CutMix tính λ theo diện tích thật của hộp sau khi cắt ra ngoài biên;
  - Mixup trộn cả nhãn;
  - weight decay = 0 cho norm/bias;
  - đóng băng backbone thì BN giữ chế độ eval;
  - gộp BN có sai số < 1e-4;
  - temperature scaling khôi phục đúng T;
  - LR warmup + cosine.

**Công thức nền T00** (GUIDE 1.4):

| Thành phần | Giá trị |
|---|---|
| Khởi tạo | Trọng số ImageNet, head 9 lớp mới, tinh chỉnh toàn bộ |
| Augmentation train | `RandomResizedCrop(224)` + lật ngang |
| Tiền xử lý val/test | ảnh 256 → `CenterCrop(224)` |
| Optimizer | AdamW, 3 nhóm tham số: backbone LR 1e-4 có WD 0,05 · norm/bias của backbone LR 1e-4, WD 0 · head LR 1e-3 |
| Lịch LR | warmup tuyến tính 1 epoch, sau đó cosine về 0, cập nhật theo từng bước |
| Loss | cross-entropy |
| Batch, số epoch | batch 64, **10 epoch** cho mọi lần chạy (trừ khi đang thử chính trục số epoch) |
| Mixed precision | AMP fp16 |
| Chọn checkpoint | epoch có macro-F1 val cao nhất; hoà thì lấy epoch sớm hơn |

**Phần cứng và phần mềm.**
- Kaggle Notebook, Tesla T4, 4 CPU.
- Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, numpy 2.1.3.
- GMAC đếm bằng fvcore.
- Ảnh được giải mã trước vào RAM để tránh nghẽn đọc đĩa.

**Seed.**
- Seed 0 cho Bước 1 và Bước 2. T00 và F01 chạy seed 0, 1 và 2.
- Seed chỉ thay đổi khởi tạo head, thứ tự batch và augmentation; không thay đổi cách chia dữ liệu.
- B02 và T00 seed 0 có cấu hình giống hệt nhau và cho cùng macro-F1 val 0,96704. Điều này chứng tỏ pipeline tái lập được trong cùng một môi trường.

**Quy tắc val/test.**
- Mọi lựa chọn (backbone, công thức, phương pháp suy luận, T) dựa **chỉ trên val**, theo quy tắc viết sẵn trong code (`experiments.py`, `step3_inference.py`).
- Test chỉ được dùng ở Bước 4, mỗi seed chạy một lần:
  - các pipeline F01, F01_uncal và F01rt được khai báo trước khi mở test;
  - dự đoán test của T00 được ghi lúc train nhưng không được mở cho tới Bước 4.

## 3. So sánh backbone (Bước 1)

Cả 6 backbone dùng công thức nền T00, seed 0, 10 epoch. Độ trễ đo ở batch 1, FP32, trên T4: warmup 20 lần, `cuda.synchronize`, 100 lần đo, không tính tiền xử lý. Nguồn: sheet `Backbones`, `figures/backbones_f1_vs_latency.png`.

| exp_id | Backbone (tag timm) | Params (M) | GMAC | macro-F1 val | top-1 val | s/epoch | b1 p50 / p95 (ms) | b32 (ảnh/s) |
|---|---|---|---|---|---|---|---|---|
| B01 | `resnet50.a1_in1k` | 23,5 | 4,11 | 0,7864 | 0,8455 | 43 | 6,1 / 7,6 | 770 |
| **B02** | **`convnext_tiny.in12k_ft_in1k`** | 27,8 | 4,47 | **0,9670** | **0,9751** | 52 | 5,8 / 9,4 | 646 |
| B03 | `deit_small_patch16_224.fb_in1k` | 21,7 | 4,25 | 0,9489 | 0,9640 | 34 | 5,2 / 5,6 | 1.046 |
| B04 | `swin_tiny_patch4_window7_224.ms_in1k` | 27,5 | 4,51 | 0,9536 | 0,9660 | 64 | 10,1 / 10,9 | 441 |
| B05 | `efficientnet_b0.ra_in1k` (nhẹ) | 4,0 | 0,40 | 0,7419 | 0,8118 | 33 | 8,9 / 9,5 | 1.416 |
| B06 | `mobilenetv3_large_100.ra_in1k` (nhẹ) | 4,2 | 0,22 | 0,6776 | 0,7686 | 23 | 6,3 / 6,8 | 2.437 |

**Chọn backbone ConvNeXt-T (B02).**
- Quy tắc định trước: lấy macro-F1 val cao nhất, trừ khi có backbone kém tối đa 0,005 mà nhanh hơn ít nhất 1,5 lần.
- B02 hơn backbone đứng thứ hai (Swin-T) 0,0134. Mức này lớn hơn nhiều so với nhiễu khoảng 0,0024 đo ở T00, nên điều kiện đổi sang backbone nhanh hơn không xảy ra.
- B02 cũng có F1 cao nhất ở hai lớp khó trên val: Chinee apple 0,938, Snake weed 0,940.

**Nhận xét.**
- **Hai nhóm tách biệt rõ.**
  - ConvNeXt-T, Swin-T và DeiT-S đạt 0,95–0,97.
  - ResNet-50, EfficientNet-B0 và MobileNetV3 chỉ đạt 0,68–0,79. Train loss của ba model này ở epoch 10 vẫn còn 0,25–0,41, so với 0,04–0,07 của nhóm đầu, tức là chúng **chưa khớp đủ (underfit)** chứ không phải quá khớp (xem `curves/B01_*.png`, `B05_*.png`, `B06_*.png`).
  - [Inference] Mình cho rằng nguyên nhân chủ yếu là **công thức nền không hợp với bộ trọng số**, không phải do kiến trúc. Các tag `a1` và `ra` của timm được tiền huấn luyện với LR lớn và augmentation mạnh; LR 1e-4 trong 10 epoch có lẽ quá nhỏ để chỉnh lại các tầng BatchNorm. Mình **chưa kiểm chứng** giả thuyết này; cách kiểm chứng là chạy lại B01 với LR lớn hơn hoặc với tag `tv2_in1k`.
  - Điều này khớp với slide trang 37 và 45: so sánh kiến trúc mà không xét công thức huấn luyện thì dễ kết luận sai.
  - ConvNeXt-T còn được hưởng lợi từ tiền huấn luyện trên ImageNet-22k (`in12k`). Vì vậy kết quả "ConvNeXt-T tốt nhất" ở đây là kết luận về **bộ trọng số kèm công thức**, không chỉ về kiến trúc.
- **Thứ hạng khác với ImageNet.** Trên ImageNet, ResNet-50 và EfficientNet-B0 không kém DeiT-S nhiều như vậy. Trên DeepWeeds, khoảng 10k ảnh và 10 epoch, khả năng thích nghi nhanh với công thức nền mới là yếu tố quyết định.
- **FLOPs không dự đoán được độ trễ** (slide trang 43).
  - EfficientNet-B0 chỉ có 0,40 GMAC (ít hơn ResNet-50 khoảng 10 lần) nhưng ở batch 1 lại **chậm hơn** (p50 8,9 ms so với 6,1 ms), vì có nhiều tầng depthwise nhỏ, bị giới hạn bởi chi phí khởi chạy kernel.
  - Ở batch 32, thứ tự đảo lại: EfficientNet-B0 đạt 1.416 ảnh/s, ResNet-50 đạt 770 ảnh/s.
  - Swin-T có GMAC tương đương ConvNeXt-T nhưng chậm gần gấp đôi ở batch 1.
- **Hội tụ.** DeiT-S train nhanh nhất mỗi epoch (34 s). ConvNeXt-T đạt macro-F1 0,82 ngay sau epoch 1 và vẫn tăng tới epoch 10. Val loss của B02 gần như đi ngang từ epoch 7 (0,096–0,107), nhưng không tăng trở lại, nên chưa có dấu hiệu quá khớp rõ (`curves/B02_convnext_tiny.png`).

## 4. Công thức huấn luyện (Bước 2)

**Thiết kế.**
- Tất cả chạy trên ConvNeXt-T. Mỗi lần chạy **chỉ khác T00 đúng một yếu tố**, seed 0, 10 epoch (riêng T14 chạy 20 epoch).
- **Nhiễu** được đo bằng T00 chạy 3 seed: macro-F1 val = 0,9668 ± **0,0024** (0,9670; 0,9642; 0,9690).
- Một yếu tố được coi là "có ích" khi Δ > max(std, 0,003) = 0,003.
- Thí nghiệm ablation chỉ chạy 1 seed. Vì vậy ngay cả khi Δ khoảng 2 std, kết luận vẫn là "có xu hướng", không phải khẳng định chắc chắn.

Nguồn: sheet `Training`, `figures/training_ablation.png`.

| exp_id | Trục | Khác T00 ở điểm nào | macro-F1 val | Δ vs T00 s0 | Δ / std | F1 Chinee | F1 Snake | Nhận xét |
|---|---|---|---|---|---|---|---|---|
| T00 | – | công thức nền (seed 0 / 1 / 2) | 0,9670 / 0,9642 / 0,9690 | – | – | 0,938 | 0,940 | mốc |
| T01 | A | train từ đầu, LR backbone 1e-3 | 0,4852 | −0,4818 | −197 | 0,362 | 0,325 | kém hẳn |
| T02 | A | đóng băng backbone, chỉ train head | 0,8437 | −0,1233 | −50 | 0,820 | 0,783 | kém hẳn |
| T03 | B | + TrivialAugmentWide | 0,9718 | +0,0047 | +1,9 | 0,950 | 0,932 | có xu hướng tốt |
| T04 | B | + lật dọc và xoay 90° | 0,9725 | +0,0055 | +2,3 | 0,953 | 0,941 | có xu hướng tốt |
| T05 | B | CutMix (α=1) | 0,9692 | +0,0022 | +0,9 | 0,946 | 0,939 | không phân biệt được |
| T06 | B | Mixup (α=0,2) | 0,9675 | +0,0005 | +0,2 | 0,930 | 0,921 | không phân biệt được |
| T07 | C | label smoothing ε=0,1 | 0,9677 | +0,0007 | +0,3 | 0,950 | 0,924 | không phân biệt được; ECE val tăng lên 0,088 |
| T08 | C | focal loss γ=2 | 0,9647 | −0,0023 | −0,9 | 0,930 | 0,922 | không phân biệt được |
| T09 | C | CE có trọng số lớp 1/n_c | 0,9632 | −0,0039 | −1,6 | 0,948 | 0,925 | có xu hướng kém |
| T10 | D | sampler cân bằng lớp | 0,9726 | +0,0055 | +2,3 | 0,960 | 0,934 | có xu hướng tốt |
| T11 | E | LR head = LR backbone (1e-4) | 0,9685 | +0,0015 | +0,6 | 0,935 | 0,936 | không phân biệt được |
| T12 | E | LR ×3 (3e-4 / 3e-3) | 0,9617 | −0,0053 | −2,2 | 0,913 | 0,920 | có xu hướng kém |
| T13 | F | EMA trọng số d=0,999 | 0,9693 | +0,0022 | +0,9 | 0,947 | 0,945 | không phân biệt được |
| T14 | G | 20 epoch thay vì 10 | **0,9730** | **+0,0059** | +2,4 | 0,943 | 0,936 | có xu hướng tốt, **được chọn** |
| C01 | kết hợp | T10 + T04 | 0,9700 | +0,0029 | +1,2 | 0,951 | 0,937 | **không cộng dồn** |

**Phân tích.**
- **Khởi tạo (trục A) có tác động lớn nhất trong Bước 2.**
  - Train từ đầu chỉ đạt 0,485 sau 10 epoch: khoảng 10k ảnh không đủ cho ConvNeXt-T học từ đầu (slide trang 53). Train loss vẫn ở 0,98.
  - Đóng băng backbone đạt 0,844: đặc trưng ImageNet có ích nhưng chưa đủ cho ảnh cỏ dại chụp từ trên xuống. Tinh chỉnh toàn bộ mạng hơn 0,12 macro-F1.
- **Augmentation (trục B).**
  - Lật dọc kèm xoay 90° (T04) có xu hướng giúp. [Inference] Lý do có thể là ảnh chụp từ trên xuống không có hướng "lên" cố định, nên các phép biến đổi này hợp lệ về mặt ngữ nghĩa.
  - TrivialAugment (T03) cũng có xu hướng tốt.
  - CutMix và Mixup không phân biệt được với mốc. [Inference] Cắt dán có thể làm mất phần lá nhỏ mang nhãn (GUIDE 9, câu 4). Train loss của CutMix là 0,49 vì nhãn bị trộn; số này không so trực tiếp được với các lần chạy khác.
- **Loss và sampler (trục C, D).**
  - Không loss nào cải thiện rõ ràng.
  - CE có trọng số lớp (T09) và focal loss (T08) còn có xu hướng kém hơn.
  - Sampler cân bằng (T10) cho F1 Chinee apple cao nhất trong Bước 2 (0,960) và balanced accuracy val 0,977, nhưng F1 `Negatives` gần như không đổi (0,985).
  - Label smoothing giữ nguyên F1 nhưng làm ECE val tăng mạnh (0,011 → 0,088), vì model trở nên thiếu tự tin một cách có hệ thống.
- **LR (trục E).** Tăng LR gấp 3 có xu hướng hại (−0,0053). Với ConvNeXt `in12k`, LR 1e-4 đã đủ. Ngược lại, giả thuyết LR quá nhỏ ở mục 3 áp dụng cho các bộ trọng số `a1`/`ra`. Hai quan sát này không mâu thuẫn, vì là những bộ trọng số khác nhau.
- **Thời lượng (trục G).** 20 epoch có xu hướng tốt nhất (+0,0059, khoảng 2,4 std) nhưng tốn gấp đôi thời gian. Khi kiểm chứng ở chung kết với 3 seed (F01, 1 view): 0,9720 ± 0,0022 so với T00 0,9668 ± 0,0024. Δ = +0,0053, khoảng 2,2 lần std, nên xu hướng này lặp lại được trên val.
- **Kết hợp không cộng dồn.**
  - Theo quy tắc, C01 gộp hai yếu tố vượt ngưỡng: sampler cân bằng (T10) và lật dọc + xoay (T04).
  - Kết quả C01 = 0,9700, **thấp hơn** từng yếu tố riêng lẻ (0,9725–0,9726).
  - [Inference] Có thể cả hai đều giúp theo cùng một kiểu (tăng đa dạng cho các lớp hiếm), nên gộp lại không thêm gì. Cũng có thể các Δ khoảng +0,0055 ban đầu chỉ là nhiễu của 1 seed. Với 1 seed, không phân biệt được hai khả năng này.
- **Cách chọn.** Mình dùng cách chọn **tham lam theo một yếu tố**: lấy cấu hình có F1 val cao nhất trong T00–T14 và C01, tức là T14. T14 không được kết hợp với T04 hoặc T10, vì quy tắc kết hợp loại trục G để giữ ngân sách. Đây là một hạn chế (xem mục 8).

## 5. Suy luận (Bước 3)

**Thiết lập.**
- Model: T14 seed 0. Đánh giá trên val. Không train lại.
- Độ trễ đo ở batch 1 trên T4, FP32 (trừ I08): warmup 20 lần, `cuda.synchronize` trước và sau mỗi lần đo, 100 lần đo, không tính tiền xử lý.
- Với TTA, độ trễ được đo thật bằng K lượt forward liên tiếp, không nhân suy ra.
- ECE của I07 đo bằng 2-fold chéo trên val (khớp T trên một nửa, đo trên nửa kia) để không lạc quan.

Nguồn: sheet `Inference`, `Latency`, `figures/inference_tradeoff.png`.

| exp_id | Phương pháp | K | macro-F1 val | Δ vs I00 | ECE val | p50 / p95 / p99 (ms) | Chi phí so với I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view, center crop 224 (mốc) | 1 | 0,9732 | – | 0,0115 | 5,9 / 9,4 / 9,4 | 1,0× |
| I01 | TTA lật ngang, gộp xác suất | 2 | 0,9729 | −0,0004 | 0,0110 | 11,5 / 12,2 / 13,8 | 1,9× |
| I02a | TTA 5 crop | 5 | 0,9716 | −0,0016 | 0,0084 | 28,7 / 31,0 / 31,7 | 4,8× |
| I02b | TTA 10 crop (5 crop + lật) | 10 | 0,9739 | +0,0006 | 0,0079 | 56,9 / 59,5 / 66,8 | 9,6× |
| I03a / I03b | như I01 / I02b nhưng gộp **logit** | 2 / 10 | 0,9726 / 0,9736 | −0,0006 / +0,0004 | 0,0103 / 0,0114 | như I01 / I02b | – |
| I04 | dò độ phân giải test 192 / **256** / 288 / 320 | 1 | 0,9684 / **0,9788** / 0,9766 / 0,9741 | −0,0049 / **+0,0055** / +0,0034 / +0,0008 | 0,0150 / 0,0076 / 0,0104 / 0,0086 | (256) 6,6 / 6,7 / 6,9 | 1,1× |
| I04_256full | ảnh 256 nguyên, không crop | 1 | 0,9725 | −0,0007 | 0,0112 | 6,3 / 10,0 / 10,0 | 1,1× |
| I05 | ensemble ConvNeXt-T + Swin-T + DeiT-S | 3 | 0,9712 | −0,0020 | 0,0124 | 21,5 / 22,8 / 23,3 | 3,6× |
| I06 | greedy soup (T14 + T10 + T03) | 1 | 0,9788 | +0,0055 | 0,0071 | 5,8 / 6,2 / 6,4 | 1,0× |
| I07 | temperature scaling (T = 1,559) | 1 | 0,9732 | 0 | **0,0033** | như I00 | 1,0× |
| I08b | FP16 (`model.half()`) | 1 | 0,9732 | 0 | 0,0114 | 5,9 / 6,5 / 7,9 | 1,0× |
| I08c | AMP autocast | 1 | 0,9730 | −0,0003 | 0,0115 | 7,9 / 8,5 / 9,0 | 1,3× |

**Nhận xét.**
- **TTA gần như không có lợi** trên model này: |Δ| ≤ 0,0016, nằm trong nhiễu, trong khi chi phí tăng 2–10 lần. Gộp xác suất hay gộp logit cũng không phân biệt được.
- **Dò độ phân giải là cải thiện lớn nhất và gần như không tốn thêm** (khớp với FixRes, slide trang 68).
  - Test ở 256 tốt hơn 224 (+0,0055 với model seed 0). Khi kiểm chứng ở chung kết trên **cả 3 seed** F01: macro-F1 val tăng từ 0,9720 ± 0,0022 (1 view, 224) lên 0,9774 ± 0,0003 (256 + TS), Δ = +0,0054.
  - [Inference] Giải thích có thể là `RandomResizedCrop` khi train phóng to vật thể lên. Test ở 224 với center crop thì vật thể nhỏ hơn lúc train; tăng độ phân giải test giúp khớp lại kích thước đó.
  - Tăng tiếp lên 288 hoặc 320 thì giảm dần, còn 192 thì kém hơn.
- **Ensemble khác backbone làm giảm F1 (−0,0020)**, vì hai thành viên Swin-T và DeiT-S yếu hơn ConvNeXt-T khoảng 0,02.
- **Greedy soup bằng I04_256 mà không tốn thêm gì khi suy luận.** Soup thử trung bình trọng số các checkpoint Bước 2 và chỉ giữ checkpoint nào làm F1 val không giảm; kết quả giữ lại T14 + T10 + T03. Quy tắc chọn cho chung kết chỉ xét các phương pháp dùng **một checkpoint** (vì mỗi seed chung kết chỉ có một model), nên soup không được dùng. Đây là một hướng tốt để làm tiếp.
- **Hiệu chuẩn.**
  - Model hơi tự tin quá mức (T = 1,56 > 1).
  - Temperature scaling giảm ECE val từ 0,0115 xuống 0,0033 mà không đổi dự đoán.
  - Trên test (chung kết): ECE giảm từ 0,0094 ± 0,0008 xuống **0,0037 ± 0,0004**; NLL giảm từ 0,0563 xuống 0,0492.
- **Gộp BN** không áp dụng được với ConvNeXt, vì ConvNeXt dùng LayerNorm (0 cặp conv–BN). Phép gộp đã được kiểm chứng riêng trên ResNet và EfficientNet trong unit test.
- **FP16 và AMP.**
  - FP16 giữ nguyên F1. Bảng `Latency` cho thấy ở batch 1, AMP **chậm hơn** FP32 (p50 8,7 so với 6,4 ms), nhưng ở batch 32 nhanh hơn 2,8 lần (609 so với 220 ảnh/s). Điều này khớp với slide trang 73: lợi ích của AMP chỉ thể hiện khi batch đủ lớn.
  - **Lưu ý về đo lường:** bảng `Latency` đo FP32 batch 1 ra p95 6,9 ms, trong khi dòng I00 ghi p95 9,4 ms dù p50 hai lần đo gần nhau (6,4 và 5,9 ms). Như vậy p95 ở mức vài ms dao động khoảng ±3 ms giữa các lần đo. Mình chỉ kết luận ở mức "dưới 10 ms", không xếp hạng các phương pháp chênh nhau 1–2 ms.
- **Ngoại tuyến hay thời gian thực.** Dữ liệu ủng hộ kết luận của slide:
  - TTA và ensemble chỉ hợp xử lý ngoại tuyến, và ở đây còn không giúp.
  - Trên robot, nên dùng các phương pháp không tốn thêm: độ phân giải đã dò, temperature scaling, FP16, model soup.
  - Trên **CPU** của máy Kaggle (số luồng mặc định của PyTorch), ConvNeXt-T 224 có p50 75,6 ms, p95 79,9 ms nhưng p99 101 ms. Nếu robot không có GPU thì sát ngân sách 100 ms.

## 6. Cấu hình tốt nhất và kết quả chung kết (Bước 4)

**F01**, đủ để tái lập:
- ConvNeXt-T `convnext_tiny.in12k_ft_in1k`;
- công thức nền T00 với **20 epoch**;
- suy luận: `Resize(293)` + `CenterCrop(256)`, 1 view;
- temperature scaling với T khớp trên val của từng seed: 1,590 / 1,616 / 1,609;
- seed 0, 1, 2; mỗi seed chạy test **đúng một lần**.

Các pipeline khai báo trước khi mở test:
- **F01_uncal**: như F01 nhưng không temperature scaling;
- **F01rt**: cùng model, 1 view ở 224, có temperature scaling;
- **T00**: mốc, công thức nền 10 epoch, 1 view ở 224.

Kết quả trên test (3.507 ảnh, mean ± std qua 3 seed, `eval.py score`). Nguồn: sheet `Final`, `results/eval/`.

| Cấu hình | macro-F1 test | top-1 test | balanced acc | ECE | NLL | recall Chinee | recall Snake |
|---|---|---|---|---|---|---|---|
| **F01** (T14 + 256 + TS) | **0,9805 ± 0,0016** | **0,9845 ± 0,0009** | 0,9784 ± 0,0024 | **0,0037 ± 0,0004** | 0,0492 | **0,960** | **0,953** |
| F01_uncal (không TS) | 0,9805 ± 0,0016 | 0,9845 ± 0,0009 | 0,9784 | 0,0094 ± 0,0008 | 0,0563 | 0,960 | 0,953 |
| F01rt (T14 + 224 + TS) | 0,9720 ± 0,0036 | 0,9781 ± 0,0026 | 0,9761 | 0,0052 | 0,0701 | 0,953 | 0,951 |
| T00 + I00 (mốc) | 0,9698 ± 0,0023 | 0,9762 ± 0,0019 | 0,9704 | 0,0101 | 0,0788 | 0,928 | 0,948 |
| *Bài báo, ResNet-50 (trích dẫn)* | – | *95,7% (weighted avg)* | – | – | – | *88,5%* | *88,8%* |

**Tự chấm phần I** (`eval.py grade`, `results/grade.txt`):

| Mục | Kết quả |
|---|---|
| I1 | top-1 98,45% → 7/7 |
| I2 | Δ = +0,0108 > s = 0,0023 và ≥ 0,01 → 5/5 |
| I3 | recall Chinee 96,0%, Snake 95,3%, cả hai trên mốc bài báo → 4/4 |
| I4a | ECE 0,0094 → 0,0037 → 1/1 |
| I4b | chênh macro-F1 val/test = 0,0032 ≤ 0,02 → 1/1 |
| I5 | p95 thời gian thực = 6,5 ms ≤ 100 ms, đo đúng cách → 2/2 |
| **Tổng đề xuất** | **20/20** (giảng viên xác nhận) |

Cách so với bài báo chỉ mang tính tham khảo:
- bài báo dùng "weighted average accuracy" trên 5 fold, 100 epoch, Keras;
- số theo lớp của bài báo được coi như recall.

**So với mốc.**
- Cải thiện +0,0108 macro-F1 và +0,0083 top-1, cả hai lớn hơn std.
- Có thể tách đóng góp **trên test** nhờ F01rt (cùng model với F01, chỉ khác độ phân giải test). Phần này chỉ để phân tích sau khi đã chốt cấu hình, không dùng để chọn:
  - **công thức** (20 epoch): F01rt − T00 = +0,0022, **nhỏ hơn** std (0,0036) → không phân biệt được trên test, dù trên val Δ khoảng 2,2 std;
  - **độ phân giải test 256**: F01 − F01rt = +0,0085, lớn hơn std.
- Temperature scaling không đổi F1, chỉ giảm ECE từ 0,0094 xuống 0,0037.

**Phân tích lỗi** (`figures/confusion_F01_seed0.png`, `figures/errors_chinee_snake.png`, `results/eval/F01_confusion_sum.csv`; tổng 3 seed).
- Tổng số ảnh sai (seed 0): F01 57 ảnh, T00 91 ảnh.
- **Lỗi chủ yếu là nhầm với `Negatives`, không phải giữa hai loài với nhau.**
  - Chinee apple: 20 lần bị đoán thành Negatives, 7 lần thành Snake weed.
  - Snake weed: 19 lần thành Negatives, 8 lần thành Chinee apple.
  - Theo chiều ngược lại, Negatives bị đoán thành Prickly acacia 20 lần.
  - Kết quả này khác với bài báo, nơi lỗi nổi bật là cặp Chinee ↔ Snake (3,4% và 4,1%).
  - Lỗi Chinee → Negatives giảm từ 30 (T00) xuống 20 (F01). Đây là lý do recall Chinee tăng từ 0,928 lên 0,960.
- **Giả thuyết từ việc xem ảnh sai:**
  1. Cây mục tiêu chỉ chiếm một phần nhỏ khung hình, lẫn trong cỏ và lá khô, nên ảnh trông giống `Negatives`. Nhãn của bộ dữ liệu là theo cả ảnh chứ không theo vùng.
  2. Ánh sáng gắt và bóng đổ (có ảnh có bóng người chụp), cùng ảnh ám màu hồng hoặc tím.
  3. Chinee apple và Snake weed đều có lá nhỏ hình bầu dục mọc dày. Các ảnh bị nhầm giữa hai lớp có độ tin cậy chỉ khoảng 0,50–0,70, tức model đang lưỡng lự đúng chỗ.
- Độ tin cậy trung bình của các ảnh sai là 0,70 (F01) so với 0,78 (T00), phù hợp với việc temperature scaling làm model bớt tự tin quá mức.

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất:** F01, gồm ConvNeXt-T `in12k`, 20 epoch, test ở 256 và temperature scaling. Kết quả: macro-F1 test 0,9805 ± 0,0016, top-1 0,9845 ± 0,0009. Cấu hình này tốt hơn mốc +0,0108 macro-F1, vượt nhiễu.
- **Yếu tố đóng góp nhiều nhất:**
  1. **Backbone kèm bộ trọng số tiền huấn luyện:** khoảng 0,29 macro-F1 val giữa backbone tốt nhất và kém nhất, và 0,013 giữa hạng 1 và hạng 2.
  2. **Suy luận** (độ phân giải test): +0,0054 trên val (3 seed) và +0,0085 trên test.
  3. **Công thức huấn luyện**, trong số các thay đổi giữ nguyên khởi tạo: các Δ chỉ khoảng ±0,006, không yếu tố nào vượt quá 2,5 std.

  Ngoài ra, chọn sai cách khởi tạo (train từ đầu, đóng băng backbone) làm mất 0,12–0,48. Vì vậy "tinh chỉnh toàn bộ từ trọng số tiền huấn luyện" là điều kiện bắt buộc.
- **Triển khai trên robot (ngân sách 30–100 ms/khung):**
  - Có GPU cỡ T4: dùng chính F01 (1 view ở 256, có temperature scaling; p95 8,5 ms, p99 11,3 ms ở batch 1, FP32). Có thể bật FP16 vì F1 không đổi.
  - Thiết bị yếu hơn: đo lại trên chính phần cứng đó. Có thể cân nhắc DeiT-S (B03, nhanh nhất ở batch 1, nhưng kém ConvNeXt-T 0,018 F1 val) hoặc xuất ONNX/TensorRT.
  - **Không** dùng TTA hoặc ensemble.
  - Nên giữ temperature scaling để ngưỡng tin cậy (ví dụ để quyết định có phun thuốc hay không) có ý nghĩa xác suất.

## 8. Hạn chế và việc tiếp theo

- **Seed và fold.**
  - Ablation ở Bước 1 và 2 chỉ chạy 1 seed. Mức nhiễu (std 0,0024) đo trên T00 và được giả định là như nhau cho mọi cấu hình.
  - Chỉ dùng **fold 0**, nên không có std qua fold.
- **Chia ngẫu nhiên, không theo địa điểm.** Ảnh cùng một buổi chụp hoặc cùng địa điểm có thể nằm ở cả train và test, nên con số test **có thể lạc quan** so với khi gặp địa điểm, mùa hoặc ánh sáng mới. Chênh val/test rất nhỏ (0,003) cũng phù hợp với việc val và test cùng phân phối. Rủi ro lệch phân phối khi triển khai thực tế chưa được đo.
- **Giảm bớt do ngân sách GPU.**
  - 10 epoch thay vì khoảng 100 epoch như bài báo.
  - Chỉ chạy ablation trên 1 backbone.
  - Không thử kết hợp T14 với các yếu tố khác (T04, T10, T03), cũng không dùng soup ở chung kết.
  - Giả thuyết "ResNet-50, EfficientNet và MobileNetV3 cần LR cao hơn" chưa được kiểm chứng.
- **Thí nghiệm thất bại hoặc không giúp:**
  - train từ đầu và đóng băng backbone;
  - CE có trọng số lớp, focal loss, LR ×3;
  - CutMix, Mixup, label smoothing không giúp (label smoothing còn làm hiệu chuẩn kém đi);
  - TTA và ensemble không giúp;
  - kết hợp C01 không cộng dồn.
- **Tái lập.**
  - GPU (cuDNN) không tất định hoàn toàn: B01 chạy trên Colab và trên Kaggle lệch nhau khoảng 0,005 F1 ở cùng epoch. Mọi số trong báo cáo đến từ **một** lần chạy trên Kaggle.
  - Code có thể tiếp tục từ epoch đã lưu khi bị ngắt; trong lần chạy Kaggle này không có lần train nào bị ngắt.
  - Chọn cấu hình theo cách tham lam theo trục, nên thứ tự các trục có thể ảnh hưởng tới kết quả.
- **Hướng tiếp theo:**
  1. Chạy đủ 5 fold.
  2. Kiểm chứng giả thuyết LR cho ResNet-50 (`a1`) và thử tag `tv2`.
  3. Kết hợp 20 epoch với lật dọc và sampler cân bằng, dùng 3 seed.
  4. Greedy soup cho từng seed chung kết.
  5. Đánh giá trên ảnh bị làm tối, làm nhiễu hoặc làm mờ để đo độ bền khi lệch phân phối.
  6. Chưng cất kiến thức từ ConvNeXt-T sang MobileNetV3 cho thiết bị không có GPU.

## 9. Phụ lục

| exp_id | Cấu hình | Ảnh biểu đồ | Log |
|---|---|---|---|
| B01–B06 | T00 trên từng backbone (mục 3) | `curves/B0x_*.png` | `logs/B0x/seed0/` |
| T00 | công thức nền, seed 0/1/2 | `curves/T00_convnext_tiny_baseline*.png` | `logs/T00/seed{0,1,2}/` |
| T01–T14, C01 | T00 + một thay đổi (mục 4) | `curves/T*.png`, `curves/C01_*.png` | `logs/T*/seed0/` |
| I00–I08 | suy luận trên model T14 seed 0 | `figures/inference_tradeoff.png` | `results/inference_val.csv`, `results/latency.csv` |
| F01 | T14 × seed 0/1/2 + I04_256 + TS | `curves/F01_convnext_tiny_T14_seed*.png` | `logs/F01/seed*/`, `results/final.json` |

- Cấu hình đầy đủ của từng lần chạy (hyperparameter, tag trọng số, phiên bản thư viện, kết quả kiểm tra split) nằm trong `logs/<exp_id>/seed<k>/config.json`. Các quyết định chọn và lý do nằm trong `results/selection.json`.
- Notebook đã chạy kèm output: `code/lab_day2_kaggle_executed.ipynb`. Notebook sạch để chạy lại: `code/lab_day2.ipynb`. Link và cách chạy có trong `README.md`.
