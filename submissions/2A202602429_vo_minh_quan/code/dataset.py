"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (giữ nguyên theo starter/):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn tiền xử lý lúc đánh giá: ảnh gốc 256x256 -> Resize(img_size / 0.875) -> CenterCrop(img_size).
Với img_size = 224 bước Resize giữ nguyên 256 nên chỉ còn CenterCrop(224) (đúng như GUIDE mục 1.4).
Với độ phân giải kiểm tra khác (I04: 256/288/320) tỉ lệ crop 0.875 được giữ cố định.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi trọng số timm dùng trong bài đều dùng mean/std ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_CROP_PCT = 0.875                   # 224 / 256
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1). Không sửa, không lọc."""
    labels_dir = Path(labels_dir)
    dfs = tuple(pd.read_csv(labels_dir / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))
    n = int(os.environ.get("LAB_SMOKE", "0"))
    if n:  # CHỈ để kiểm thử code nhanh (smoke test), không bao giờ dùng cho thí nghiệm thật
        dfs = tuple(d.groupby("Label", group_keys=False).head(n).reset_index(drop=True) for d in dfs)
    return dfs


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Lỗi thì raise AssertionError."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in splits.items()}
    total = sum(n.values())
    ratio = {k: v / total for k, v in n.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).tolist()
                 for k, v in splits.items()}
    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert v["Filename"].is_unique, f"{k}: Filename bị trùng"
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    on_disk = set(os.listdir(images_dir))
    missing = {k: sorted(v - on_disk)[:5] for k, v in names.items() if v - on_disk}

    assert all(c == 0 for c in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    smoke = bool(int(os.environ.get("LAB_SMOKE", "0")))
    assert smoke or union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"thiếu file ảnh: {missing}"
    for k, target in {"train": 0.6, "val": 0.2, "test": 0.2}.items():
        assert smoke or abs(ratio[k] - target) < 0.01, f"tỉ lệ {k} = {ratio[k]:.4f} lệch > 1 điểm % khỏi {target}"

    out = {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap, "union": union,
           "missing_files": 0}
    if verbose:
        print("Số ảnh:", n, "| tỉ lệ:", {k: round(v, 4) for k, v in ratio.items()})
        print("Giao:", overlap, "| hợp:", union, "| thiếu file: 0")
        print(pd.DataFrame(per_class, index=CLASS_NAMES))
    return out


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Transform theo `train` và `aug` ("basic" | "color" | "trivial" | "randaug" | "vflip").

    - basic   : RandomResizedCrop + lật ngang
    - color   : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
    - trivial : basic + TrivialAugmentWide
    - randaug : basic + RandAugment(2, 9)
    - vflip   : basic + lật dọc + xoay 90° ngẫu nhiên (ảnh chụp từ trên xuống nên hướng không cố định)
    Đánh giá: Resize(img_size / 0.875) + CenterCrop(img_size), không ngẫu nhiên.
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(int(round(img_size / EVAL_CROP_PCT))), T.CenterCrop(img_size), *norm])

    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "vflip":
        ops += [T.RandomVerticalFlip(), T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)])]
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*ops, *norm])


def denormalize(x):
    """Đảo Normalize cho tensor (C, H, W) hoặc (N, C, H, W) để vẽ ảnh sau augmentation."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1)


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # cho phép import module không cần torch (đọc CSV, check_split)
    _TorchDataset = object

# Bộ nhớ đệm ảnh đã giải mã (uint8 HxWx3), nạp ở tiến trình chính. Trên Linux (Colab/Kaggle) worker
# của DataLoader được fork nên dùng chung bộ nhớ này, không phải giải mã JPEG lại (Colab chỉ có 2 CPU).
RAM_CACHE: dict[str, np.ndarray] = {}


def preload_images(filenames, images_dir: str | Path, threads: int = 8) -> None:
    """Giải mã trước toàn bộ ảnh vào RAM_CACHE (17.509 ảnh 256x256 ≈ 3,4 GB)."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    todo = [f for f in dict.fromkeys(filenames) if f not in RAM_CACHE]

    def load(f):
        with Image.open(Path(images_dir) / f) as im:
            return f, np.asarray(im.convert("RGB"))

    with ThreadPoolExecutor(threads) as ex:
        for f, arr in ex.map(load, todo):
            RAM_CACHE[f] = arr


class DeepWeedsDataset(_TorchDataset):
    """Đọc ảnh từ `images_dir` theo DataFrame (Filename, Label). __getitem__ -> (tensor, int, str)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        from PIL import Image

        f = self.filenames[i]
        if f in RAM_CACHE:
            img = Image.fromarray(RAM_CACHE[f])
        else:
            with Image.open(self.images_dir / f) as im:
                img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], f


def _worker_init(worker_id: int) -> None:
    import random

    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """DataLoader. Train: shuffle hoặc WeightedRandomSampler ("balanced"); eval: giữ thứ tự df."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator().manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[np.asarray(ds.labels)]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
        drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, worker_init_fn=_worker_init, generator=g,
    )
