"""Bước 0: kiểm tra chia dữ liệu, EDA và kiểm tra pipeline (GUIDE mục 1.2-1.3).

Chạy từ thư mục bài nộp:  python code/step0_checks.py
Ghi: figures/eda_*.png, figures/aug_*.png, figures/overfit_one_batch.png, results/step0.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds  # noqa: E402
import losses  # noqa: E402
from model import build_model, set_train_mode  # noqa: E402
from train import Config, get_device, set_seed  # noqa: E402

PAPER_TABLE1 = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]


def main():
    cfg = Config()
    out_fig, out_res = Path("figures"), Path("results")
    out_fig.mkdir(exist_ok=True)
    out_res.mkdir(exist_ok=True)
    report = {}

    # ---- 1. kiểm tra chia dữ liệu (README 2.1) ----
    tr, va, te = ds.load_split(cfg.labels_dir, 0)
    info = ds.check_split(tr, va, te, cfg.images_dir)
    labels_all = pd.read_csv(Path(cfg.labels_dir) / "labels.csv")
    counts_all = labels_all["Label"].value_counts().reindex(range(9)).tolist()
    report["split"] = info
    report["counts_all"] = counts_all
    report["matches_paper_table1"] = counts_all == PAPER_TABLE1
    report["imbalance_ratio_max_min"] = max(counts_all) / min(counts_all)
    print("Khớp Table 1 bài báo:", report["matches_paper_table1"],
          "| tỉ lệ lớn nhất/nhỏ nhất = %.2f" % report["imbalance_ratio_max_min"])

    # ---- 2. EDA: phân bố lớp ----
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = np.arange(9)
    for i, (k, c) in enumerate(info["per_class"].items()):
        ax.bar(xs + (i - 1) * 0.27, c, width=0.27, label=f"{k} ({sum(c)})")
    ax.set_xticks(xs, ds.CLASS_NAMES, rotation=20)
    ax.set_ylabel("số ảnh")
    ax.set_title("DeepWeeds fold 0: số ảnh mỗi lớp theo tập (Negatives ≈ 52%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / "eda_class_distribution.png", dpi=120)
    plt.close(fig)

    # ảnh mẫu: 4 ảnh mỗi lớp, lấy từ TRAIN
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(9, 4, figsize=(8, 18))
    for c in range(9):
        files = tr[tr.Label == c].Filename.to_numpy()
        for j, f in enumerate(rng.choice(files, 4, replace=False)):
            axes[c, j].imshow(Image.open(Path(cfg.images_dir) / f))
            axes[c, j].axis("off")
        axes[c, 0].set_title(ds.CLASS_NAMES[c], loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "eda_samples.png", dpi=80)
    plt.close(fig)

    # thống kê ảnh: kích thước, mode, mean/std kênh (trên 500 ảnh train)
    sizes, modes, px = set(), set(), []
    for f in rng.choice(tr.Filename.to_numpy(), 500, replace=False):
        im = Image.open(Path(cfg.images_dir) / f)
        sizes.add(im.size)
        modes.add(im.mode)
        px.append(np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0)
    px = np.concatenate(px)
    report["image_stats"] = {"sizes": sorted(map(list, sizes)), "modes": sorted(modes),
                             "mean_rgb": px.mean(0).round(4).tolist(), "std_rgb": px.std(0).round(4).tolist()}
    print("Thống kê ảnh:", report["image_stats"])

    # ---- 3. kiểm tra ảnh sau augmentation (đã giải chuẩn hoá) ----
    for aug in ("basic", "trivial"):
        d = ds.DeepWeedsDataset(tr.sample(8, random_state=1), cfg.images_dir, ds.build_transforms(True, 224, aug))
        fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
        for a, i in zip(axes.flat, range(8)):
            x, y, f = d[i]
            a.imshow(ds.denormalize(x).permute(1, 2, 0).numpy())
            a.set_title(f"{ds.CLASS_NAMES[y]}\n{f}", fontsize=8)
            a.axis("off")
        fig.suptitle(f"Ảnh train sau augmentation '{aug}' (đã giải chuẩn hoá) và nhãn")
        fig.tight_layout()
        fig.savefig(out_fig / f"aug_{aug}.png", dpi=90)
        plt.close(fig)
    # CutMix trên một batch
    d = ds.DeepWeedsDataset(tr.sample(4, random_state=2), cfg.images_dir, ds.build_transforms(False, 224))
    x = torch.stack([d[i][0] for i in range(4)])
    y = torch.tensor([d[i][1] for i in range(4)])
    xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix", rng=np.random.default_rng(3))
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, a in enumerate(axes):
        a.imshow(ds.denormalize(xm[i]).permute(1, 2, 0).numpy())
        a.set_title(f"{ds.CLASS_NAMES[ya[i]]} / {ds.CLASS_NAMES[yb[i]]}\nλ={lam:.3f}", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_fig / "aug_cutmix.png", dpi=90)
    plt.close(fig)

    # ---- 4. loss ban đầu ≈ ln 9 và overfit một batch nhỏ ----
    set_seed(0)
    device = get_device()
    model = build_model("resnet50", pretrained=True).to(device)
    loader = ds.make_loader(tr, cfg.images_dir, ds.build_transforms(False, 224), 64, True, num_workers=4)
    xb, yb_, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        init_loss = F.cross_entropy(model(xb.to(device)), yb_.to(device)).item()
    report["initial_loss"] = {"value": init_loss, "ln9": float(np.log(9)), "model": model.weight_tag}
    print(f"Loss ban đầu {init_loss:.4f} (ln 9 = {np.log(9):.4f})")

    xs16, ys16 = xb[:16].to(device), yb_[:16].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    hist = []
    for step in range(150):
        set_train_mode(model)
        loss = F.cross_entropy(model(xs16), ys16)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = (model(xs16).argmax(1) == ys16).float().mean().item()
    report["overfit_16"] = {"loss_first": hist[0], "loss_last": hist[-1], "train_acc_eval_mode": acc, "steps": 150}
    print(f"Overfit 16 ảnh: loss {hist[0]:.3f} -> {hist[-1]:.4f}, acc (eval mode) {acc:.3f}")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogy(hist)
    ax.set(xlabel="bước", ylabel="loss (log)", title="Overfit 16 ảnh, resnet50, AdamW 1e-4")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / "overfit_one_batch.png", dpi=110)
    plt.close(fig)

    (out_res / "step0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
