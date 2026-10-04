"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh (từ thư mục code/):
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc (không sửa eval.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


def _find_repo_root() -> Path:
    """Thư mục chứa eval.py gốc: biến môi trường LAB_REPO, hoặc tìm ngược lên từ file này."""
    if os.environ.get("LAB_REPO"):
        return Path(os.environ["LAB_REPO"]).resolve()
    for p in Path(__file__).resolve().parents:
        if (p / "eval.py").exists() and (p / "RUBRIC.md").exists():
            return p
    return Path.cwd()


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
SUBMISSION_DIR = CODE_DIR.parent


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn, dùng cho tên ảnh curves/<exp_id>_<desc>.png
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    grad_clip: float | None = None
    amp: bool = True                  # chỉ có hiệu lực trên CUDA (trên MPS đo được AMP chậm hơn FP32)
    num_workers: int = min(6, os.cpu_count() or 2)
    preload: bool = False             # giải mã trước mọi ảnh vào RAM (nên bật trên Colab/Kaggle)
    device: str = "auto"              # auto | cuda | mps | cpu
    # --- đường dẫn ---
    images_dir: str = str(REPO_ROOT / "data" / "images")
    labels_dir: str = str(REPO_ROOT / "data" / "labels")
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def get_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU/CUDA/MPS). Worker DataLoader seed qua generator + worker_init_fn.

    Mức tái lập: cùng seed cho cùng thứ tự batch, augmentation và khởi tạo head; kernel GPU (MPS/cuDNN)
    không bảo đảm tất định hoàn toàn nên số có thể lệch rất nhỏ giữa hai lần chạy.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum 0.9) với 3 nhóm tham số (model.param_groups)."""
    import torch

    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (mỗi iteration)."""
    import torch

    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước tối ưu (slide trang 56).

    Bản sao riêng `self.module` dùng để đánh giá. Buffer dạng float (running_mean/var của BN) cũng được
    trung bình động như tham số; buffer nguyên (num_batches_tracked) chép thẳng. d có warmup
    d_t = min(d, (1 + t) / (10 + t)) để những bước đầu không bị trọng số ngẫu nhiên của head chi phối.
    """

    def __init__(self, model, decay: float):
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model) -> None:
        import torch

        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src, alpha=1.0 - d)
                else:
                    v.copy_(src)


def _autocast(device, enabled: bool):
    import torch

    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, use_amp: bool = False) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr_steps"}."""
    import torch

    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    tot_loss, tot_correct, tot_n, lrs = 0.0, 0.0, 0, []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with _autocast(device, use_amp):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        # Lịch LR bám theo số bước (iteration), kể cả bước mà GradScaler bỏ qua optimizer.step() vì gradient
        # tràn số (vài bước đầu khi bật AMP). PyTorch cảnh báo "lr_scheduler.step() before optimizer.step()"
        # trong trường hợp này; đó là hành vi cố ý nên tắt cảnh báo, không đổi cách tính LR.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`")
            scheduler.step()
        if ema is not None:
            ema.update(model)
        n = y.size(0)
        tot_loss += loss.item() * n
        # accuracy train: với Mixup/CutMix chỉ mang tính tham khảo (so với nhãn gốc y_a)
        tot_correct += (logits.argmax(1) == y).float().sum().item()
        tot_n += n
    return {"train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n, "lr_steps": lrs}


def evaluate(model, loader, criterion, device, use_amp: bool = False):
    """Eval, không gradient. Trả về (filenames, y_true[N], logits[N, 9], loss) đúng thứ tự loader."""
    import numpy as np
    import torch

    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            with _autocast(device, use_amp):
                logits = model(x.to(device, non_blocking=True))
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y = torch.cat(ys)
    loss = float(torch.nn.functional.cross_entropy(logits, y).item()) if criterion is not None else float("nan")
    return names, y.numpy().astype(np.int64), logits.numpy(), loss


def softmax_np(z):
    import numpy as np

    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list[float] | None = None) -> None:
    """curves/<exp_id>_<desc>.png: loss train/val, macro-F1 + top-1 val, LR theo bước."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ncols = 3 if lr_steps else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    ax[1].plot(ep, [h["train_acc"] for h in history], "^--", alpha=0.6, label="train acc")
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best ep {best['epoch']} (F1 {best['val_macro_f1']:.4f})")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Metric")
    ax[1].legend(fontsize=8)
    if lr_steps:
        ax[2].plot(lr_steps)
        ax[2].set(xlabel="step", ylabel="LR (nhóm backbone)", title="LR (warmup + cosine)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _env_info(device) -> dict:
    import numpy
    import timm
    import torch
    import torchvision

    gpu = (torch.cuda.get_device_name(0) if device.type == "cuda"
           else f"Apple {platform.processor() or 'silicon'} GPU (MPS)" if device.type == "mps" else "CPU")
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "numpy": numpy.__version__, "device": str(device), "gpu": gpu,
            "platform": platform.platform()}


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    import numpy as np
    import pandas as pd
    import torch

    import dataset as ds
    from eval import compute_metrics, save_predictions
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params

    # 1. seed, thư mục, config
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)

    # 2. dữ liệu + kiểm tra S1-S4
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    split_info = ds.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    if cfg.preload:
        ds.preload_images(list(train_df.Filename) + list(val_df.Filename)
                          + (list(test_df.Filename) if cfg.save_test_predictions else []), cfg.images_dir)

    # 3. loader (test chỉ khi chung kết)
    tf_train = ds.build_transforms(True, cfg.img_size, cfg.aug)
    tf_eval = ds.build_transforms(False, cfg.img_size)
    train_loader = ds.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler,
                                  cfg.num_workers, seed=cfg.seed)
    val_loader = ds.make_loader(val_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers)

    # 4. model, loss, optimizer, scheduler, EMA
    model = build_model(cfg.backbone, True, ds.NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate)
    n_params = count_params(model)
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except Exception as e:  # fvcore không đếm được một số op: ghi lại, không dừng
        print("count_gmacs lỗi:", e)
        gmacs = float("nan")
    model.to(device)
    kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = np.bincount(train_df["Label"], minlength=ds.NUM_CLASSES)  # chỉ dùng TRAIN
        w = class_weights(counts, cfg.class_weight_beta or 0.0)
        kw["weight"] = w.to(device)
        if cfg.loss == "focal":
            kw["alpha"] = w
    criterion = build_criterion(cfg.loss, **kw).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    record = {"config": dataclasses.asdict(cfg), "weight_tag": model.weight_tag, "params_M": n_params,
              "gmacs": gmacs, "amp_effective": use_amp, "env": _env_info(device), "split": split_info}
    (rd / "config.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(f"[{cfg.exp_id} seed{cfg.seed}] {model.weight_tag} | {n_params:.2f}M params | {gmacs:.3f} GMAC | "
          f"device {device} | amp {use_amp} | {len(train_loader)} bước/epoch")

    # 5. vòng epoch, chọn checkpoint theo macro-F1 val (hòa -> epoch sớm hơn: chỉ thay khi lớn hơn hẳn)
    history, all_lrs = [], []
    best_f1, best_epoch, start_epoch, prev_time = -1.0, -1, 1, 0.0
    if (rd / "last.pt").exists():  # tiếp tục sau khi phiên Colab bị ngắt
        st = torch.load(rd / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if ema is not None:
            ema.module.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng_cpu"].cpu())  # map_location=cuda đã chuyển cả RNG state lên GPU
        history, all_lrs = st["history"], st["lrs"]
        best_f1, best_epoch, prev_time = st["best_f1"], st["best_epoch"], st["elapsed"]
        start_epoch = history[-1]["epoch"] + 1 if history else 1
        print(f"  tiếp tục từ epoch {start_epoch} (best F1 {best_f1:.4f} ở epoch {best_epoch})")
    t_start = time.time() - prev_time
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema, use_amp)
        if device.type == "mps":
            torch.mps.synchronize()
        t_train = time.time() - t0
        all_lrs.extend(tr.pop("lr_steps"))
        eval_model = ema.module if ema is not None else model
        _, yv, lv, vloss = evaluate(eval_model, val_loader, criterion, device, use_amp)
        pv = softmax_np(lv)
        m = compute_metrics(yv, pv.argmax(1), pv)
        row = {"epoch": epoch, **tr, "val_loss": vloss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"], "lr_end": all_lrs[-1],
               "train_time_s": t_train, "epoch_time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        flag = ""
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), rd / "best.pt")
            np.save(rd / "val_logits.npy", lv)
            flag = " *"
        print(f"  ep {epoch:2d} | train {tr['train_loss']:.4f} | val loss {vloss:.4f} | val F1 {m['macro_f1']:.4f} "
              f"| top1 {m['top1']:.4f} | {t_train:.0f}s{flag}", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "ema": ema.module.state_dict() if ema else None,
                    "ema_updates": ema.updates if ema else 0, "scaler": scaler.state_dict() if scaler else None,
                    "rng_cpu": torch.get_rng_state(), "history": history, "lrs": all_lrs, "best_f1": best_f1,
                    "best_epoch": best_epoch, "elapsed": time.time() - t_start}, rd / "last.pt")
    total_time = time.time() - t_start

    # 6. checkpoint tốt nhất -> dự đoán val
    eval_model = ema.module if ema is not None else model
    eval_model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=True))
    fv, yv, lv, _ = evaluate(eval_model, val_loader, criterion, device, use_amp)
    np.save(rd / "val_logits.npy", lv)
    pd.Series(fv).to_csv(rd / "val_filenames.csv", index=False, header=["Filename"])
    save_predictions(pred_path(cfg, "val"), fv, yv, softmax_np(lv))
    pv = softmax_np(lv)
    mv = compute_metrics(yv, pv.argmax(1), pv)

    # 7. test: đúng MỘT lần, chỉ ở chung kết
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(test_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers)
        ft, yt, lt, _ = evaluate(eval_model, test_loader, criterion, device, use_amp)
        np.save(rd / "test_logits.npy", lt)
        pd.Series(ft).to_csv(rd / "test_filenames.csv", index=False, header=["Filename"])
        save_predictions(pred_path(cfg, "test"), ft, yt, softmax_np(lt))
        pt = softmax_np(lt)
        mt = compute_metrics(yt, pt.argmax(1), pt)
        test_metrics = {"macro_f1": mt["macro_f1"], "top1": mt["top1"], "ece": mt["ece"]}

    # 8. đường cong + tóm tắt
    desc = cfg.desc or cfg.backbone
    name = f"{cfg.exp_id}_{desc}" + (f"_seed{cfg.seed}" if cfg.exp_id.startswith("F") or cfg.seed != 0 else "")
    plot_curves(history, Path(cfg.curves_dir) / f"{name}.png",
                f"{cfg.exp_id} · {model.weight_tag} · seed {cfg.seed} · {cfg.epochs} ep", all_lrs)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "desc": desc, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "params_M": n_params, "gmacs": gmacs, "best_epoch": best_epoch, "val_macro_f1": mv["macro_f1"],
        "val_top1": mv["top1"], "val_bal_acc": mv["balanced_acc"], "val_ece": mv["ece"],
        "val_f1_per_class": mv["f1"].tolist(), "val_recall_per_class": mv["recall"].tolist(),
        "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
        "total_time_s": total_time, "curve": f"{name}.png", "test": test_metrics,
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (rd / "last.pt").unlink(missing_ok=True)  # chỉ giữ best.pt để tiết kiệm dung lượng Drive
    print(f"[{cfg.exp_id} seed{cfg.seed}] best ep {best_epoch} | val F1 {mv['macro_f1']:.4f} | top1 {mv['top1']:.4f}"
          f" | {total_time / 60:.1f} phút")
    return summary


def _cast(value: str, type_str: str):
    t = str(type_str)
    if "None" in t and value.lower() in ("none", "null", ""):
        return None
    if "bool" in t:
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"không đọc được bool: {value}")
    if t.startswith("int"):
        return int(value)
    if t.startswith("float"):
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict, ép kiểu theo field của Config."""
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"cần dạng KEY=VALUE, nhận: {pair}")
        k, v = pair.split("=", 1)
        if k not in fields:
            raise KeyError(f"Config không có trường '{k}'. Các trường: {sorted(fields)}")
        out[k] = _cast(v, fields[k])
    return out


def main() -> None:
    """python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
