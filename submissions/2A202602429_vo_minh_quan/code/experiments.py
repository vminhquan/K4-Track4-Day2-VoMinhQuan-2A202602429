"""Danh sách thí nghiệm (một nơi duy nhất), trình chạy tuần tự và các quy tắc chọn cấu hình TRÊN VAL.

Chạy từ thư mục bài nộp (cwd chứa runs/, curves/, predictions/, results/):
    python code/experiments.py --group B                 # Bước 1: 6 backbone, seed 0
    python code/experiments.py --select-backbone         # đo độ trễ sơ bộ B0x + chọn backbone (val)
    python code/experiments.py --group T                 # Bước 2: T00 (3 seed) + T01..T14 (seed 0)
    python code/experiments.py --group C                 # Bước 2: kết hợp yếu tố tốt (C01, C02)
    python code/experiments.py --select-recipe           # chọn công thức tốt nhất (val, seed 0)
Lần chạy đã có runs/<exp_id>/seed<k>/summary.json thì bỏ qua; lần chạy dở (last.pt) được tiếp tục.
Mọi quy tắc chọn chỉ đọc chỉ số VAL (summary.json); test không được đọc ở bất kỳ bước nào trước Bước 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Config, run, run_dir  # noqa: E402

SEL_PATH = Path("results/selection.json")
BASE = dict(epochs=10)  # cùng số epoch cho mọi lần chạy (GUIDE cho phép 10-15)
if os.environ.get("LAB_SMOKE"):  # chỉ để kiểm thử code nhanh
    BASE = dict(epochs=1, batch_size=16)
BASELINE_SEEDS = (0, 1, 2)  # T00 chạy 3 seed ngay từ đầu: vừa đo nhiễu cho ablation, vừa là mốc chung kết

BACKBONES = [  # (exp_id, key trong model.SUGGESTED_BACKBONES)
    ("B01", "resnet50"),
    ("B02", "convnext_tiny"),
    ("B03", "deit_small"),
    ("B04", "swin_tiny"),
    ("B05", "efficientnet_b0"),
    ("B06", "mobilenetv3"),
]

TRAINING = [  # (exp_id, trục, khác T00 ở điểm nào, overrides)
    ("T00", "-", "công thức nền", {}),
    ("T01", "A", "khởi tạo ngẫu nhiên (từ đầu), LR backbone 1e-3", {"init": "scratch", "lr_backbone": 1e-3}),
    ("T02", "A", "đóng băng backbone, chỉ train head", {"init": "frozen"}),
    ("T03", "B", "aug + TrivialAugmentWide", {"aug": "trivial"}),
    ("T04", "B", "aug + lật dọc + xoay 90°", {"aug": "vflip"}),
    ("T05", "B", "CutMix (α=1)", {"mix": "cutmix"}),
    ("T06", "B", "Mixup (α=0.2)", {"mix": "mixup", "mix_alpha": 0.2}),
    ("T07", "C", "label smoothing ε=0.1", {"loss": "ls", "label_smoothing": 0.1}),
    ("T08", "C", "focal loss γ=2", {"loss": "focal", "focal_gamma": 2.0}),
    ("T09", "C", "CE trọng số lớp 1/n_c (train)", {"loss": "ce_weighted", "class_weight_beta": 0.0}),
    ("T10", "D", "sampler cân bằng lớp", {"sampler": "balanced"}),
    ("T11", "E", "LR head = LR backbone = 1e-4", {"lr_head": 1e-4}),
    ("T12", "E", "LR x3 (backbone 3e-4, head 3e-3)", {"lr_backbone": 3e-4, "lr_head": 3e-3}),
    ("T13", "F", "EMA trọng số d=0.999", {"ema_decay": 0.999}),
    ("T14", "G", "20 epoch thay vì 10", {"epochs": 20}),
]
# Các trục mà giá trị loại trừ nhau khi kết hợp (chọn giá trị tốt nhất của mỗi nhóm)
EXCLUSIVE = {"aug": ["T03", "T04"], "mix": ["T05", "T06"], "loss": ["T07", "T08", "T09"], "lr": ["T11", "T12"]}
NOT_IN_COMBO = {"T01", "T02", "T14"}  # đổi khởi tạo / gấp đôi ngân sách: không đưa vào kết hợp


def load_sel() -> dict:
    return json.loads(SEL_PATH.read_text()) if SEL_PATH.exists() else {}


def save_sel(sel: dict) -> None:
    SEL_PATH.parent.mkdir(exist_ok=True)
    SEL_PATH.write_text(json.dumps(sel, indent=2, ensure_ascii=False))


def summary(exp_id: str, seed: int = 0) -> dict | None:
    p = Path("runs") / exp_id / f"seed{seed}" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def t_overrides() -> dict[str, dict]:
    sel = load_sel()
    d = {eid: ov for eid, _, _, ov in TRAINING}
    for c in sel.get("combos", []):
        d[c["exp_id"]] = c["overrides"]
    return d


def configs(group: str, only: list[str] | None = None) -> list[Config]:
    sel = load_sel()
    out = []
    if group == "B":
        out = [Config(exp_id=eid, backbone=bb, desc=bb, seed=0, **BASE) for eid, bb in BACKBONES]
    elif group in ("T", "C"):
        bb = sel.get("backbone")
        if not bb:
            raise SystemExit("Chưa chọn backbone: chạy `--select-backbone` sau Bước 1.")
        items = ([(e, ov) for e, _, _, ov in TRAINING] if group == "T"
                 else [(c["exp_id"], c["overrides"]) for c in sel.get("combos", [])])
        for eid, ov in items:
            seeds = BASELINE_SEEDS if eid == "T00" else (0,)
            for s in seeds:
                # T00: lưu dự đoán TEST lúc train (được phép theo GUIDE N2) nhưng KHÔNG mở cho tới Bước 4
                out.append(Config(exp_id=eid, backbone=bb, desc=f"{bb}_{_slug(ov)}", seed=s,
                                  save_test_predictions=(eid == "T00"), **{**BASE, **ov}))
    if only:
        out = [c for c in out if c.exp_id in only]
    return out


def _slug(ov: dict) -> str:
    if not ov:
        return "baseline"
    return "-".join(f"{k}{v}" for k, v in ov.items()).replace(".", "p").replace("_", "")[:40]


def run_all(cfgs: list[Config], preload: bool) -> list[str]:
    """Chạy tuần tự; lỗi ở một lần chạy không dừng các lần sau. Trả về danh sách lần chạy bị lỗi."""
    failed = []
    for cfg in cfgs:
        cfg.preload = preload
        if (run_dir(cfg) / "summary.json").exists():
            print(f"bỏ qua {cfg.exp_id} seed{cfg.seed} (đã xong)")
            continue
        try:
            s = run(cfg)
            print(json.dumps({k: s[k] for k in ("exp_id", "seed", "val_macro_f1", "val_top1", "best_epoch")}))
        except Exception:
            traceback.print_exc()
            print(f"LỖI ở {cfg.exp_id} seed{cfg.seed}, chạy tiếp thí nghiệm sau", flush=True)
            failed.append(f"{cfg.exp_id} seed{cfg.seed}")
    return failed


# --------------------------------------------------------------------------- quy tắc chọn (chỉ val)
def select_backbone(iters: int = 100) -> dict:
    """Đo độ trễ batch 1 sơ bộ của mỗi B0x rồi chọn backbone cho Bước 2.

    Quy tắc (định trước, chỉ val): lấy backbone có macro-F1 val cao nhất; nếu một backbone khác kém hơn
    không quá 0,005 (cỡ nhiễu 1 seed) mà nhanh hơn >= 1,5 lần (p50 batch 1) thì chọn backbone nhanh hơn.
    """
    import pandas as pd
    import torch

    import benchmark as bm
    from model import build_model
    from train import get_device

    dev = str(get_device())
    rows = []
    for eid, bb in BACKBONES:
        s = summary(eid)
        if s is None:
            print(f"thiếu {eid}, bỏ qua")
            continue
        m = build_model(bb, pretrained=False)
        m.load_state_dict(torch.load(Path("runs") / eid / "seed0" / "best.pt", map_location="cpu", weights_only=True))
        r = bm.latency_report(m, 1, 224, "fp32", dev, iters=iters)
        r32 = bm.latency_report(m, 32, 224, "amp" if dev == "cuda" else "fp32", dev, iters=30)
        rows.append({"exp_id": eid, "backbone": bb, "weight_tag": s["weight_tag"], "params_M": s["params_M"],
                     "gmacs": s["gmacs"], "img_size": 224, "epochs": BASE["epochs"], "seed": 0,
                     "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"], "best_epoch": s["best_epoch"],
                     "train_time_per_epoch_s": s["train_time_per_epoch_s"], "lat_b1_p50_ms": r["p50"],
                     "lat_b1_p95_ms": r["p95"], "lat_b1_p99_ms": r["p99"], "thr_b32_imgs_s": r32["images_per_s"],
                     "device": r["gpu"], "curve": s["curve"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/backbones.csv", index=False)
    best = df.loc[df.val_macro_f1.idxmax()]
    choice, reason = best, f"macro-F1 val cao nhất ({best.val_macro_f1:.4f})"
    near = df[(df.val_macro_f1 >= best.val_macro_f1 - 0.005) & (df.lat_b1_p50_ms * 1.5 <= best.lat_b1_p50_ms)]
    if len(near):
        choice = near.loc[near.val_macro_f1.idxmax()]
        reason = (f"kém {best.backbone} {best.val_macro_f1 - choice.val_macro_f1:.4f} macro-F1 (<= 0,005, cỡ nhiễu "
                  f"1 seed) nhưng nhanh hơn {best.lat_b1_p50_ms / choice.lat_b1_p50_ms:.1f} lần ở batch 1")
    sel = load_sel()
    sel["backbone"] = choice.backbone
    sel["backbone_exp"] = choice.exp_id
    sel["backbone_reason"] = reason
    save_sel(sel)
    print("CHỌN backbone:", choice.backbone, "-", reason)
    return sel


def baseline_noise() -> tuple[float, float]:
    import numpy as np

    f = [summary("T00", s)["val_macro_f1"] for s in BASELINE_SEEDS if summary("T00", s)]
    return float(np.mean(f)), float(np.std(f, ddof=1)) if len(f) > 1 else float("nan")


def select_combos() -> dict:
    """Kết hợp (chỉ val, seed 0): yếu tố "có ích" = Δ macro-F1 so với T00 seed 0 > max(std T00 qua 3 seed, 0,003).
    Trong mỗi nhóm loại trừ (EXCLUSIVE) chỉ giữ giá trị tốt nhất. C01 = mọi yếu tố có ích; C02 = 2 yếu tố có Δ lớn nhất.
    """
    base = summary("T00", 0)["val_macro_f1"]
    _, std = baseline_noise()
    thr = max(std if std == std else 0.0, 0.003)
    gains = {}
    for eid, _, _, _ in TRAINING:
        s = summary(eid, 0)
        if eid == "T00" or eid in NOT_IN_COMBO or s is None:
            continue
        gains[eid] = s["val_macro_f1"] - base
    helpful = {e: g for e, g in gains.items() if g > thr}
    for _, members in EXCLUSIVE.items():
        inside = [e for e in members if e in helpful]
        for e in sorted(inside, key=lambda e: -helpful[e])[1:]:
            helpful.pop(e)
    ov = {e: ov for e, _, _, ov in TRAINING}
    ranked = sorted(helpful, key=lambda e: -helpful[e])
    group_of = {e: g for g, members in EXCLUSIVE.items() for e in members}

    def compatible(es):
        gs = [group_of.get(e, e) for e in es]
        return len(gs) == len(set(gs))

    def merge(es):
        out = {}
        for e in es:
            out.update(ov[e])
        return out

    combos = []
    if len(ranked) >= 2:
        combos.append({"exp_id": "C01", "from": ranked, "overrides": merge(ranked),
                       "desc": "kết hợp mọi yếu tố có ích: " + " + ".join(ranked)})
        if len(ranked) >= 3:
            combos.append({"exp_id": "C02", "from": ranked[:2], "overrides": merge(ranked[:2]),
                           "desc": f"kết hợp 2 yếu tố tốt nhất: {ranked[0]} + {ranked[1]}"})
    else:
        # < 2 yếu tố vượt ngưỡng nhiễu: vẫn thử một kết hợp (RUBRIC C) từ 2 yếu tố có Δ lớn nhất, tương thích nhau
        order = sorted(gains, key=lambda e: -gains[e])
        pair = next(([a, b] for i, a in enumerate(order) for b in order[i + 1:] if compatible([a, b])), None)
        if pair:
            combos.append({"exp_id": "C01", "from": pair, "overrides": merge(pair),
                           "desc": f"kết hợp 2 yếu tố có Δ lớn nhất (dưới/ngang ngưỡng nhiễu): {pair[0]} + {pair[1]}"})
    sel = load_sel()
    sel.update({"t00_seed0_val_f1": base, "t00_std_3seeds": std, "combo_threshold": thr,
                "gains_vs_T00": gains, "helpful": ranked, "combos": combos})
    save_sel(sel)
    print(json.dumps({k: sel[k] for k in ("gains_vs_T00", "combo_threshold", "helpful", "combos")}, indent=2,
                     ensure_ascii=False))
    return sel


def select_recipe() -> dict:
    """Công thức chung kết = macro-F1 val (seed 0) cao nhất trong T00..T14, C01, C02 (cùng seed 0)."""
    ovs = t_overrides()
    cands = {e: summary(e, 0) for e in ovs if summary(e, 0)}
    best = max(cands, key=lambda e: cands[e]["val_macro_f1"])
    sel = load_sel()
    sel.update({"recipe_exp": best, "recipe_overrides": ovs[best],
                "recipe_val_f1_seed0": cands[best]["val_macro_f1"],
                "recipe_candidates": {e: c["val_macro_f1"] for e, c in cands.items()}})
    save_sel(sel)
    print("CHỌN công thức:", best, ovs[best], f"(val F1 {cands[best]['val_macro_f1']:.4f})")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["B", "T", "C"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--preload", action="store_true", help="giải mã trước ảnh vào RAM (Colab/Kaggle)")
    ap.add_argument("--select-backbone", action="store_true")
    ap.add_argument("--select-combos", action="store_true")
    ap.add_argument("--select-recipe", action="store_true")
    args = ap.parse_args()
    if args.select_backbone:
        select_backbone()
    if args.select_combos:
        select_combos()
    if args.group == "C" and not load_sel().get("combos") and not args.select_combos:
        select_combos()
    if args.group:
        failed = run_all(configs(args.group, args.only), args.preload)
        if failed:  # mã thoát khác 0 để notebook dừng lại thay vì đi tiếp với kết quả thiếu
            raise SystemExit(f"Các lần chạy bị lỗi: {failed}. Sửa lỗi rồi chạy lại (lần đã xong sẽ được bỏ qua).")
    if args.select_recipe:
        select_recipe()


if __name__ == "__main__":
    main()
