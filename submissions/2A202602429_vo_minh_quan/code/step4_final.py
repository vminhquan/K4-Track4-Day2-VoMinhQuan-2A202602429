"""Bước 4: chung kết. Huấn luyện cấu hình đã chốt trên VAL với 3 seed, chạy TEST đúng một lần mỗi seed.

Chạy từ thư mục bài nộp (sau step3_inference.py):
    python code/step4_final.py --preload
Cấu hình (khai báo TRƯỚC khi mở test, đọc từ results/selection.json):
    F01   = backbone + công thức `recipe_exp` + suy luận `inference` + temperature scaling (T khớp trên val)
    F01_uncal = như F01 nhưng không temperature scaling (để chấm I4a)
    F01rt = cấu hình thời gian thực: cùng mô hình, 1 view + temperature scaling (I5)
    T00   = mốc: công thức nền + 1 view (dự đoán test đã ghi lúc train T00, chưa mở)
Ghi: predictions/F01_seed{k}_test.csv, F01_uncal_seed{k}_test.csv, F01_seed{k}_val.csv, F01rt_seed{k}_test.csv,
     results/final.json, results/eval/*, figures/confusion_*.png, figures/errors_chinee_snake.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402
import dataset as ds  # noqa: E402
import inference as inf  # noqa: E402
from experiments import BASE, BASELINE_SEEDS  # noqa: E402
from step3_inference import SPECS, load_run, model_for_res, spec_logits  # noqa: E402
from train import REPO_ROOT, Config, get_device, run, run_dir  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402

SEEDS = (0, 1, 2)


def calibrated(L: list[np.ndarray], space: str, T: float) -> np.ndarray:
    """Xác suất sau TTA với nhiệt độ T áp lên logit của từng view (T = 1 là bản chưa hiệu chuẩn)."""
    Ls = [l / T for l in L]
    return inf.aggregate_views(Ls, space) if len(Ls) > 1 else inf._softmax(Ls[0])


def fit_T_views(L: list[np.ndarray], y: np.ndarray, space: str) -> float:
    """T cực tiểu NLL trên VAL của xác suất cuối (sau gộp view). Một view: dùng inf.fit_temperature."""
    if len(L) == 1:
        return inf.fit_temperature(L[0], y)
    def nll(T):
        p = calibrated(L, space, T)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    grid = np.exp(np.linspace(np.log(0.1), np.log(10), 200))
    t0 = grid[int(np.argmin([nll(t) for t in grid]))]
    fine = np.linspace(t0 / 1.05, t0 * 1.05, 101)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def sh(cmd: list[str]) -> str:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sel = json.loads(Path("results/selection.json").read_text())
    bb, ov, inf_id = sel["backbone"], sel["recipe_overrides"], sel["inference"]
    spec, rt_spec = SPECS[inf_id], SPECS["I00"]
    print(f"Chung kết: {bb} + {sel['recipe_exp']} {ov} + {inf_id} {spec} + temperature scaling")
    device = get_device()
    out = {"backbone": bb, "recipe_exp": sel["recipe_exp"], "recipe_overrides": ov, "inference": inf_id,
           "inference_spec": spec, "seeds": list(SEEDS), "per_seed": []}

    # 1) huấn luyện F01 với 3 seed (không ghi test lúc train; test chạy một lần ở bước 2 với đúng pipeline cuối)
    for s in SEEDS:
        cfg = Config(exp_id="F01", backbone=bb, desc=f"{bb}_{sel['recipe_exp']}", seed=s, preload=args.preload,
                     **{**BASE, **ov})
        if not (run_dir(cfg) / "summary.json").exists():
            run(cfg)
    assert all(Path(f"predictions/T00_seed{s}_test.csv").exists() for s in BASELINE_SEEDS), "thiếu mốc T00"

    # 2) mỗi seed: val -> khớp T -> test MỘT lần
    _, val_df, test_df = ds.load_split(Config().labels_dir, 0)
    images_dir = Config().images_dir
    if args.preload:
        ds.preload_images(list(val_df.Filename) + list(test_df.Filename), images_dir)
    for s in SEEDS:
        model, rec = load_run(f"runs/F01/seed{s}", device)
        fv, yv, Lv = spec_logits(model, val_df, images_dir, device, spec, args.workers)
        yv = np.asarray(yv)
        T = fit_T_views(Lv, yv, spec["space"])
        pv = calibrated(Lv, spec["space"], T)
        save_predictions(f"predictions/F01_seed{s}_val.csv", fv, yv, pv)
        fr, yr, Lr = spec_logits(model, val_df, images_dir, device, rt_spec, args.workers)
        T_rt = inf.fit_temperature(Lr[0], np.asarray(yr))
        # ---- TEST: đúng một lượt suy luận cho mỗi pipeline đã khai báo ----
        ft, yt, Lt = spec_logits(model, test_df, images_dir, device, spec, args.workers)
        yt = np.asarray(yt)
        save_predictions(f"predictions/F01_seed{s}_test.csv", ft, yt, calibrated(Lt, spec["space"], T))
        save_predictions(f"predictions/F01_uncal_seed{s}_test.csv", ft, yt, calibrated(Lt, spec["space"], 1.0))
        if inf_id == "I00":
            Lt_rt, ft_rt = Lt, ft
        else:
            ft_rt, _, Lt_rt = spec_logits(model, test_df, images_dir, device, rt_spec, args.workers)
        save_predictions(f"predictions/F01rt_seed{s}_test.csv", ft_rt, yt, inf.apply_temperature(Lt_rt[0], T_rt))
        np.savez_compressed(f"runs/F01/seed{s}/final_logits.npz", val=np.stack(Lv), test=np.stack(Lt))
        mv = compute_metrics(yv, pv.argmax(1), pv)
        out["per_seed"].append({"seed": s, "T": T, "T_rt": T_rt, "val_macro_f1": mv["macro_f1"],
                                "val_top1": mv["top1"], "val_ece": mv["ece"]})
        print(out["per_seed"][-1], flush=True)
        model.cpu()

    # 3) độ trễ đúng cách của cấu hình chung kết và cấu hình thời gian thực (batch 1)
    model, _ = load_run("runs/F01/seed0", device)
    dev = str(device)
    lat_final = bm.latency_report(model_for_res(model, spec["res"]), 1, spec["res"], "fp32", dev, iters=200, n_views=spec["K"])
    lat_rt = bm.latency_report(model, 1, 224, "fp32", dev, iters=200)
    out["latency_final"], out["latency_rt"] = lat_final, lat_rt
    print("latency final", lat_final, "\nlatency rt", lat_rt)

    # 4) eval.py chính thức
    labels = str(Path(Config().labels_dir))
    common = ["--test-csv", f"{labels}/test_subset0.csv", "--labels", f"{labels}/labels.csv", "--out", "results/eval"]
    py = sys.executable
    for tag in ("F01", "F01_uncal", "F01rt", "T00"):
        sh([py, str(REPO_ROOT / "eval.py"), "score", "--pred", f"predictions/{tag}_seed*_test.csv", "--tag", tag,
            *common])
    g = sh([py, str(REPO_ROOT / "eval.py"), "grade", "--final", "predictions/F01_seed*_test.csv",
            "--baseline", "predictions/T00_seed*_test.csv", "--uncal", "predictions/F01_uncal_seed*_test.csv",
            "--final-val", "predictions/F01_seed*_val.csv", "--val-csv", f"{labels}/val_subset0.csv",
            "--latency-p95-ms", f"{lat_rt['p95']:.2f}", "--latency-method", "proper", *common])
    Path("results").mkdir(exist_ok=True)
    Path("results/grade.txt").write_text(g)
    Path("results/final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))

    # 5) phân tích lỗi trên test (sau khi đã chốt, không quay lại chỉnh): ma trận nhầm lẫn + ảnh sai
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    names = ds.CLASS_NAMES
    for tag in ("F01", "T00"):
        cms = []
        for s in SEEDS:
            d = pd.read_csv(f"predictions/{tag}_seed{s}_test.csv")
            cms.append(compute_metrics(d.y_true.to_numpy(), d.y_pred.to_numpy(),
                                       d[[f"p{i}" for i in range(9)]].to_numpy())["confusion"])
        cm = cms[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
        for i in range(9):
            for j in range(9):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm[i].sum() * 0.5 else "black")
        ax.set_xticks(range(9), names, rotation=45, ha="right")
        ax.set_yticks(range(9), names)
        ax.set(xlabel="dự đoán", ylabel="nhãn thật", title=f"Ma trận nhầm lẫn TEST · {tag} · seed 0 (số ảnh)")
        fig.tight_layout()
        fig.savefig(f"figures/confusion_{tag}_seed0.png", dpi=120)
        plt.close(fig)
        np.save(f"results/confusion_{tag}_seeds.npy", np.stack(cms))
    d = pd.read_csv("predictions/F01_seed0_test.csv")
    pairs = [(0, 7), (7, 0), (0, 8), (7, 8)]
    fig, axes = plt.subplots(len(pairs), 6, figsize=(13, 2.4 * len(pairs)))
    for r, (t, p) in enumerate(pairs):
        sub = d[(d.y_true == t) & (d.y_pred == p)]
        for c in range(6):
            a = axes[r, c]
            a.axis("off")
            if c < len(sub):
                row = sub.iloc[c]
                a.imshow(Image.open(Path(images_dir) / row.Filename))
                a.set_title(f"{names[t]}→{names[p]}\nconf {row[f'p{p}']:.2f}", fontsize=7)
        axes[r, 0].text(-0.1, 0.5, f"{names[t]} → {names[p]}\n({len(sub)} ảnh)", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Ảnh test bị đoán sai (F01 seed 0)")
    fig.tight_layout()
    fig.savefig("figures/errors_chinee_snake.png", dpi=90)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
