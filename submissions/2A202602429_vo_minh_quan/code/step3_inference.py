"""Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại) + đo độ trễ.

Chạy từ thư mục bài nộp (sau khi đã `experiments.py --select-recipe`):
    python code/step3_inference.py --preload
Mặc định: mô hình chính = runs/<recipe_exp>/seed0 (công thức chọn ở Bước 2); ensemble với 2 backbone khác
tốt nhất của Bước 1; greedy soup từ các lần chạy Bước 2 cùng backbone.
Ghi: results/inference_val.csv, results/latency.csv, figures/inference_tradeoff.png, results/selection.json
Mọi lựa chọn chỉ dựa trên val. Test không được đụng tới ở đây.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402
import dataset as ds  # noqa: E402
import inference as inf  # noqa: E402
from model import build_model  # noqa: E402
from train import REPO_ROOT, get_device  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics  # noqa: E402

# Phương pháp một mô hình có thể dùng ở chung kết: tên -> đặc tả (dùng lại ở step4_final.py)
SPECS = {
    "I00": {"views": "center", "space": "prob", "res": 224, "K": 1},
    "I01": {"views": "flip", "space": "prob", "res": 224, "K": 2},
    "I02a": {"views": "5crop", "space": "prob", "res": 224, "K": 5},
    "I02b": {"views": "10crop", "space": "prob", "res": 224, "K": 10},
    "I03a": {"views": "flip", "space": "logit", "res": 224, "K": 2},
    "I03b": {"views": "10crop", "space": "logit", "res": 224, "K": 10},
    "I04_256full": {"views": "full", "space": "prob", "res": 256, "K": 1},
    "I04_192": {"views": "resize", "space": "prob", "res": 192, "K": 1},
    "I04_256": {"views": "resize", "space": "prob", "res": 256, "K": 1},
    "I04_288": {"views": "resize", "space": "prob", "res": 288, "K": 1},
    "I04_320": {"views": "resize", "space": "prob", "res": 320, "K": 1},
}


def load_run(run: str | Path, device):
    run = Path(run)
    rec = json.loads((run / "config.json").read_text())
    c = rec["config"]
    m = build_model(c["backbone"], pretrained=False, init="finetune", drop_rate=c["drop_rate"],
                    drop_path_rate=c.get("drop_path_rate", 0.0))
    m.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    return m.to(device).eval(), rec


def metrics(y, probs):
    m = compute_metrics(y, probs.argmax(1), probs)
    return {"macro_f1": m["macro_f1"], "top1": m["top1"], "bal_acc": m["balanced_acc"], "ece": m["ece"],
            "nll": m["nll"], "f1_chinee": m["f1"][0], "f1_snake": m["f1"][7]}


def center(x, s=224):
    o = (x.shape[-1] - s) // 2
    return x[..., o:o + s, o:o + s]


def model_for_res(model, size: int):
    """Mô hình chạy được ở độ phân giải `size`. CNN có global pooling: dùng nguyên. ViT/DeiT/Swin (timm) cố định
    224 nên tạo bản sao rồi gọi set_input_size (nội suy position embedding / dựng lại cửa sổ attention)."""
    if size == 224 or not hasattr(model, "set_input_size") or \
            getattr(getattr(model, "patch_embed", None), "img_size", None) is None:
        return model
    m = copy.deepcopy(model)
    m.set_input_size(img_size=(size, size))
    return m.eval()


def spec_logits(model, df, images_dir, device, spec, workers=4):
    """Logit theo từng view của một đặc tả SPECS -> (filenames, y, list[logits]). Dùng cho val và test."""
    model = model_for_res(model, spec["res"])
    if spec["views"] == "resize":
        ld = ds.make_loader(df, images_dir, ds.build_transforms(False, spec["res"]), 64, False, None, workers)
        f, y, l = inf.predict_logits(model, ld, device)
        return f, y, [l]
    tf = T.Compose([T.ToTensor(), T.Normalize(ds.IMAGENET_MEAN, ds.IMAGENET_STD)])  # ảnh 256 nguyên
    ld = ds.make_loader(df, images_dir, tf, 64, False, None, workers)
    view = {
        "center": lambda x: [center(x)],
        "flip": lambda x: [center(x), inf.view_hflip(center(x))],
        "5crop": lambda x: inf.views_multicrop(x, 224),
        "10crop": lambda x: inf.views_multicrop(x, 224, flip=True),
        "full": lambda x: [x],
    }[spec["views"]]
    return inf.predict_logits(model, ld, device, view)


def spec_probs(logits_list, spec):
    return inf.aggregate_views(logits_list, spec["space"]) if len(logits_list) > 1 else inf._softmax(logits_list[0])


def spec_logit_mean(logits_list):
    """Logit gộp (trung bình) dùng cho temperature scaling sau TTA."""
    return np.mean(np.stack(logits_list), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="thư mục lần chạy chính; mặc định runs/<recipe_exp>/seed0")
    ap.add_argument("--ensemble", nargs="*", help="lần chạy khác để ensemble; mặc định 2 backbone khác tốt nhất")
    ap.add_argument("--soup", nargs="*", help="ứng viên soup cùng kiến trúc; mặc định các lần chạy Bước 2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()

    sel_path = Path("results/selection.json")
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    run_main = args.run or f"runs/{sel['recipe_exp']}/seed0"
    if args.ensemble is None:
        bdf = pd.read_csv("results/backbones.csv")
        bdf = bdf[bdf.backbone != sel.get("backbone")].sort_values("val_macro_f1", ascending=False)
        args.ensemble = [f"runs/{e}/seed0" for e in bdf.exp_id.head(2)]
    if args.soup is None:
        args.soup = []
        for p in sorted(Path("runs").glob("[TC]*/seed0/config.json")):
            c = json.loads(p.read_text())["config"]
            if (c["backbone"] == sel.get("backbone") and c["init"] == "finetune" and str(p.parent) != run_main
                    and (p.parent / "best.pt").exists()):
                args.soup.append(str(p.parent))

    device = get_device()
    dev = str(device)
    model, rec = load_run(run_main, device)
    cfg = rec["config"]
    _, val_df, _ = ds.load_split(cfg["labels_dir"], 0)
    if args.preload:
        ds.preload_images(val_df.Filename, cfg["images_dir"])
    images_dir = cfg["images_dir"]
    rows, lat_rows, store = [], [], {}
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    tag = rec["weight_tag"]
    src = Path(run_main).parent.name

    def lat(m, k=1, size=224, dtype="fp32", fused=False):
        return bm.latency_report(model_for_res(m, size), 1, size, dtype, dev, iters=args.iters, n_views=k,
                                 fused_bn=fused)

    def add(eid, method, K, probs, y, latency=None, note="", models=src):
        r = {"exp_id": eid, "method": method, "model": models, "K": K, **metrics(y, probs), "note": note}
        if latency:
            r.update({"p50_ms": latency["p50"], "p95_ms": latency["p95"], "p99_ms": latency["p99"],
                      "imgs_per_s_b1": latency["images_per_s"]})
        rows.append(r)
        store[eid] = probs
        print(f"{eid:12s} {method:55s} F1 {r['macro_f1']:.4f} top1 {r['top1']:.4f} ECE {r['ece']:.4f}"
              + (f" p95 {latency['p95']:.1f}ms" if latency else ""), flush=True)

    # ---- I00-I04: các phương pháp một mô hình ----
    cache = {}
    names_ref = None
    for eid, spec in SPECS.items():
        try:
            key = (spec["views"], spec["res"])
            if key not in cache:
                cache[key] = spec_logits(model, val_df, images_dir, device, spec, args.workers)
            f, y, L = cache[key]
            names_ref = names_ref or f
            assert f == names_ref
            desc = {"center": "1 view (center crop 224 từ ảnh 256)", "flip": "TTA lật ngang",
                    "5crop": "TTA 5 crop 224 (4 góc + giữa) từ ảnh 256", "10crop": "TTA 10 crop (5 crop + lật)",
                    "full": "ảnh 256 nguyên (không crop)",
                    "resize": f"độ phân giải {spec['res']} (Resize {round(spec['res'] / 0.875)} + CenterCrop)"}
            method = desc[spec["views"]] + (f", gộp {'LOGIT' if spec['space'] == 'logit' else 'xác suất'}"
                                            if spec["K"] > 1 else "")
            same_cost = eid.startswith("I03")
            add(eid, method, spec["K"], spec_probs(L, spec), y,
                None if same_cost else lat(model, spec["K"], spec["res"]),
                note="cùng chi phí với bản gộp xác suất" if same_cost else "")
        except Exception as e:  # ví dụ DeiT/Swin không nhận ảnh khác 224
            print(f"{eid}: không áp dụng được ({type(e).__name__}: {e})")
            rows.append({"exp_id": eid, "method": str(spec), "model": src, "K": spec["K"],
                         "note": f"không áp dụng: {type(e).__name__}"})
    y = np.asarray(cache[("center", 224)][1])
    l_center = cache[("center", 224)][2][0]

    # ---- I05: ensemble khác backbone (trung bình xác suất, mỗi mô hình 1 view) ----
    ens_probs, ens_models = [inf._softmax(l_center)], [model]
    for r in args.ensemble:
        m2, rec2 = load_run(r, device)
        f2, y2, l2 = spec_logits(m2, val_df, images_dir, device, SPECS["I00"], args.workers)
        assert (np.asarray(y2) == y).all() and f2 == names_ref
        ens_probs.append(inf._softmax(l2[0]))
        ens_models.append(m2)
        add(f"I05_{Path(r).parent.name}", f"(thành viên ensemble) {rec2['weight_tag']} 1 view", 1, ens_probs[-1], y,
            models=Path(r).parent.name, note="thành viên")
    if len(ens_probs) > 1:
        class Ens(torch.nn.Module):
            def __init__(self, ms):
                super().__init__()
                self.ms = torch.nn.ModuleList(ms)

            def forward(self, x):
                return torch.stack([m(x).softmax(1) for m in self.ms]).mean(0)

        names_e = "+".join([src] + [Path(r).parent.name for r in args.ensemble])
        add("I05", f"Ensemble {len(ens_probs)} mô hình khác backbone (TB xác suất)", len(ens_probs),
            inf.ensemble_probs(ens_probs), y, lat(Ens(ens_models)), models=names_e)
        for mm in ens_models[1:]:
            mm.cpu()

    # ---- I06: greedy soup (Wortsman et al. 2022): thêm checkpoint nếu macro-F1 val không giảm ----
    if args.soup:
        def soup_eval(paths):
            sds = [torch.load(Path(p) / "best.pt", map_location="cpu", weights_only=True) for p in paths]
            avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype)
                   if sds[0][k].dtype.is_floating_point else sds[0][k] for k in sds[0]}
            sm = copy.deepcopy(model).cpu()
            sm.load_state_dict(avg)
            sm.to(device).eval()
            _, _, Ls = spec_logits(sm, val_df, images_dir, device, SPECS["I00"], args.workers)
            p = inf._softmax(Ls[0])
            return sm, p, metrics(y, p)["macro_f1"]

        f1_of = {}
        for p in args.soup:
            s = json.loads((Path(p) / "summary.json").read_text())
            f1_of[p] = s["val_macro_f1"]
        ingredients = [run_main]
        best_sm, best_p, best_f1 = model, inf._softmax(l_center), metrics(y, inf._softmax(l_center))["macro_f1"]
        log = []
        for p in sorted(f1_of, key=lambda p: -f1_of[p]):
            sm, pr, f1 = soup_eval(ingredients + [p])
            keep = f1 >= best_f1
            log.append(f"{Path(p).parent.name}:{'+' if keep else '-'}{f1:.4f}")
            if keep:
                ingredients.append(p)
                best_sm, best_p, best_f1 = sm, pr, f1
            else:
                sm.cpu()
        add("I06", f"Greedy soup {len(ingredients)} checkpoint", 1, best_p, y, lat(best_sm),
            models="+".join(Path(p).parent.name for p in ingredients),
            note="thứ tự thử: " + ", ".join(log) + " (BN stats trung bình cùng trọng số)")
        if best_sm is not model:  # không có checkpoint nào được thêm thì best_sm chính là model gốc: giữ trên GPU
            best_sm.cpu()

    # ---- I07: temperature scaling: T khớp trên val; ECE báo cáo bằng 2-fold chéo trên val ----
    def ts_crossfit(logits):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        a, b = idx[: len(y) // 2], idx[len(y) // 2:]
        p = np.zeros((len(y), 9))
        Ta, Tb = inf.fit_temperature(logits[a], y[a]), inf.fit_temperature(logits[b], y[b])
        p[b] = inf.apply_temperature(logits[b], Ta)
        p[a] = inf.apply_temperature(logits[a], Tb)
        return p, inf.fit_temperature(logits, y), Ta, Tb

    p, T_full, Ta, Tb = ts_crossfit(l_center)
    add("I07", f"1 view + temperature scaling (T={T_full:.3f})", 1, p, y, None,
        note=f"ECE đo bằng 2-fold chéo trên val (T nửa A={Ta:.3f}, B={Tb:.3f}); chi phí như I00")
    L_flip = cache.get(("flip", 224))
    if L_flip:
        lf = spec_logit_mean(L_flip[2])
        p, T_fl, _, _ = ts_crossfit(lf)
        add("I07b", f"TTA lật (gộp logit) + temperature scaling (T={T_fl:.3f})", 2, p, y, None,
            note="ECE 2-fold chéo trên val; chi phí như I01")

    # ---- I08: gộp BN, FP16, AMP ----
    xchk = torch.randn(2, 3, 224, 224, device=device)
    fused = inf.fuse_conv_bn(model, check_input=xchk)
    ld = ds.make_loader(val_df, images_dir, ds.build_transforms(False, 224), 64, False, None, args.workers)
    if fused.n_fused_bn:
        _, _, ls = inf.predict_logits(fused, ld, device)
        add("I08a", f"Gộp BN vào conv ({fused.n_fused_bn} cặp), FP32", 1, inf._softmax(ls), y, lat(fused, fused=True),
            note=f"sai số tuyệt đối lớn nhất so với gốc {fused.fuse_max_abs_diff:.1e}")
    base_half = fused if fused.n_fused_bn else model
    half = copy.deepcopy(base_half).half().eval()
    ls = []
    with torch.inference_mode():
        for xb, _, _ in ld:
            ls.append(half(xb.to(device).half()).float().cpu().numpy())
    del half
    add("I08b", "FP16 (model.half()" + (" + gộp BN)" if fused.n_fused_bn else ")"), 1,
        inf._softmax(np.concatenate(ls)), y, lat(base_half, dtype="fp16", fused=bool(fused.n_fused_bn)))
    _, _, ls = inf.predict_logits(model, ld, device, amp=True)
    add("I08c", "AMP autocast fp16", 1, inf._softmax(ls), y, lat(model, dtype="amp"))

    # ---- bảng Latency: batch 1 và 32; GPU và CPU; FP32/FP16/AMP; có/không gộp BN ----
    for d in dict.fromkeys([dev, "cpu"]):
        for batch in (1, 32):
            for dtype, m, fz in [("fp32", model, False), ("fp32", fused, True), ("fp16", fused, True),
                                 ("amp", model, False)]:
                if (fz and not fused.n_fused_bn) or (d == "cpu" and dtype != "fp32"):
                    continue
                r = bm.latency_report(m, batch, 224, dtype, d, iters=args.iters if batch == 1 else 30, fused_bn=fz)
                lat_rows.append({"config": f"{tag} 1 view", **r})
                print(f"latency {d} b{batch} {dtype} fusedBN={fz}: p50 {r['p50']:.2f} p95 {r['p95']:.2f} "
                      f"p99 {r['p99']:.2f} ms | {r['images_per_s']:.0f} ảnh/s", flush=True)
    for k, name in [(2, "TTA lật"), (10, "TTA 10 crop")]:
        r = bm.tta_latency(model, k, batch_size=1, img_size=224, dtype="fp32", device=dev, iters=args.iters)
        lat_rows.append({"config": f"{tag} {name} (K={k})", **r})

    df = pd.DataFrame(rows)
    base = df.loc[df.exp_id == "I00"].iloc[0]
    df["delta_f1_vs_I00"] = df["macro_f1"] - base["macro_f1"]
    df["cost_vs_I00"] = df["p50_ms"] / base["p50_ms"]
    df.insert(3, "checkpoint", f"{run_main} ({tag})")
    df.to_csv("results/inference_val.csv", index=False)
    pd.DataFrame(lat_rows).to_csv("results/latency.csv", index=False)
    np.savez_compressed("results/inference_val_probs.npz", y=y, **store)

    # ---- chọn phương pháp suy luận cho chung kết (chỉ val) ----
    # Quy tắc: trong SPECS (một mô hình, không cần nhiều checkpoint), lấy macro-F1 val cao nhất; nếu phương
    # pháp rẻ hơn (K x độ phân giải nhỏ hơn) kém không quá 0,002 thì lấy phương pháp rẻ hơn.
    # Sau đó luôn áp dụng temperature scaling (T khớp trên val): không đổi dự đoán, chỉ cải thiện hiệu chuẩn.
    c = df[df.exp_id.isin(SPECS) & df.macro_f1.notna()].copy()
    c["cost"] = [SPECS[e]["K"] * (SPECS[e]["res"] / 224) ** 2 for e in c.exp_id]
    top = c.macro_f1.max()
    pick = c[c.macro_f1 >= top - 0.002].sort_values(["cost", "macro_f1"], ascending=[True, False]).iloc[0]
    # cấu hình thời gian thực: rẻ nhất có p95 <= 100 ms, ưu tiên F1
    rt = df[(df.p95_ms <= 100) & df.exp_id.isin(SPECS)].sort_values("macro_f1", ascending=False)
    sel.update({"inference": pick.exp_id, "inference_spec": SPECS[pick.exp_id],
                "inference_val_f1_seed0": float(pick.macro_f1), "inference_rule_top_f1": float(top),
                "realtime_inference": rt.exp_id.iloc[0] if len(rt) else None,
                "T_val_seed0": T_full, "inference_device": bm.device_name(dev)})
    sel_path.write_text(json.dumps(sel, indent=2, ensure_ascii=False))
    print("CHỌN suy luận:", pick.exp_id, SPECS[pick.exp_id], f"(val F1 {pick.macro_f1:.4f}; tốt nhất {top:.4f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d2 = df.dropna(subset=["p95_ms"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(d2["p95_ms"], d2["macro_f1"])
    for _, r in d2.iterrows():
        ax.annotate(r["exp_id"], (r["p95_ms"], r["macro_f1"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axvline(100, color="red", ls="--", lw=1, label="ngân sách 100 ms (p95, batch 1)")
    ax.set(xscale="log", xlabel=f"độ trễ p95 batch 1 (ms, log) · {bm.device_name(dev)} · FP32 trừ I08",
           ylabel="macro-F1 val", title=f"Đánh đổi độ chính xác - độ trễ ({tag}, {src})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/inference_tradeoff.png", dpi=120)
    print(df.to_string())


if __name__ == "__main__":
    main()
