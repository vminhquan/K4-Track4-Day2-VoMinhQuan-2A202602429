"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc áp dụng:
  - warmup: bỏ 10 lần chạy đầu (mặc định 20)
  - đồng bộ thiết bị TRƯỚC và SAU đoạn đo: torch.cuda.synchronize() hoặc torch.mps.synchronize()
  - >= 50 lần đo (mặc định 100), báo cáo p50, p95, p99
  - ghi thiết bị, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đọc JPEG, resize, chuẩn hoá): chỉ đo forward của model trên tensor đã ở thiết bị
"""
from __future__ import annotations

import platform
import time

import numpy as np


def _sync_fn(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device.startswith("mps"):
        return torch.mps.synchronize
    return None


def device_name(device: str) -> str:
    import subprocess

    import torch

    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        chip = platform.processor()
    return f"{chip} {'GPU (MPS)' if device.startswith('mps') else 'CPU'}"


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây): warmup, rồi mỗi lần sync -> t0 -> fn -> sync -> t1."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 20, iters: int = 100, n_views: int = 1, fused_bn: bool = False) -> dict:
    """Độ trễ forward của `model` trên đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half()). n_views > 1: mỗi lần gọi chạy
    n_views lượt forward liên tiếp (đo TTA thật, không nhân K).
    """
    import copy

    import torch

    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        m = m.half()
        x = x.half()
    amp = dtype == "amp"
    dev_type = torch.device(device).type

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp):
            for _ in range(n_views):
                m(x)

    r = bench(fn, warmup, iters, _sync_fn(device))
    out = {"gpu": device_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
           "views": n_views, "fused_bn": fused_bn, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
           "mean": r["mean"], "n": r["n"], "warmup": warmup, "images_per_s": batch_size / (r["p50"] / 1000),
           "torch": torch.__version__, "includes_preprocessing": False}
    del m
    return out


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view, đo thật (K lượt forward liên tiếp); kèm tỉ lệ so với K * p50 của 1 view."""
    one = latency_report(model, n_views=1, **kw)
    k = latency_report(model, n_views=k_views, **kw)
    k["ratio_vs_k_times_1view"] = k["p50"] / (k_views * one["p50"])
    return k
