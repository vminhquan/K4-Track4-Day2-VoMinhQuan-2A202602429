"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện (giữ nguyên theo starter/):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """"ce" | "ls" (label smoothing) | "focal" | "ce_weighted". kw: smoothing, gamma, alpha, weight."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=tensor")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """CE với label smoothing, tự cài: q'(k) = (1 - eps) * 1[k == y] + eps / K. eps = 0 cho đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = float(smoothing)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        k = logits.shape[-1]
        q = torch.full_like(logp, self.eps / k)
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.eps + self.eps / k)
        return -(q * logp).sum(-1).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t). gamma = 0 và alpha = None cho đúng CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp_min(0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh TRAIN. beta=0: 1/n_c chuẩn hoá trung bình 1; beta>0: Cui et al. (tổng = K)."""
    n = np.asarray(counts, dtype=np.float64)
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        w = 1.0 / n
    w = w * len(n) / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = rng.integers(h), rng.integers(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn batch. Trả (x_mix, (y_a, y_b, lam)); với CutMix lam = 1 - diện tích THỰC của hộp / diện tích ảnh."""
    rng = rng or np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
