"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện (giữ nguyên theo starter/):
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader, gom logit theo đúng thứ tự file.

    `view`: hàm biến đổi batch (một batch -> một batch) hoặc (một batch -> list batch); khi trả list,
    kết quả là list logit theo từng view (dùng cho TTA nhiều view trong một lượt đọc dữ liệu).
    """
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device)
            xs = view(x) if view is not None else x
            multi = isinstance(xs, (list, tuple))
            xs = xs if multi else [xs]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                ls = [model(v).float().cpu() for v in xs]
            if outs is None:
                outs = [[] for _ in ls]
            for o, l in zip(outs, ls):
                o.append(l)
            ys.append(y)
            names.extend(f)
    y = torch.cat(ys).numpy().astype(np.int64)
    logits = [torch.cat(o).numpy() for o in outs]
    return names, y, (logits if multi else logits[0])


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch ảnh lớn hơn; tuỳ chọn thêm bản lật (10 view)."""
    h, w = x.shape[-2:]
    tops = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]
    views = [x[..., t:t + crop, l:l + crop] for t, l in tops]
    if flip:
        views += [view_hflip(v) for v in views]
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). CNN có global pooling chạy
    được với mọi kích thước; DeiT/Swin cần nội suy position embedding/cửa sổ nên không dùng ở đây."""
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear", antialias=True,
                                                     align_corners=False) for s in sizes]


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax."""
    L = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = np.mean([_softmax(l) for l in L], axis=0)
    elif space == "logit":
        p = _softmax(L.mean(0))
    else:
        raise ValueError(f"space không hợp lệ: {space}")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    shapes = {np.shape(p) for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình có số dòng khác nhau: {shapes}")
    p = np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T). LBFGS trên log T, kiểm tra lại bằng lưới."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t_lbfgs = float(log_t.exp().item())
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    t_grid = float(grid[int(np.argmin(nll))])
    nll_l = F.cross_entropy(z / t_lbfgs, y).item()
    return t_lbfgs if nll_l <= min(nll) + 1e-9 else t_grid


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(bn.bias + (b - bn.running_mean) * scale)
    return fused.to(conv.weight.device)


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity; BatchNormAct2d của timm (BN + drop + act gộp sẵn) -> Sequential(drop, act)."""
    if hasattr(bn, "act") and hasattr(bn, "drop"):
        return nn.Sequential(bn.drop, bn.act)
    return nn.Identity()


def fuse_conv_bn(model, check_input=None, verbose: bool = True):
    """Gộp BatchNorm vào Conv2d liền trước (theo thứ tự đăng ký module con, đúng với ResNet/EfficientNet/
    MobileNetV3 của timm vì forward đi đúng thứ tự đó). Trả về bản sao đã gộp; không đụng model gốc.

    Nếu có `check_input`, in sai số tuyệt đối lớn nhất giữa đầu ra trước và sau gộp.
    ViT/DeiT/Swin/ConvNeXt dùng LayerNorm nên không áp dụng (số cặp gộp = 0).
    """
    model.eval()
    fused = copy.deepcopy(model).eval()
    n_fused = 0

    def visit(parent: nn.Module):
        nonlocal n_fused
        names = list(parent._modules.keys())
        for a, b in zip(names, names[1:]):
            ca, cb = parent._modules[a], parent._modules[b]
            if isinstance(ca, nn.Conv2d) and isinstance(cb, nn.BatchNorm2d) and cb.track_running_stats \
                    and ca.out_channels == cb.num_features:
                parent._modules[a] = _fuse_pair(ca, cb)
                parent._modules[b] = _bn_replacement(cb)
                n_fused += 1
        for child in parent._modules.values():
            if child is not None:
                visit(child)

    visit(fused)
    fused.n_fused_bn = n_fused
    if check_input is not None:
        with torch.inference_mode():
            diff = (model(check_input) - fused(check_input)).abs().max().item()
        fused.fuse_max_abs_diff = diff
        if verbose:
            print(f"fuse_conv_bn: gộp {n_fused} cặp conv-BN, sai số lớn nhất {diff:.2e}")
    return fused
