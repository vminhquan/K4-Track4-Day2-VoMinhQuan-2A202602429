"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ nguyên theo starter/):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Tên timm kèm tag trọng số cố định (timm 1.0.x) để tái lập; tag được ghi vào config.json mỗi lần chạy.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0):
    """Tạo model phân loại 9 lớp. `init`: "scratch" | "frozen" | "finetune" (trục A)."""
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    name = SUGGESTED_BACKBONES.get(name, name)
    kw = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
    model = timm.create_model(name, pretrained=(pretrained and init != "scratch"),
                              num_classes=num_classes, drop_rate=drop_rate, **kw)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.weight_tag = f"{cfg.get('architecture', name)}.{cfg.get('tag', '')}".rstrip(".")
    model.init_mode = init
    if init == "scratch":
        model.weight_tag = f"{cfg.get('architecture', name)} (random init)"
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head. BN của backbone được giữ ở eval bởi set_train_mode()."""
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì mọi module ngoài head về eval
    (BatchNorm không cập nhật running stats, dropout trong backbone tắt)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head_modules = set(model.get_classifier().modules())
        for m in model.modules():
            if m is not model and m not in head_modules:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm như slide trang 52: backbone (ndim>1, có WD) · norm/bias backbone (WD=0) · head (lr_head, WD).

    Bias của head (ndim = 1) cũng không áp WD.
    """
    head = _head_param_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "backbone"
        kind = "decay" if p.ndim > 1 else "no_decay"
        groups[f"{part}_{kind}"].append(p)
    spec = {
        "backbone_decay": (lr_backbone, weight_decay),
        "backbone_no_decay": (lr_backbone, 0.0),
        "head_decay": (lr_head, weight_decay),
        "head_no_decay": (lr_head, 0.0),
    }
    return [{"params": ps, "lr": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, đếm bằng fvcore (fvcore đếm 1 MAC = 1 "flop")."""
    import copy

    import torch
    from fvcore.nn import FlopCountAnalysis

    m = copy.deepcopy(model).cpu().float().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    fca = FlopCountAnalysis(m, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    with torch.no_grad():
        return fca.total() / 1e9
