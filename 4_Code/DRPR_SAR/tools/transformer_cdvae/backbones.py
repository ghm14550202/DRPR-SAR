import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


ATR_BENCH = (
    Path(os.environ["ATR_BENCH_ROOT"]).expanduser()
    if os.environ.get("ATR_BENCH_ROOT")
    else None
)
VIT_DIR = ATR_BENCH / "Classification" / "ViT" if ATR_BENCH else None
HIVIT_DIR = ATR_BENCH / "Classification" / "HiViT" if ATR_BENCH else None
SWIN_DIR = ATR_BENCH / "Classification" / "Swin_Transformer" if ATR_BENCH else None
HIVIT_PRETRAINED = HIVIT_DIR / "model" / "mae_hivit_base_1600ep.pth" if HIVIT_DIR else None


def strip_module_prefix(state_dict):
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def strip_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def add_prefix(state_dict, prefix):
    return {k if k.startswith(prefix) else prefix + k: v for k, v in state_dict.items()}


def flexible_load_state_dict(model, state_dict, strict=False):
    candidates = [
        strip_module_prefix(state_dict),
        strip_prefix(strip_module_prefix(state_dict), "model."),
        add_prefix(strip_module_prefix(state_dict), "model."),
    ]

    best_msg = None
    best_score = None
    best_state = None
    model_keys = set(model.state_dict().keys())
    for candidate in candidates:
        matched = len(model_keys.intersection(candidate.keys()))
        if best_score is None or matched > best_score:
            best_score = matched
            best_state = candidate

    best_msg = model.load_state_dict(best_state, strict=strict)
    print(
        "Loaded state dict with {} matched keys, {} missing keys, {} unexpected keys.".format(
            best_score, len(best_msg.missing_keys), len(best_msg.unexpected_keys)
        )
    )
    return best_msg


def load_checkpoint(model, checkpoint_path, strict=False):
    if not checkpoint_path:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return flexible_load_state_dict(model, checkpoint, strict=strict)


def _import_from(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def build_vit(num_classes, pretrained=True):
    weights = None
    if pretrained:
        weights_cls = getattr(models, "ViT_B_16_Weights", None)
        weights = getattr(weights_cls, "DEFAULT", None) if weights_cls is not None else None
    try:
        model = models.vit_b_16(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Warning: failed to load pretrained ViT weights: {exc}. Falling back to random initialization.")
        model = models.vit_b_16(weights=None)
    model.heads = nn.Linear(768, num_classes)
    return model


def build_hivit(num_classes, pretrained=True):
    if HIVIT_DIR is None:
        raise RuntimeError("Set ATR_BENCH_ROOT before using the HiViT backbone.")
    _import_from(HIVIT_DIR)
    if pretrained and HIVIT_PRETRAINED is not None and HIVIT_PRETRAINED.is_file():
        os.environ.setdefault("HIVIT_PRETRAINED", str(HIVIT_PRETRAINED))
    elif not pretrained:
        os.environ["HIVIT_PRETRAINED"] = ""

    from model.HiVit import HiViT_base

    model = HiViT_base(num_classes)
    return model


def build_swin(num_classes, pretrained=True, swin_backbone="swin_t"):
    if SWIN_DIR is None:
        raise RuntimeError("Set ATR_BENCH_ROOT before using the Swin backbone.")
    _import_from(SWIN_DIR)
    from model.SwinTransformer import SwinTransformer

    model = SwinTransformer(
        num_classes=num_classes,
        backbone=swin_backbone,
        pretrained=pretrained,
        allow_pretrained_fallback=True,
    )
    return model


def build_classifier(model_name, num_classes, pretrained=True, checkpoint_path=None, strict=False):
    model_name = model_name.lower()
    if model_name == "vit":
        model = build_vit(num_classes, pretrained=pretrained)
    elif model_name == "hivit":
        model = build_hivit(num_classes, pretrained=pretrained)
    elif model_name in {"swin", "swin_t", "swin_transformer"}:
        model = build_swin(num_classes, pretrained=pretrained, swin_backbone="swin_t")
    elif model_name == "swin_s":
        model = build_swin(num_classes, pretrained=pretrained, swin_backbone="swin_s")
    elif model_name == "swin_b":
        model = build_swin(num_classes, pretrained=pretrained, swin_backbone="swin_b")
    else:
        raise ValueError(f"Unsupported transformer backbone: {model_name}")

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=strict)
        print(f"Loaded classifier checkpoint from {checkpoint_path}")
    return model


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizedClassifier(nn.Module):
    def __init__(self, classifier, normalize=True):
        super().__init__()
        self.normalize = ImageNetNormalize() if normalize else nn.Identity()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.normalize(x))


def classifier_needs_imagenet_normalize(model_name):
    # HiViT training scripts in this repo use ToTensor only, while ViT/Swin use ImageNet normalization.
    return model_name.lower() not in {"hivit"}
