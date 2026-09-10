from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _to_numpy_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def _stretch01(x: np.ndarray, p_low: float = 0.5, p_high: float = 99.9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo):
        lo = float(np.min(x))
    if not np.isfinite(hi):
        hi = float(np.max(x))
    if hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def save_preview(path: Path, masked: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = _to_numpy_2d(masked)
    mask = _to_numpy_2d(mask)
    pred = _to_numpy_2d(pred)
    target = _to_numpy_2d(target)

    masked_s = _stretch01(masked, p_low=0.5, p_high=99.9)
    pred_s = _stretch01(pred, p_low=0.5, p_high=99.9)
    target_s = _stretch01(target, p_low=0.5, p_high=99.9)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(masked_s, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input (masked)")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask (1=hidden)")
    axes[2].imshow(pred_s, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(target_s, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Target")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
