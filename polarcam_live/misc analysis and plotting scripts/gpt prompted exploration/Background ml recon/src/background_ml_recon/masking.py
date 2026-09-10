from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class MaskConfig:
    min_rectangles: int = 1
    max_rectangles: int = 4
    min_rect_frac: float = 0.08
    max_rect_frac: float = 0.35
    blur_sigma: float = 1.2
    threshold: float = 0.35
    extra_random_pixel_prob: float = 0.0


def random_patch_mask(height: int, width: int, cfg: MaskConfig) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    n_rects = random.randint(cfg.min_rectangles, cfg.max_rectangles)

    for _ in range(n_rects):
        rh = int(height * random.uniform(cfg.min_rect_frac, cfg.max_rect_frac))
        rw = int(width * random.uniform(cfg.min_rect_frac, cfg.max_rect_frac))
        rh = max(4, min(rh, height))
        rw = max(4, min(rw, width))
        y0 = random.randint(0, max(0, height - rh))
        x0 = random.randint(0, max(0, width - rw))
        mask[y0 : y0 + rh, x0 : x0 + rw] = 1.0

    if cfg.blur_sigma > 0:
        mask = gaussian_filter(mask, sigma=cfg.blur_sigma)
        mask = (mask > cfg.threshold).astype(np.float32)

    if cfg.extra_random_pixel_prob > 0:
        sparse = np.random.rand(height, width) < cfg.extra_random_pixel_prob
        mask[sparse] = 1.0

    return mask
