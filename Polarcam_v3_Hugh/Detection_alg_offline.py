# Detection_alg_offline.py
#
# Simplified spot detection backend using a Difference of Gaussians (DoG)
# band-pass filter on the (unnormalised) S-map.
#
# All tunable parameters are collected here near the top so you can quickly
# adjust the detector behavior without touching the GUI logic.

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------------------------------------------------------
# Tunable Parameters
# -----------------------------------------------------------------------------

# Number of frames used to compute the initial S-map (range metric).
S_MAP_FRAMES: int = 80

# Spatial smoothing applied to X/Y before range tracking.
S_MAP_SMOOTH_K: int = 5  # box filter kernel size

# Exclude candidates close to the edge (in full-resolution pixel units).
EDGE_EXCLUDE_PX: int = 10

# DoG band-pass parameters (full-resolution pixels).
DOG_SIGMA_SMALL: float = 1.0
DOG_SIGMA_LARGE: float = 6.0

# Thresholding: keep DoG pixels > mean + k*std (and positive).
DOG_K_STD: float = 8.0

# Connected component area filter (in full-resolution pixels).
DOG_MIN_AREA: int = 10
DOG_MAX_AREA: int = 250

# 8-connected is typical for blobs.
DOG_CONNECTIVITY: int = 8


def to_u8_preview(
    img: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 100.0
) -> np.ndarray:
    """
    Convert a 2D image to uint8 for display using percentile clipping,
    but treat exact zeros as background and keep them at 0 in the output.
    Percentiles are computed over NONZERO pixels only.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    x = img.astype(np.float32, copy=False)

    zero_mask = (x == 0.0)
    nz = x[~zero_mask]
    if nz.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    lo = float(np.percentile(nz, lo_pct))
    hi = float(np.percentile(nz, hi_pct))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(nz))
        hi = float(np.max(nz))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)

    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    u8[zero_mask] = 0
    return u8


def save_s_histogram(
    s_map: np.ndarray,
    out_path,
    bins: int = 120,
    title: str = "S_map pixel distribution",
    ignore_zeros: bool = False,
) -> None:
    """
    Save a PNG histogram of per-pixel S values (from an S_map).
    """
    if s_map.ndim != 2:
        raise ValueError(f"Expected 2D s_map, got shape {s_map.shape}")

    vals = s_map[np.isfinite(s_map)].astype(np.float32, copy=False)
    if ignore_zeros:
        vals = vals[vals != 0.0]

    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 20, 50, 70
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    if vals.size == 0:
        draw.text(
            (margin_left, margin_top),
            "No finite pixels to histogram",
            fill=(0, 0, 0),
            font=font,
        )
        img.save(out_path)
        return

    lo = float(np.percentile(vals, 0.5))
    hi = float(np.percentile(vals, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0

    counts, _edges = np.histogram(vals, bins=int(bins), range=(lo, hi))
    counts = counts.astype(np.int64, copy=False)
    max_c = int(counts.max()) if counts.size else 1
    max_c = max(1, max_c)

    plot_w = max(1, width - margin_left - margin_right)
    plot_h = max(1, height - margin_top - margin_bottom)
    x0, y0 = margin_left, margin_top
    x1, y1 = margin_left + plot_w - 1, margin_top + plot_h - 1

    draw.text((margin_left, 12), title, fill=(0, 0, 0), font=font)
    draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=1)
    draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=1)

    n = counts.size
    if n > 0:
        bar_w = max(1, int(plot_w / n))
        for i, c in enumerate(counts):
            bh = int(round((c / max_c) * (plot_h - 1)))
            bx0 = x0 + i * bar_w
            bx1 = min(x1, bx0 + bar_w - 1)
            by0 = y1 - bh
            draw.rectangle([bx0, by0, bx1, y1], fill=(70, 130, 180))

    for frac, val in [(0.0, lo), (0.5, (lo + hi) * 0.5), (1.0, hi)]:
        tx = x0 + int(round(frac * (plot_w - 1)))
        draw.line([(tx, y1), (tx, y1 + 4)], fill=(0, 0, 0), width=1)
        draw.text((tx - 18, y1 + 8), f"{val:.2f}", fill=(0, 0, 0), font=font)

    draw.text((x0 - 30, y1 - 6), "0", fill=(0, 0, 0), font=font)
    draw.text((x0 - 44, y0 - 6), str(max_c), fill=(0, 0, 0), font=font)

    draw.text(
        (margin_left + plot_w // 2 - 30, height - margin_bottom + 35),
        "S value",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((10, margin_top + plot_h // 2 - 10), "Count", fill=(0, 0, 0), font=font)

    img.save(out_path)


def _finite_stats(x: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x)
    if not finite.any():
        return (0.0, 0.0)
    vals = x[finite].astype(np.float32, copy=False)
    return (float(vals.mean()), float(vals.std()))


def find_spot_centers_dog(
    img: np.ndarray,
    sigma_small: float = DOG_SIGMA_SMALL,
    sigma_large: float = DOG_SIGMA_LARGE,
    k_std: float = DOG_K_STD,
    min_area: int = DOG_MIN_AREA,
    max_area: Optional[int] = DOG_MAX_AREA,
    connectivity: int = DOG_CONNECTIVITY,
) -> list[tuple[float, float]]:
    """
    DoG blob detection for 3-15px diameter-ish spots (tune sigmas/area as needed).

    Steps:
      - DoG = Gaussian(img, sigma_small) - Gaussian(img, sigma_large)
      - threshold at mean + k*std (and >0)
      - connected components
      - area filter
      - return centroids (x,y)
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D img, got shape {img.shape}")
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for DoG spot detection.")

    x = img.astype(np.float32, copy=False)
    # Replicate borders to avoid artificial negative halos at edges.
    g1 = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=float(sigma_small), borderType=cv2.BORDER_REPLICATE)
    g2 = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=float(sigma_large), borderType=cv2.BORDER_REPLICATE)
    dog = g1 - g2

    mu, sd = _finite_stats(dog)
    thr = max(0.0, mu + float(k_std) * sd)
    mask = (dog > thr) & np.isfinite(dog)
    if not mask.any():
        return []

    mask_u8 = mask.astype(np.uint8, copy=False)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=int(connectivity))

    max_area_i = None if max_area is None else int(max_area)
    centers: list[tuple[float, float]] = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        if max_area_i is not None and area > max_area_i:
            continue
        cx, cy = centroids[lbl]
        centers.append((float(cx), float(cy)))

    return centers
