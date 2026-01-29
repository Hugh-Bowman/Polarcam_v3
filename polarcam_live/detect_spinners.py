# detect_spinners.py
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# -----------------------------------------------------------------------------
# TUNABLE MIXING WEIGHT
# a = 0.0  -> pure "change" metric (median S_t^2)
# a = 1.0  -> pure "intensity" metric (median (Q^2 + U^2))
# 0<a<1    -> linear blend:  a*I_med + (1-a)*st2_med
# -----------------------------------------------------------------------------
a: float = 0.0  # <-- set this between 0 and 1

# Number of frames used to compute the initial S-map used for spot finding in the GUI.
ST2_MEDIAN_FRAMES: int = 50


def save_s_histogram(
    s_map: np.ndarray,
    out_path,
    bins: int = 120,
    title: str = "S_map pixel distribution",
    ignore_zeros: bool = False,
) -> None:
    """
    Save a PNG histogram of per-pixel S values (from an S_map).

    Parameters
    ----------
    s_map : np.ndarray
        2D array of S values.
    out_path : str | PathLike
        Output PNG path.
    bins : int
        Histogram bin count.
    title : str
        Plot title.
    ignore_zeros : bool
        If True, compute histogram over non-zero pixels only.
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

    # Robust range so a few outliers don't destroy readability.
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

    # Title
    draw.text((margin_left, 12), title, fill=(0, 0, 0), font=font)

    # Axes
    draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=1)
    draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=1)

    # Bars
    n = counts.size
    if n > 0:
        bar_w = max(1, int(plot_w / n))
        for i, c in enumerate(counts):
            bh = int(round((c / max_c) * (plot_h - 1)))
            bx0 = x0 + i * bar_w
            bx1 = min(x1, bx0 + bar_w - 1)
            by0 = y1 - bh
            draw.rectangle([bx0, by0, bx1, y1], fill=(70, 130, 180))

    # Ticks + labels
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


def median_st2(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Return a blended per-pixel map:
        out = a * I_med + (1-a) * st2_med

    where
        st2_med = median_t( (Q[t+1]-Q[t])^2 + (U[t+1]-U[t])^2 )
        I_med   = median_t( Q[t]^2 + U[t]^2 )

    Parameters
    ----------
    Q, U : np.ndarray
        Arrays of shape (T, H, W). Any integer/float dtype is accepted.

    Returns
    -------
    out : np.ndarray
        Array of shape (H, W), dtype float32.
    """
    if Q.shape != U.shape:
        raise ValueError(
            f"Q and U must have same shape, got {Q.shape} vs {U.shape}"
        )
    if Q.ndim != 3:
        raise ValueError(
            f"Expected Q/U shape (T,H,W); got ndim={Q.ndim}, shape={Q.shape}"
        )
    if Q.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute differences.")

    # Clamp a defensively (so GUI doesn't crash if you accidentally set it out of range)
    aa = float(a)
    if aa < 0.0:
        aa = 0.0
    elif aa > 1.0:
        aa = 1.0

    # Promote before squaring to avoid overflow for int16/int32.
    Q = Q.astype(np.int32, copy=False)
    U = U.astype(np.int32, copy=False)

    # --- Change metric: median S_t^2 over time differences ---
    dQ = np.diff(Q, axis=0)  # (T-1, H, W)
    dU = np.diff(U, axis=0)
    st2 = dQ * dQ + dU * dU
    st2_med = np.median(st2, axis=0).astype(np.float32, copy=False)

    # --- Intensity metric: median (Q^2 + U^2) over time ---
    I = Q * Q + U * U
    I_med = np.median(I, axis=0).astype(np.float32, copy=False)

    # Blend
    out = aa * I_med + (1.0 - aa) * st2_med
    return out.astype(np.float32, copy=False)


def to_u8_preview(
    img: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 100.0
) -> np.ndarray:
    """
    Convert a 2D image to uint8 for display using percentile clipping,
    but treat exact zeros as background and keep them at 0 in the output.

    Percentiles are computed over NONZERO pixels only.

    Parameters
    ----------
    img : np.ndarray
        2D array (H, W)
    lo_pct, hi_pct : float
        Percentiles for contrast stretching (computed over nonzero pixels).

    Returns
    -------
    u8 : np.ndarray
        uint8 image (H, W)
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


def find_spot_centers_smap(
    s_map: np.ndarray,
    percentile: float = 99.98,
    min_area: int = 10,
    max_area: Optional[int] = None,
    connect_radius: int = 2,
) -> list[tuple[float, float]]:
    """
    Find bright connected regions on S_map and return their centroid positions.

    Parameters
    ----------
    s_map : np.ndarray
        2D array (H, W) of the S_map values.
    percentile : float
        Percentile threshold used for binary masking.
    min_area : int
        Minimum connected component area in pixels.
    max_area : Optional[int]
        Maximum connected component area in pixels (None to disable).
    connect_radius : int
        Connection radius in pixels (2 -> 5x5 neighborhood).

    Returns
    -------
    centers : list of (x, y)
        Centroid coordinates for each detected spot.
    """
    if s_map.ndim != 2:
        raise ValueError(f"Expected 2D S_map, got shape {s_map.shape}")

    finite_mask = np.isfinite(s_map)
    if not finite_mask.any():
        return []

    thr = float(np.percentile(s_map[finite_mask], percentile))
    if not np.isfinite(thr):
        return []

    mask = s_map >= thr
    if not mask.any():
        return []

    h, w = mask.shape
    r = 1
    max_area_i = None if max_area is None else int(max_area)

    # Fast path: use OpenCV connected components (much faster than Python flood fill).
    if cv2 is not None:
        mask_u8 = mask.astype(np.uint8, copy=False)
        if r > 1:
            k = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
            work = cv2.dilate(mask_u8, k, iterations=1)
        else:
            work = mask_u8

        num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            work, connectivity=8
        )

        ys, xs = np.nonzero(mask_u8)
        if ys.size == 0:
            return []
        lab = labels[ys, xs]
        counts = np.bincount(lab, minlength=num)
        sum_x = np.bincount(lab, weights=xs.astype(np.float64), minlength=num)
        sum_y = np.bincount(lab, weights=ys.astype(np.float64), minlength=num)

        centers: list[tuple[float, float]] = []
        for lbl in range(1, num):
            cnt = int(counts[lbl])
            if cnt < min_area:
                continue
            if max_area_i is not None and cnt > max_area_i:
                continue
            centers.append((float(sum_x[lbl] / cnt), float(sum_y[lbl] / cnt)))
        return centers

    # Fallback: Python flood fill (slow for large masks).
    visited = np.zeros_like(mask, dtype=bool)
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    centers: list[tuple[float, float]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        sum_x = 0.0
        sum_y = 0.0
        count = 0

        while stack:
            cy, cx = stack.pop()
            sum_x += float(cx)
            sum_y += float(cy)
            count += 1
            for dy, dx in offsets:
                ny = cy + dy
                nx = cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if count < min_area:
            continue
        if max_area_i is not None and count > max_area_i:
            continue
        centers.append((sum_x / count, sum_y / count))

    return centers
