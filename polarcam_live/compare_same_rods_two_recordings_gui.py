from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


DEFAULT_RECORDING_A = Path(r"E:\11082026 DNA\well 2 single strange\frame_stack_20260811-212602.npy")
DEFAULT_RECORDING_B = Path(r"E:\11082026 DNA\well 2 single strange\frame_stack_20260811-230030.npy")

DISPLAY_CUTOUT_SIZE = 30
PREVIEW_FPS = 35
TRAJECTORY_WINDOW_RAW = 14
MAX_MEAN_FRAMES = 256
MAX_CANDIDATES = 300
SEED_BRIGHT_PCT_STRICT = 92.0
SEED_BRIGHT_PCT_RELAXED = 85.0
SEED_MIN_AREA = 2
EDGE_EXCLUDE_PX = 8
MATCH_SHIFT_X_PX = -10.0
MATCH_SHIFT_Y_PX = 10.0
MATCH_TOL_X_PX = 5.0
MATCH_TOL_Y_PX = 5.0
REFINE_RADIUS_PX = 8
MIN_SEP_PX = 10.0
THETA_RECON_LUT_STEP_DEG = 0.1
THETA_RECON_MODEL = {
    "label": "hole+fresnel",
    "a": 0.1865937176,
    "b": 0.5576753053,
    "c": 0.4215514426,
    "r_max": 0.9170101839,
}


@dataclass
class RecordingData:
    path: Path
    fps: float
    seq: np.ndarray
    mean_frame: np.ndarray
    centers_strict: list[tuple[float, float]]
    brightness_strict: list[float]
    centers_relaxed: list[tuple[float, float]]
    brightness_relaxed: list[float]
    frame_shape: tuple[int, int]
    frame_count: int


@dataclass
class RodPair:
    center_a: tuple[float, float]
    center_b: tuple[float, float]
    detected_a: bool
    detected_b: bool
    status_a: str
    status_b: str
    brightness: float


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if int(arr.shape[-1]) == 1:
            return np.asarray(arr[..., 0], dtype=np.float32)
        if int(arr.shape[-1]) in (3, 4):
            rgb = np.asarray(arr[..., :3], dtype=np.float32)
            return np.mean(rgb, axis=2, dtype=np.float32)
    raise RuntimeError(f"Unsupported frame shape for grayscale conversion: {arr.shape}")


def _strip_phase_marker_array(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.asarray(arr)
    roi_meta: dict = {}
    if out.ndim < 3 or int(out.shape[0]) < 2:
        return out, roi_meta
    try:
        marker = np.asarray(out[-1])
    except Exception:
        return out, roi_meta
    if marker.ndim != 2:
        return out, roi_meta
    try:
        nz = np.argwhere(marker != 0)
    except Exception:
        return out, roi_meta
    if nz.shape[0] != 1:
        return out, roi_meta
    my, mx = int(nz[0][0]), int(nz[0][1])
    try:
        mv = float(marker[my, mx])
        marker_sum = float(np.sum(marker, dtype=np.float64))
    except Exception:
        return out, roi_meta
    if mv != 1.0 or marker_sum != 1.0:
        return out, roi_meta
    roi_meta["phase_x"] = int(mx) % 2
    roi_meta["phase_y"] = int(my) % 2
    return np.asarray(out[:-1]), roi_meta


def _normalize_loaded_array(arr: np.ndarray) -> tuple[np.ndarray, int]:
    arr_use, _roi_meta = _strip_phase_marker_array(arr)
    if arr_use.dtype == object:
        raise RuntimeError("Object-array NPY files are not supported.")
    if arr_use.ndim == 2:
        return np.asarray(arr_use)[None, ...], 1
    if arr_use.ndim == 3:
        if int(arr_use.shape[-1]) in (1, 3, 4) and int(arr_use.shape[0]) > 4 and int(arr_use.shape[1]) > 4:
            return np.asarray(arr_use)[None, ...], 1
        return np.asarray(arr_use), int(arr_use.shape[0])
    if arr_use.ndim == 4:
        return np.asarray(arr_use), int(arr_use.shape[0])
    raise RuntimeError(f"Unsupported NPY shape: {arr_use.shape}")


def _sample_frame_indices(frame_count: int, max_samples: int) -> np.ndarray:
    if frame_count <= max_samples:
        return np.arange(frame_count, dtype=np.int64)
    idx = np.linspace(0, frame_count - 1, max_samples)
    idx = np.unique(np.round(idx).astype(np.int64))
    return idx


def _extract_window_even(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError("Expected 2D image for window extraction.")
    h, w = arr.shape
    half = size // 2
    x = int(round(float(cx)))
    y = int(round(float(cy)))
    x0 = x - half
    y0 = y - half
    x1 = x0 + size
    y1 = y0 + size
    out = np.zeros((size, size), dtype=arr.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def _local_brightness(img: np.ndarray, center: tuple[float, float], size: int = 7) -> float:
    win = _extract_window_even(np.asarray(img), center[0], center[1], size)
    vals = np.asarray(win, dtype=np.float32)
    if vals.size <= 0:
        return 0.0
    return float(np.mean(vals))


def _detect_candidate_centers(mean_frame: np.ndarray, bright_pct: float) -> tuple[list[tuple[float, float]], list[float]]:
    gray = np.asarray(mean_frame, dtype=np.float32)
    if gray.ndim != 2 or gray.size <= 0:
        return [], []
    h, w = gray.shape
    edge = int(EDGE_EXCLUDE_PX)

    work = gray.copy()
    if cv2 is not None:
        try:
            work = cv2.GaussianBlur(work, (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE)
        except Exception:
            pass

    try:
        thr = float(np.percentile(work, float(bright_pct)))
    except Exception:
        thr = float(np.max(work))
    mask = (work >= thr).astype(np.uint8)
    if cv2 is not None:
        try:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        except Exception:
            pass

    centers: list[tuple[float, float]] = []
    if cv2 is not None:
        try:
            n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        except Exception:
            n_labels = 0
            stats = None
            centroids = None
        if n_labels > 1 and stats is not None and centroids is not None:
            for i in range(1, int(n_labels)):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < int(SEED_MIN_AREA):
                    continue
                cx = float(centroids[i, 0])
                cy = float(centroids[i, 1])
                if edge > 0 and (cx < edge or cy < edge or cx > (w - 1 - edge) or cy > (h - 1 - edge)):
                    continue
                centers.append((cx, cy))

    if not centers:
        ys, xs = np.where(mask > 0)
        if xs.size > 0:
            vals = work[ys, xs].astype(np.float32)
            order = np.argsort(vals)[::-1]
            min_sep2 = float(MIN_SEP_PX * MIN_SEP_PX)
            for j in order:
                x = float(xs[j])
                y = float(ys[j])
                if edge > 0 and (x < edge or y < edge or x > (w - 1 - edge) or y > (h - 1 - edge)):
                    continue
                too_close = False
                for cx, cy in centers:
                    dx = cx - x
                    dy = cy - y
                    if (dx * dx) + (dy * dy) < min_sep2:
                        too_close = True
                        break
                if too_close:
                    continue
                centers.append((x, y))
                if len(centers) >= int(MAX_CANDIDATES):
                    break

    scored = [(_local_brightness(gray, c, size=7), c) for c in centers]
    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[: int(MAX_CANDIDATES)]
    return [c for _b, c in scored], [float(b) for b, _c in scored]


def _refine_center(mean_frame: np.ndarray, center: tuple[float, float], radius: int) -> tuple[float, float]:
    arr = np.asarray(mean_frame, dtype=np.float32)
    if arr.ndim != 2 or arr.size <= 0:
        return (float(center[0]), float(center[1]))
    h, w = arr.shape
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    best_x = cx
    best_y = cy
    best_score = -float("inf")
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            score = _local_brightness(arr, (x, y), size=7)
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y
    return (float(best_x), float(best_y))


def _theta_recon_lut() -> tuple[np.ndarray, np.ndarray, float]:
    a = float(THETA_RECON_MODEL["a"])
    b = float(THETA_RECON_MODEL["b"])
    c = float(THETA_RECON_MODEL["c"])
    r_max = float(THETA_RECON_MODEL["r_max"])
    theta_deg = np.arange(0.0, 90.0, float(THETA_RECON_LUT_STEP_DEG), dtype=np.float64)
    theta_deg = np.append(theta_deg, 90.0)
    theta_rad = np.radians(theta_deg)
    r_vals = np.full(theta_rad.shape, np.nan, dtype=np.float64)
    sin2 = np.sin(theta_rad) ** 2
    finite = np.isfinite(sin2)
    den = a + (c * sin2[finite])
    ok = np.isfinite(den) & (den > 0.0)
    r_tmp = np.full(den.shape, np.nan, dtype=np.float64)
    r_tmp[ok] = (b * sin2[finite][ok]) / den[ok]
    r_vals[np.where(finite)[0]] = r_tmp
    if r_vals.size > 0:
        r_vals[0] = 0.0
        r_vals[-1] = r_max
    return r_vals, theta_rad, r_max


THETA_R_LUT, THETA_RAD_LUT, THETA_R_MAX = _theta_recon_lut()


def _theta_from_r(r: np.ndarray) -> np.ndarray:
    arr = np.asarray(r, dtype=np.float64)
    out = np.interp(np.clip(arr, 0.0, float(THETA_R_MAX)), THETA_R_LUT, THETA_RAD_LUT, left=0.0, right=0.5 * np.pi)
    out[arr >= float(THETA_R_MAX)] = 0.5 * np.pi
    return np.degrees(out)


def _xy_phi_stats_from_channel_windows(
    a0: np.ndarray,
    a45: np.ndarray,
    a135: np.ndarray,
    a90: np.ndarray,
) -> tuple[float, float, float]:
    eps = 1e-6
    if a0.size <= 0 or a45.size <= 0 or a135.size <= 0 or a90.size <= 0:
        return (0.0, 0.0, 0.0)
    h = min(int(a0.shape[0]), int(a45.shape[0]), int(a135.shape[0]), int(a90.shape[0]))
    w = min(int(a0.shape[1]), int(a45.shape[1]), int(a135.shape[1]), int(a90.shape[1]))
    if h <= 0 or w <= 0:
        return (0.0, 0.0, 0.0)
    a0 = np.asarray(a0[:h, :w], dtype=np.float32)
    a45 = np.asarray(a45[:h, :w], dtype=np.float32)
    a135 = np.asarray(a135[:h, :w], dtype=np.float32)
    a90 = np.asarray(a90[:h, :w], dtype=np.float32)
    finite = np.isfinite(a0) & np.isfinite(a45) & np.isfinite(a135) & np.isfinite(a90)
    if not np.any(finite):
        return (0.0, 0.0, 0.0)
    m0 = float(np.mean(a0[finite]))
    m90 = float(np.mean(a90[finite]))
    m45 = float(np.mean(a45[finite]))
    m135 = float(np.mean(a135[finite]))
    x = (m0 - m90) / (m0 + m90 + eps)
    y = (m45 - m135) / (m45 + m135 + eps)
    phi = float(0.5 * np.arctan2(y, x))
    return (float(x), float(y), phi)


def _xy_phi_stats_from_raw_window(raw_win: np.ndarray, origin_x: int, origin_y: int) -> tuple[float, float, float]:
    g = np.asarray(raw_win)
    if g.ndim != 2:
        return (0.0, 0.0, 0.0)
    h, w = int(g.shape[0]), int(g.shape[1])
    if h < 2 or w < 2:
        return (0.0, 0.0, 0.0)
    gf = np.asarray(g, dtype=np.float32)
    px = int(origin_x) % 2
    py = int(origin_y) % 2
    i90 = gf[py::2, px::2]
    i45 = gf[py::2, (1 - px) :: 2]
    i135 = gf[(1 - py) :: 2, px::2]
    i0 = gf[(1 - py) :: 2, (1 - px) :: 2]
    return _xy_phi_stats_from_channel_windows(a0=i0, a45=i45, a135=i135, a90=i90)


def _load_sidecar_json(npy_path: Path) -> dict:
    sidecar = npy_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        obj = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_nested_float(meta: dict, *path: str) -> Optional[float]:
    obj: object = meta
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    try:
        val = float(obj)
    except Exception:
        return None
    return val if np.isfinite(val) and val > 0.0 else None


def _resolve_fps(meta: dict) -> float:
    for path in (
        ("actual", "fps"),
        ("requested", "fps"),
        ("fetch_frames_output", "actual_fps"),
        ("fetch_frames_output", "fps"),
    ):
        val = _read_nested_float(meta, *path)
        if val is not None:
            return float(val)
    return 1600.0


def _load_recording(path: Path) -> RecordingData:
    meta = _load_sidecar_json(path)
    fps = _resolve_fps(meta)
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    seq, frame_count = _normalize_loaded_array(arr)
    if frame_count <= 0:
        raise RuntimeError(f"{path.name}: no frames found.")

    idx = _sample_frame_indices(frame_count, MAX_MEAN_FRAMES)
    acc = None
    shape = None
    for i in idx:
        gray = _to_gray_float(seq[int(i)])
        if shape is None:
            shape = tuple(int(v) for v in gray.shape)
            acc = np.zeros(shape, dtype=np.float64)
        if tuple(gray.shape) != tuple(shape):
            continue
        acc += np.asarray(gray, dtype=np.float64)
    if acc is None or shape is None or idx.size <= 0:
        raise RuntimeError(f"{path.name}: could not build mean frame.")
    mean_frame = np.asarray(acc / float(idx.size), dtype=np.float32)
    centers_strict, brightness_strict = _detect_candidate_centers(mean_frame, bright_pct=float(SEED_BRIGHT_PCT_STRICT))
    centers_relaxed, brightness_relaxed = _detect_candidate_centers(mean_frame, bright_pct=float(SEED_BRIGHT_PCT_RELAXED))
    return RecordingData(
        path=path,
        fps=float(fps),
        seq=seq,
        mean_frame=mean_frame,
        centers_strict=centers_strict,
        brightness_strict=brightness_strict,
        centers_relaxed=centers_relaxed,
        brightness_relaxed=brightness_relaxed,
        frame_shape=tuple(shape),
        frame_count=int(frame_count),
    )


def _pair_rods(rec_a: RecordingData, rec_b: RecordingData) -> list[RodPair]:
    centers_a_strict = list(rec_a.centers_strict)
    centers_b_strict = list(rec_b.centers_strict)
    centers_a_relaxed = list(rec_a.centers_relaxed)
    centers_b_relaxed = list(rec_b.centers_relaxed)
    bright_a_strict = list(rec_a.brightness_strict)
    bright_b_strict = list(rec_b.brightness_strict)
    bright_a_relaxed = list(rec_a.brightness_relaxed)
    bright_b_relaxed = list(rec_b.brightness_relaxed)
    used_b_strict: set[int] = set()
    used_a_strict: set[int] = set()
    used_b_relaxed: set[int] = set()
    used_a_relaxed: set[int] = set()
    pairs: list[RodPair] = []
    def best_match(
        center: tuple[float, float],
        candidates: list[tuple[float, float]],
        used: set[int],
        shift_x: float,
        shift_y: float,
    ) -> Optional[int]:
        best_j = None
        best_d2 = None
        target_x = float(center[0]) + float(shift_x)
        target_y = float(center[1]) + float(shift_y)
        for j, cand in enumerate(candidates):
            if j in used:
                continue
            dx = target_x - float(cand[0])
            dy = target_y - float(cand[1])
            if abs(dx) > float(MATCH_TOL_X_PX) or abs(dy) > float(MATCH_TOL_Y_PX):
                continue
            d2 = (dx * dx) + (dy * dy)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j
        return best_j

    for i, center_a in enumerate(centers_a_strict):
        best_j = best_match(center_a, centers_b_strict, used_b_strict, MATCH_SHIFT_X_PX, MATCH_SHIFT_Y_PX)
        if best_j is None:
            continue
        used_a_strict.add(int(i))
        used_b_strict.add(int(best_j))
        center_a_ref = _refine_center(rec_a.mean_frame, center_a, REFINE_RADIUS_PX)
        center_b_ref = _refine_center(rec_b.mean_frame, centers_b_strict[int(best_j)], REFINE_RADIUS_PX)
        brightness = max(float(bright_a_strict[i]), float(bright_b_strict[int(best_j)]))
        pairs.append(
            RodPair(
                center_a=center_a_ref,
                center_b=center_b_ref,
                detected_a=True,
                detected_b=True,
                status_a="strict",
                status_b="strict",
                brightness=float(brightness),
            )
        )

    for i, center_a in enumerate(centers_a_strict):
        if i in used_a_strict:
            continue
        best_j = best_match(center_a, centers_b_relaxed, used_b_relaxed, MATCH_SHIFT_X_PX, MATCH_SHIFT_Y_PX)
        center_a_ref = _refine_center(rec_a.mean_frame, center_a, REFINE_RADIUS_PX)
        if best_j is not None:
            used_a_strict.add(int(i))
            used_b_relaxed.add(int(best_j))
            center_b_ref = _refine_center(rec_b.mean_frame, centers_b_relaxed[int(best_j)], REFINE_RADIUS_PX)
            brightness = max(float(bright_a_strict[i]), float(bright_b_relaxed[int(best_j)]))
            pairs.append(
                RodPair(
                    center_a=center_a_ref,
                    center_b=center_b_ref,
                    detected_a=True,
                    detected_b=True,
                    status_a="strict",
                    status_b="relaxed",
                    brightness=float(brightness),
                )
            )
            continue
        center_b_ref = _refine_center(
            rec_b.mean_frame,
            (float(center_a_ref[0]) + float(MATCH_SHIFT_X_PX), float(center_a_ref[1]) + float(MATCH_SHIFT_Y_PX)),
            REFINE_RADIUS_PX,
        )
        pairs.append(
            RodPair(
                center_a=center_a_ref,
                center_b=center_b_ref,
                detected_a=True,
                detected_b=False,
                status_a="strict",
                status_b="refined",
                brightness=float(bright_a_strict[i]),
            )
        )

    for j, center_b in enumerate(centers_b_strict):
        if j in used_b_strict:
            continue
        best_i = best_match(center_b, centers_a_relaxed, used_a_relaxed, -MATCH_SHIFT_X_PX, -MATCH_SHIFT_Y_PX)
        center_b_ref = _refine_center(rec_b.mean_frame, center_b, REFINE_RADIUS_PX)
        if best_i is not None:
            used_b_strict.add(int(j))
            used_a_relaxed.add(int(best_i))
            center_a_ref = _refine_center(rec_a.mean_frame, centers_a_relaxed[int(best_i)], REFINE_RADIUS_PX)
            brightness = max(float(bright_b_strict[j]), float(bright_a_relaxed[int(best_i)]))
            pairs.append(
                RodPair(
                    center_a=center_a_ref,
                    center_b=center_b_ref,
                    detected_a=True,
                    detected_b=True,
                    status_a="relaxed",
                    status_b="strict",
                    brightness=float(brightness),
                )
            )
            continue
        center_a_ref = _refine_center(
            rec_a.mean_frame,
            (float(center_b_ref[0]) - float(MATCH_SHIFT_X_PX), float(center_b_ref[1]) - float(MATCH_SHIFT_Y_PX)),
            REFINE_RADIUS_PX,
        )
        pairs.append(
            RodPair(
                center_a=center_a_ref,
                center_b=center_b_ref,
                detected_a=False,
                detected_b=True,
                status_a="refined",
                status_b="strict",
                brightness=float(bright_b_strict[j]),
            )
        )

    for j, center_b in enumerate(centers_b_relaxed):
        if j in used_b_relaxed:
            continue
        too_close = False
        for pair in pairs:
            dx = float(pair.center_b[0]) - float(center_b[0])
            dy = float(pair.center_b[1]) - float(center_b[1])
            if abs(dx) <= float(MATCH_TOL_X_PX) and abs(dy) <= float(MATCH_TOL_Y_PX):
                too_close = True
                break
        if too_close:
            continue
        best_i = best_match(center_b, centers_a_strict, used_a_strict, -MATCH_SHIFT_X_PX, -MATCH_SHIFT_Y_PX)
        if best_i is None:
            continue
        used_a_strict.add(int(best_i))
        used_b_relaxed.add(int(j))
        center_a_ref = _refine_center(rec_a.mean_frame, centers_a_strict[int(best_i)], REFINE_RADIUS_PX)
        center_b_ref = _refine_center(rec_b.mean_frame, center_b, REFINE_RADIUS_PX)
        brightness = max(float(bright_a_strict[int(best_i)]), float(bright_b_relaxed[j]))
        pairs.append(
            RodPair(
                center_a=center_a_ref,
                center_b=center_b_ref,
                detected_a=True,
                detected_b=True,
                status_a="strict",
                status_b="relaxed",
                brightness=float(brightness),
            )
        )

    for i, center_a in enumerate(centers_a_relaxed):
        if i in used_a_relaxed or i in used_a_strict:
            continue
        too_close = False
        for pair in pairs:
            dx = float(pair.center_a[0]) - float(center_a[0])
            dy = float(pair.center_a[1]) - float(center_a[1])
            if abs(dx) <= float(MATCH_TOL_X_PX) and abs(dy) <= float(MATCH_TOL_Y_PX):
                too_close = True
                break
        if too_close:
            continue
        best_j = best_match(center_a, centers_b_strict, used_b_strict, MATCH_SHIFT_X_PX, MATCH_SHIFT_Y_PX)
        if best_j is None:
            continue
        used_a_relaxed.add(int(i))
        used_b_strict.add(int(best_j))
        center_a_ref = _refine_center(rec_a.mean_frame, center_a, REFINE_RADIUS_PX)
        center_b_ref = _refine_center(rec_b.mean_frame, centers_b_strict[int(best_j)], REFINE_RADIUS_PX)
        brightness = max(float(bright_a_relaxed[i]), float(bright_b_strict[int(best_j)]))
        pairs.append(
            RodPair(
                center_a=center_a_ref,
                center_b=center_b_ref,
                detected_a=True,
                detected_b=True,
                status_a="relaxed",
                status_b="strict",
                brightness=float(brightness),
            )
        )

    pairs.sort(key=lambda pair: float(pair.brightness), reverse=True)
    deduped: list[RodPair] = []
    for pair in pairs:
        dup = False
        for prev in deduped:
            dx_a = float(prev.center_a[0]) - float(pair.center_a[0])
            dy_a = float(prev.center_a[1]) - float(pair.center_a[1])
            dx_b = float(prev.center_b[0]) - float(pair.center_b[0])
            dy_b = float(prev.center_b[1]) - float(pair.center_b[1])
            if ((dx_a * dx_a) + (dy_a * dy_a) <= 9.0) and ((dx_b * dx_b) + (dy_b * dy_b) <= 9.0):
                dup = True
                break
        if not dup:
            deduped.append(pair)
    return deduped


def _compute_series_for_centers(recording_path: Path, centers: list[tuple[float, float]]) -> list[dict[str, np.ndarray]]:
    arr = np.load(recording_path, mmap_mode="r", allow_pickle=True)
    seq, frame_count = _normalize_loaded_array(arr)
    gray0 = _to_gray_float(seq[0])
    h, w = (int(gray0.shape[0]), int(gray0.shape[1]))
    win_raw = int(TRAJECTORY_WINDOW_RAW)
    half = win_raw // 2
    bounds: list[tuple[int, int, int, int]] = []
    for cx, cy in centers:
        ix = int(round(float(cx)))
        iy = int(round(float(cy)))
        x0 = max(0, ix - half)
        y0 = max(0, iy - half)
        x1 = min(w, x0 + win_raw)
        y1 = min(h, y0 + win_raw)
        bounds.append((x0, x1, y0, y1))

    x_all = [np.zeros(frame_count, dtype=np.float32) for _ in centers]
    y_all = [np.zeros(frame_count, dtype=np.float32) for _ in centers]
    phi_all = [np.zeros(frame_count, dtype=np.float32) for _ in centers]

    for frame_i in range(frame_count):
        gray = _to_gray_float(seq[frame_i])
        for i, (x0, x1, y0, y1) in enumerate(bounds):
            raw_win = np.asarray(gray[y0:y1, x0:x1])
            x, y, phi = _xy_phi_stats_from_raw_window(raw_win, origin_x=int(x0), origin_y=int(y0))
            x_all[i][frame_i] = float(x)
            y_all[i][frame_i] = float(y)
            phi_all[i][frame_i] = float(phi)

    out: list[dict[str, np.ndarray]] = []
    for i in range(len(centers)):
        x = np.asarray(x_all[i], dtype=np.float64)
        y = np.asarray(y_all[i], dtype=np.float64)
        r = np.sqrt((x * x) + (y * y))
        phi_deg = np.degrees(np.unwrap(np.asarray(phi_all[i], dtype=np.float64)))
        theta_deg = _theta_from_r(r)
        out.append(
            {
                "x": x,
                "y": y,
                "r": r,
                "phi_deg": phi_deg,
                "theta_deg": theta_deg,
            }
        )
    return out


def _anisotropy_span_score(series: dict[str, np.ndarray]) -> float:
    x = np.asarray(series.get("x", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    y = np.asarray(series.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    if x.size <= 0 or y.size <= 0:
        return 0.0
    span_x = float(np.nanmax(x) - np.nanmin(x)) if np.any(np.isfinite(x)) else 0.0
    span_y = float(np.nanmax(y) - np.nanmin(y)) if np.any(np.isfinite(y)) else 0.0
    return float(math.hypot(span_x, span_y))


class CompareSameRodsApp:
    def __init__(self, root: tk.Tk, path_a: Path, path_b: Path) -> None:
        self.root = root
        self.root.title("Same Rods Comparison")
        self.root.geometry("1600x980")

        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.path_a_var = tk.StringVar(value=str(self.path_a))
        self.path_b_var = tk.StringVar(value=str(self.path_b))
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._load_token = 0

        self.rec_a: Optional[RecordingData] = None
        self.rec_b: Optional[RecordingData] = None
        self.pairs: list[RodPair] = []
        self.series_a: list[dict[str, np.ndarray]] = []
        self.series_b: list[dict[str, np.ndarray]] = []
        self.rod_idx = 0
        self.ready = False
        self.show_detected_both_var = tk.BooleanVar(value=False)
        self.manual_offsets: dict[tuple[int, int], tuple[int, int]] = {}
        self._preview_after_id = None
        self._preview_frame_idx = [0, 0]
        self._img_artist = [None, None]

        self.selection = [
            {"t0": 0.0, "t1": 0.0, "tmax": 0.0},
            {"t0": 0.0, "t1": 0.0, "tmax": 0.0},
        ]
        self._drag: Optional[dict[str, int | str]] = None

        file_row = ttk.Frame(root, padding=(8, 8, 8, 0))
        file_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(file_row, text="Choose Left .npy", command=lambda: self._choose_path(0)).pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.path_a_var, width=90).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 10))
        ttk.Button(file_row, text="Choose Right .npy", command=lambda: self._choose_path(1)).pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.path_b_var, width=90).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 10))
        self.reload_btn = ttk.Button(file_row, text="Load Pair", command=self._reload_from_paths)
        self.reload_btn.pack(side=tk.LEFT)

        controls = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls.pack(side=tk.TOP, fill=tk.X)
        self.prev_btn = ttk.Button(controls, text="<", command=self._prev_rod)
        self.prev_btn.pack(side=tk.LEFT)
        self.next_btn = ttk.Button(controls, text=">", command=self._next_rod)
        self.next_btn.pack(side=tk.LEFT, padx=(6, 12))
        self.reset_a_btn = ttk.Button(controls, text="Reset Left Duration", command=lambda: self._reset_selection(0))
        self.reset_a_btn.pack(side=tk.LEFT)
        self.reset_b_btn = ttk.Button(controls, text="Reset Right Duration", command=lambda: self._reset_selection(1))
        self.reset_b_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.detected_both_chk = ttk.Checkbutton(
            controls,
            text="Detected in both only",
            variable=self.show_detected_both_var,
            command=self._on_filter_toggle,
        )
        self.detected_both_chk.pack(side=tk.LEFT, padx=(12, 0))
        self.status_var = tk.StringVar(value="Loading recordings...")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        nudge_row = ttk.Frame(root, padding=(8, 6, 8, 0))
        nudge_row.pack(side=tk.TOP, fill=tk.X)
        self._build_nudge_controls(nudge_row, 0, "Left playback window")
        self._build_nudge_controls(nudge_row, 1, "Right playback window")

        self.fig = Figure(figsize=(14, 9), dpi=100)
        gs = self.fig.add_gridspec(4, 2, height_ratios=(1.0, 1.25, 1.0, 1.0))
        self.ax_img = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])]
        self.ax_xy = [self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])]
        self.ax_theta = [self.fig.add_subplot(gs[2, 0]), self.fig.add_subplot(gs[2, 1])]
        self.ax_phi = [self.fig.add_subplot(gs[3, 0]), self.fig.add_subplot(gs[3, 1])]

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)

        self._set_buttons_enabled(False)
        self._draw_loading()
        self._start_worker()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.reload_btn.configure(state=tk.NORMAL)
        buttons = [self.prev_btn, self.next_btn, self.reset_a_btn, self.reset_b_btn, self.detected_both_chk]
        buttons.extend(getattr(self, "_nudge_buttons", []))
        for btn in buttons:
            btn.configure(state=state)

    def _build_nudge_controls(self, parent: tk.Widget, col: int, title: str) -> None:
        if not hasattr(self, "_nudge_buttons"):
            self._nudge_buttons = []
        box = ttk.LabelFrame(parent, text=title, padding=(8, 4, 8, 6))
        box.pack(side=tk.LEFT, padx=(0, 12), anchor="n")
        ttk.Label(box, text="Move analysis window").grid(row=0, column=0, columnspan=3)
        up_btn = ttk.Button(box, text="Up", width=7, command=lambda c=col: self._nudge_current(c, 0, -1))
        up_btn.grid(row=1, column=1, pady=(4, 2))
        left_btn = ttk.Button(box, text="Left", width=7, command=lambda c=col: self._nudge_current(c, -1, 0))
        left_btn.grid(row=2, column=0, padx=(0, 4))
        reset_btn = ttk.Button(box, text="Reset", width=7, command=lambda c=col: self._reset_nudge_current(c))
        reset_btn.grid(row=2, column=1)
        right_btn = ttk.Button(box, text="Right", width=7, command=lambda c=col: self._nudge_current(c, 1, 0))
        right_btn.grid(row=2, column=2, padx=(4, 0))
        down_btn = ttk.Button(box, text="Down", width=7, command=lambda c=col: self._nudge_current(c, 0, 1))
        down_btn.grid(row=3, column=1, pady=(2, 0))
        self._nudge_buttons.extend([up_btn, left_btn, reset_btn, right_btn, down_btn])

    def _draw_loading(self) -> None:
        for ax in self.ax_img + self.ax_xy + self.ax_theta + self.ax_phi:
            ax.clear()
            ax.text(0.5, 0.5, "Loading...", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _start_worker(self) -> None:
        token = int(self._load_token)
        self._worker = threading.Thread(target=self._worker_main, args=(token,), daemon=True)
        self._worker.start()
        self.root.after(100, self._poll_worker)

    def _worker_main(self, token: int) -> None:
        try:
            rec_a = _load_recording(self.path_a)
            rec_b = _load_recording(self.path_b)
            pairs = _pair_rods(rec_a, rec_b)
            centers_a = [pair.center_a for pair in pairs]
            centers_b = [pair.center_b for pair in pairs]
            series_a = _compute_series_for_centers(rec_a.path, centers_a)
            series_b = _compute_series_for_centers(rec_b.path, centers_b)
            if pairs:
                order = sorted(
                    range(len(pairs)),
                    key=lambda i: _anisotropy_span_score(series_a[i]),
                    reverse=True,
                )
                pairs = [pairs[i] for i in order]
                series_a = [series_a[i] for i in order]
                series_b = [series_b[i] for i in order]
            self._queue.put(("ok", token, rec_a, rec_b, pairs, series_a, series_b))
        except Exception as exc:
            self._queue.put(("err", token, str(exc)))

    def _poll_worker(self) -> None:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return
        if int(item[1]) != int(self._load_token):
            self.root.after(100, self._poll_worker)
            return
        if item[0] == "err":
            messagebox.showerror("Same Rods Comparison", str(item[2]))
            self.status_var.set(str(item[2]))
            return
        _tag, _token, rec_a, rec_b, pairs, series_a, series_b = item
        self.rec_a = rec_a
        self.rec_b = rec_b
        self.pairs = list(pairs)
        self.series_a = list(series_a)
        self.series_b = list(series_b)
        self.ready = True
        self._set_buttons_enabled(True)
        if not self._visible_pair_indices():
            self.status_var.set("No rods detected.")
            self._render_empty()
            return
        self._reset_selection(0, render=False)
        self._reset_selection(1, render=False)
        self._render_current_rod()
        self._schedule_preview_tick()

    def _choose_path(self, side: int) -> None:
        initial = self.path_a if int(side) == 0 else self.path_b
        path = filedialog.askopenfilename(
            title="Choose NumPy recording",
            initialdir=str(initial.parent if initial.exists() else Path.cwd()),
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not path:
            return
        if int(side) == 0:
            self.path_a_var.set(str(Path(path)))
        else:
            self.path_b_var.set(str(Path(path)))

    def _stop_preview_loop(self) -> None:
        if self._preview_after_id is None:
            return
        try:
            self.root.after_cancel(self._preview_after_id)
        except Exception:
            pass
        self._preview_after_id = None

    def _reload_from_paths(self) -> None:
        path_a = Path(self.path_a_var.get().strip())
        path_b = Path(self.path_b_var.get().strip())
        if not path_a.exists():
            messagebox.showerror("Same Rods Comparison", f"Left file not found:\n{path_a}")
            return
        if not path_b.exists():
            messagebox.showerror("Same Rods Comparison", f"Right file not found:\n{path_b}")
            return
        self._stop_preview_loop()
        self._load_token += 1
        self.path_a = path_a
        self.path_b = path_b
        self.rec_a = None
        self.rec_b = None
        self.pairs = []
        self.series_a = []
        self.series_b = []
        self.manual_offsets = {}
        self.rod_idx = 0
        self.ready = False
        self._preview_frame_idx = [0, 0]
        self._img_artist = [None, None]
        self.selection = [
            {"t0": 0.0, "t1": 0.0, "tmax": 0.0},
            {"t0": 0.0, "t1": 0.0, "tmax": 0.0},
        ]
        self.status_var.set(f"Loading {path_a.name} vs {path_b.name} ...")
        self._set_buttons_enabled(False)
        self._draw_loading()
        self._start_worker()

    def _render_empty(self) -> None:
        for ax in self.ax_img + self.ax_xy + self.ax_theta + self.ax_phi:
            ax.clear()
            ax.text(0.5, 0.5, "No rods detected", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _current_pair(self) -> Optional[RodPair]:
        idx = self._current_global_idx()
        if idx is None:
            return None
        return self.pairs[idx]

    def _visible_pair_indices(self) -> list[int]:
        if not self.pairs:
            return []
        if not bool(self.show_detected_both_var.get()):
            return list(range(len(self.pairs)))
        return [i for i, pair in enumerate(self.pairs) if bool(pair.detected_a) and bool(pair.detected_b)]

    def _current_global_idx(self) -> Optional[int]:
        visible = self._visible_pair_indices()
        if not visible:
            return None
        self.rod_idx = max(0, min(int(self.rod_idx), len(visible) - 1))
        return int(visible[self.rod_idx])

    def _reset_selection(self, col: int, render: bool = True) -> None:
        if col not in (0, 1):
            return
        if not self.ready:
            self.selection[col] = {"t0": 0.0, "t1": 0.0, "tmax": 0.0}
        else:
            idx = self._current_global_idx()
            if idx is None:
                self.selection[col] = {"t0": 0.0, "t1": 0.0, "tmax": 0.0}
                return
            series = self.series_a[idx] if col == 0 else self.series_b[idx]
            fps = self.rec_a.fps if col == 0 else self.rec_b.fps
            n = int(np.asarray(series["x"]).size)
            tmax = float(max(0, n - 1)) / max(1e-9, float(fps))
            self.selection[col] = {"t0": 0.0, "t1": tmax, "tmax": tmax}
        if render and self.ready:
            self._render_current_rod()

    def _prev_rod(self) -> None:
        visible = self._visible_pair_indices()
        if not visible:
            return
        self.rod_idx = (int(self.rod_idx) - 1) % len(visible)
        self._reset_selection(0, render=False)
        self._reset_selection(1, render=False)
        self._preview_frame_idx = [0, 0]
        self._render_current_rod()

    def _next_rod(self) -> None:
        visible = self._visible_pair_indices()
        if not visible:
            return
        self.rod_idx = (int(self.rod_idx) + 1) % len(visible)
        self._reset_selection(0, render=False)
        self._reset_selection(1, render=False)
        self._preview_frame_idx = [0, 0]
        self._render_current_rod()

    def _on_filter_toggle(self) -> None:
        visible = self._visible_pair_indices()
        self.rod_idx = 0
        self._preview_frame_idx = [0, 0]
        if not visible:
            self.status_var.set("No rods match current filter.")
            self._render_empty()
            return
        if self.ready:
            self._reset_selection(0, render=False)
            self._reset_selection(1, render=False)
            self._render_current_rod()

    def _selection_text(self, col: int) -> str:
        sel = self.selection[col]
        return f"{float(sel['t0']):.3f} to {float(sel['t1']):.3f} s"

    def _current_offset(self, global_idx: int, col: int) -> tuple[int, int]:
        return tuple(self.manual_offsets.get((int(global_idx), int(col)), (0, 0)))

    def _adjusted_center(self, pair: RodPair, global_idx: int, col: int) -> tuple[float, float]:
        base = pair.center_a if int(col) == 0 else pair.center_b
        dx, dy = self._current_offset(global_idx, col)
        return (float(base[0]) + float(dx), float(base[1]) + float(dy))

    def _recompute_current_side(self, global_idx: int, col: int) -> None:
        pair = self.pairs[global_idx]
        center = self._adjusted_center(pair, global_idx, col)
        rec = self.rec_a if int(col) == 0 else self.rec_b
        if rec is None:
            return
        series = _compute_series_for_centers(rec.path, [center])[0]
        if int(col) == 0:
            self.series_a[global_idx] = series
        else:
            self.series_b[global_idx] = series

    def _nudge_current(self, col: int, dx: int, dy: int) -> None:
        if not self.ready:
            return
        global_idx = self._current_global_idx()
        if global_idx is None:
            return
        cur_dx, cur_dy = self._current_offset(global_idx, col)
        self.manual_offsets[(int(global_idx), int(col))] = (int(cur_dx + dx), int(cur_dy + dy))
        self._recompute_current_side(global_idx, col)
        self._preview_frame_idx[col] = 0
        self._render_current_rod()

    def _reset_nudge_current(self, col: int) -> None:
        if not self.ready:
            return
        global_idx = self._current_global_idx()
        if global_idx is None:
            return
        if (int(global_idx), int(col)) in self.manual_offsets:
            self.manual_offsets.pop((int(global_idx), int(col)), None)
            self._recompute_current_side(global_idx, col)
            self._preview_frame_idx[col] = 0
            self._render_current_rod()

    @staticmethod
    def _status_label(status: str) -> str:
        s = str(status).strip().lower()
        if s == "strict":
            return "detected strict"
        if s == "relaxed":
            return "detected relaxed"
        return "refined from other recording"

    def _render_current_rod(self) -> None:
        pair = self._current_pair()
        if pair is None or self.rec_a is None or self.rec_b is None:
            self._render_empty()
            return
        idx = self._current_global_idx()
        if idx is None:
            self._render_empty()
            return

        self.status_var.set(
            f"Rod {self.rod_idx + 1} / {len(self._visible_pair_indices())} | "
            f"left={self._status_label(pair.status_a)} | right={self._status_label(pair.status_b)} | "
            f"theta model {THETA_RECON_MODEL['label']} r_max={float(THETA_RECON_MODEL['r_max']):.3f}"
        )

        recs = [self.rec_a, self.rec_b]
        series_list = [self.series_a[idx], self.series_b[idx]]
        centers = [self._adjusted_center(pair, idx, 0), self._adjusted_center(pair, idx, 1)]
        statuses = [pair.status_a, pair.status_b]

        for col in (0, 1):
            rec = recs[col]
            series = series_list[col]
            center = centers[col]
            frame_idx = self._preview_frame_index_for_col(col)
            frame = _to_gray_float(rec.seq[frame_idx])
            frame_cut = _extract_window_even(frame, center[0], center[1], DISPLAY_CUTOUT_SIZE)

            ax = self.ax_img[col]
            ax.clear()
            self._img_artist[col] = ax.imshow(np.asarray(frame_cut, dtype=np.float32), cmap="gray", interpolation="nearest")
            ax.set_title(rec.path.name)
            ax.set_xticks([])
            ax.set_yticks([])
            det_label = self._status_label(statuses[col])
            off_x, off_y = self._current_offset(idx, col)
            ax.set_xlabel(
                f"center=({center[0]:.1f}, {center[1]:.1f}) px | offset=({off_x:+d}, {off_y:+d}) | "
                f"{det_label} | frame {frame_idx + 1}/{rec.frame_count}"
            )

            x = np.asarray(series["x"], dtype=np.float64)
            y = np.asarray(series["y"], dtype=np.float64)
            phi_deg = np.asarray(series["phi_deg"], dtype=np.float64)
            theta_deg = np.asarray(series["theta_deg"], dtype=np.float64)
            n = int(x.size)
            fps = float(rec.fps)
            t = np.arange(n, dtype=np.float64) / max(1e-9, fps)
            tmax = float(t[-1]) if n > 0 else 0.0
            self.selection[col]["tmax"] = tmax
            self.selection[col]["t0"] = max(0.0, min(float(self.selection[col]["t0"]), tmax))
            self.selection[col]["t1"] = max(0.0, min(float(self.selection[col]["t1"]), tmax))
            if float(self.selection[col]["t1"]) < float(self.selection[col]["t0"]):
                self.selection[col]["t0"], self.selection[col]["t1"] = self.selection[col]["t1"], self.selection[col]["t0"]

            ax_xy = self.ax_xy[col]
            ax_xy.clear()
            ax_xy.scatter(x, y, s=7, color="tab:blue", alpha=0.8)
            ax_xy.axhline(0.0, color="0.84", lw=0.9)
            ax_xy.axvline(0.0, color="0.84", lw=0.9)
            ax_xy.set_xlim(-1.0, 1.0)
            ax_xy.set_ylim(-1.0, 1.0)
            ax_xy.set_aspect("equal", adjustable="box")
            ax_xy.set_xlabel("X")
            ax_xy.set_ylabel("Y")
            ax_xy.set_title("XY plot")

            ax_theta = self.ax_theta[col]
            ax_theta.clear()
            ax_theta.plot(t, theta_deg, color="tab:red", lw=0.9)
            ax_theta.set_xlabel("time (s)")
            ax_theta.set_ylabel("theta (deg)")
            ax_theta.set_ylim(0.0, 90.0)
            ax_theta.set_title(f"theta(t) | selection {self._selection_text(col)}")
            ax_theta.grid(alpha=0.25)

            ax_phi = self.ax_phi[col]
            ax_phi.clear()
            ax_phi.plot(t, phi_deg, color="tab:green", lw=0.9)
            ax_phi.set_xlabel("time (s)")
            ax_phi.set_ylabel("phi (deg, unwrapped)")
            ax_phi.set_title("phi(t)")
            ax_phi.grid(alpha=0.25)

            self._draw_selection_overlay(col)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _preview_frame_index_for_col(self, col: int) -> int:
        rec = self.rec_a if col == 0 else self.rec_b
        if rec is None or int(rec.frame_count) <= 0:
            return 0
        return max(0, min(int(self._preview_frame_idx[col]), int(rec.frame_count) - 1))

    def _schedule_preview_tick(self) -> None:
        if self._preview_after_id is not None:
            return
        dt_ms = max(1, int(round(1000.0 / float(PREVIEW_FPS))))
        self._preview_after_id = self.root.after(dt_ms, self._preview_tick)

    def _preview_tick(self) -> None:
        self._preview_after_id = None
        if not self.ready or self.rec_a is None or self.rec_b is None:
            return
        idx = self._current_global_idx()
        if idx is None:
            self._schedule_preview_tick()
            return
        pair = self.pairs[idx]
        recs = [self.rec_a, self.rec_b]
        centers = [self._adjusted_center(pair, idx, 0), self._adjusted_center(pair, idx, 1)]
        changed = False
        for col in (0, 1):
            rec = recs[col]
            if int(rec.frame_count) <= 0:
                continue
            self._preview_frame_idx[col] = (int(self._preview_frame_idx[col]) + 1) % int(rec.frame_count)
            frame_idx = self._preview_frame_index_for_col(col)
            frame = _to_gray_float(rec.seq[frame_idx])
            cut = _extract_window_even(frame, centers[col][0], centers[col][1], DISPLAY_CUTOUT_SIZE)
            if self._img_artist[col] is not None:
                self._img_artist[col].set_data(np.asarray(cut, dtype=np.float32))
                off_x, off_y = self._current_offset(idx, col)
                self.ax_img[col].set_xlabel(
                    f"center=({centers[col][0]:.1f}, {centers[col][1]:.1f}) px | "
                    f"offset=({off_x:+d}, {off_y:+d}) | "
                    f"{self._status_label(pair.status_a if col == 0 else pair.status_b)} | "
                    f"frame {frame_idx + 1}/{rec.frame_count}"
                )
                changed = True
        if changed:
            self.canvas.draw_idle()
        self._schedule_preview_tick()

    def _draw_selection_overlay(self, col: int) -> None:
        sel = self.selection[col]
        t0 = float(sel["t0"])
        t1 = float(sel["t1"])
        lo = min(t0, t1)
        hi = max(t0, t1)
        for ax in (self.ax_theta[col], self.ax_phi[col]):
            ax.axvspan(lo, hi, color="tab:blue", alpha=0.07, zorder=0)
            ax.axvline(lo, color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axvline(hi, color="tab:blue", linestyle="--", linewidth=1.2)

    def _column_for_axes(self, ax) -> Optional[int]:
        for col in (0, 1):
            if ax is self.ax_theta[col] or ax is self.ax_phi[col]:
                return col
        return None

    def _pick_handle(self, col: int, x: float) -> str:
        sel = self.selection[col]
        tmax = max(1e-6, float(sel["tmax"]))
        tol = 0.02 * tmax
        d0 = abs(float(x) - float(sel["t0"]))
        d1 = abs(float(x) - float(sel["t1"]))
        if d0 <= tol and d0 <= d1:
            return "t0"
        if d1 <= tol:
            return "t1"
        return "t0" if d0 <= d1 else "t1"

    def _on_plot_press(self, event) -> None:
        if not self.ready or event.xdata is None:
            return
        col = self._column_for_axes(event.inaxes)
        if col is None:
            return
        handle = self._pick_handle(col, float(event.xdata))
        self._drag = {"col": int(col), "handle": handle}
        self._update_drag(float(event.xdata))

    def _on_plot_motion(self, event) -> None:
        if self._drag is None or event.xdata is None:
            return
        self._update_drag(float(event.xdata))

    def _on_plot_release(self, _event) -> None:
        self._drag = None

    def _update_drag(self, x: float) -> None:
        if self._drag is None:
            return
        col = int(self._drag["col"])
        handle = str(self._drag["handle"])
        tmax = float(self.selection[col]["tmax"])
        xv = max(0.0, min(float(x), tmax))
        self.selection[col][handle] = xv
        self._render_current_rod()


def main() -> None:
    root = tk.Tk()
    app = CompareSameRodsApp(root, DEFAULT_RECORDING_A, DEFAULT_RECORDING_B)
    root.mainloop()


if __name__ == "__main__":
    main()
