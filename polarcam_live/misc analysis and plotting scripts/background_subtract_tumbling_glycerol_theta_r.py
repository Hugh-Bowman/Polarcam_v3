from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


RANGE_THRESHOLD = 1.0
R_MAX_PHYSICAL = 1.0
PHI_BIN_DEG = 5.0
RNG_SEED_BALANCE = 12345
N_BOOT = 250
BOOT_SEED = 86420

OUTPUT_ROOT = Path("datasets") / "glycerol_tumbling_background_subtracted_20260715"
BACKGROUND_ROOT = Path("datasets") / "background characterisation"

DATASETS = [
    {
        "name": "25x65nm",
        "root": Path("datasets") / "tumbling 25nm glycerol",
        "color_original": "#2ca02c",
        "color_corrected": "#0b5d2a",
    },
    {
        "name": "40x65nm",
        "root": Path("glycerol suspended rods 17062026"),
        "color_original": "#d62728",
        "color_corrected": "#7a1020",
    },
]


@dataclass
class BackgroundProfile:
    path: Path
    timestamp: datetime
    frame: np.ndarray


@dataclass
class Recording:
    source: str
    rod: str
    x: np.ndarray
    y: np.ndarray
    n_points_total: int = 0
    n_points_r_gt_max: int = 0


def _timestamp_from_text(text: str) -> datetime | None:
    m = re.search(r"(20\d{6})[-_](\d{6})", text)
    if not m:
        return None
    try:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _load_background_profiles(root: Path) -> list[BackgroundProfile]:
    profiles: list[BackgroundProfile] = []
    for path in sorted(root.rglob("*average frame*.npy")):
        ts = _timestamp_from_text(path.name)
        if ts is None:
            ts = _timestamp_from_text(str(path.parent))
        if ts is None:
            continue
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)
        if arr.ndim != 2:
            continue
        profiles.append(BackgroundProfile(path=path, timestamp=ts, frame=np.asarray(arr, dtype=np.float32)))
    if not profiles:
        raise SystemExit(f"No timestamped average background profiles found under {root}")
    return sorted(profiles, key=lambda b: b.timestamp)


def _nearest_background(profiles: list[BackgroundProfile], ts: datetime) -> BackgroundProfile:
    return min(profiles, key=lambda b: abs((b.timestamp - ts).total_seconds()))


def _strip_phase_marker_frame(arr: np.ndarray, roi_meta: dict[str, Any]) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 3 or int(a.shape[0]) < 2:
        return a
    marker = np.asarray(a[-1])
    if marker.ndim != 2:
        return a
    nz = np.argwhere(marker != 0)
    if nz.shape[0] != 1:
        return a
    my, mx = int(nz[0][0]), int(nz[0][1])
    if float(marker[my, mx]) != 1.0:
        return a
    if float(np.sum(marker, dtype=np.float64)) != 1.0:
        return a
    roi_meta["phase_x"] = int(mx) % 2
    roi_meta["phase_y"] = int(my) % 2
    return np.asarray(a[:-1])


def _xy_phi_stats_from_channel_windows(
    a0: np.ndarray,
    a45: np.ndarray,
    a135: np.ndarray,
    a90: np.ndarray,
) -> tuple[float, float, float, float]:
    eps = 1e-6
    if a0.size <= 0 or a45.size <= 0 or a135.size <= 0 or a90.size <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    h = min(int(a0.shape[0]), int(a45.shape[0]), int(a135.shape[0]), int(a90.shape[0]))
    w = min(int(a0.shape[1]), int(a45.shape[1]), int(a135.shape[1]), int(a90.shape[1]))
    if h <= 0 or w <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    a0 = np.asarray(a0[:h, :w], dtype=np.float32)
    a45 = np.asarray(a45[:h, :w], dtype=np.float32)
    a135 = np.asarray(a135[:h, :w], dtype=np.float32)
    a90 = np.asarray(a90[:h, :w], dtype=np.float32)
    finite = np.isfinite(a0) & np.isfinite(a45) & np.isfinite(a135) & np.isfinite(a90)
    pass_fraction = float(np.mean(finite)) if finite.size else 0.0
    if not np.any(finite):
        return (0.0, 0.0, 0.0, pass_fraction)
    m0 = float(np.mean(a0[finite]))
    m90 = float(np.mean(a90[finite]))
    m45 = float(np.mean(a45[finite]))
    m135 = float(np.mean(a135[finite]))
    x = (m0 - m90) / (m0 + m90 + eps)
    y = (m45 - m135) / (m45 + m135 + eps)
    phi = float(0.5 * np.arctan2(y, x))
    return (float(x), float(y), phi, pass_fraction)


def _xy_phi_stats_from_frame(gray: np.ndarray, roi_meta: dict[str, Any], win_raw: int) -> tuple[float, float, float, float]:
    g = np.asarray(gray)
    if g.ndim != 2:
        g = g[..., 0]
    gh, gw = int(g.shape[0]), int(g.shape[1])
    win_raw = max(2, int(win_raw))
    if (win_raw % 2) != 0:
        win_raw -= 1

    cx = (gw - 1) / 2.0
    cy = (gh - 1) / 2.0
    try:
        cx = float(roi_meta["cx"]) - float(roi_meta["x"])
        cy = float(roi_meta["cy"]) - float(roi_meta["y"])
    except Exception:
        pass

    x0 = int(round(cx)) - (win_raw // 2)
    y0 = int(round(cy)) - (win_raw // 2)
    x1 = x0 + win_raw
    y1 = y0 + win_raw
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(gw, x1)
    y1 = min(gh, y1)
    raw_win = np.asarray(g[y0:y1, x0:x1], dtype=np.float32)
    if raw_win.ndim != 2 or raw_win.shape[0] < 2 or raw_win.shape[1] < 2:
        return (0.0, 0.0, 0.0, 0.0)

    px = int(x0) % 2
    py = int(y0) % 2
    i90 = raw_win[py::2, px::2]
    i45 = raw_win[py::2, (1 - px) :: 2]
    i135 = raw_win[(1 - py) :: 2, px::2]
    i0 = raw_win[(1 - py) :: 2, (1 - px) :: 2]
    return _xy_phi_stats_from_channel_windows(a0=i0, a45=i45, a135=i135, a90=i90)


def _series_metrics(xy_series: list[tuple[float, float]]) -> dict[str, float | int]:
    if not xy_series:
        return {
            "n_frames": 0,
            "range_x": 0.0,
            "range_y": 0.0,
            "motion_max_axis_range": 0.0,
            "r_mean": 0.0,
            "r_std": 0.0,
            "r_min": 0.0,
            "r_max": 0.0,
        }
    arr = np.asarray(xy_series, dtype=np.float32)
    x = arr[:, 0]
    y = arr[:, 1]
    r = np.sqrt((x * x) + (y * y))
    range_x = float(np.max(x) - np.min(x))
    range_y = float(np.max(y) - np.min(y))
    return {
        "n_frames": int(arr.shape[0]),
        "range_x": range_x,
        "range_y": range_y,
        "motion_max_axis_range": float(max(range_x, range_y)),
        "r_mean": float(np.mean(r)),
        "r_std": float(np.std(r)),
        "r_min": float(np.min(r)),
        "r_max": float(np.max(r)),
    }


def _background_crop(bg: BackgroundProfile, roi_meta: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    y = int(round(float(roi_meta["y"])))
    x = int(round(float(roi_meta["x"])))
    h, w = int(shape[0]), int(shape[1])
    crop = bg.frame[y : y + h, x : x + w]
    if crop.shape != (h, w):
        raise ValueError(f"background crop shape {crop.shape} does not match recording frame {(h, w)}")
    return np.asarray(crop, dtype=np.float32)


def _correct_recording(
    meta_path: Path,
    source_root: Path,
    derived_root: Path,
    profiles: list[BackgroundProfile],
) -> dict[str, Any] | None:
    src_dir = meta_path.parent
    rel_dir = src_dir.relative_to(source_root)
    out_dir = derived_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    original_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npy_name = str(original_meta.get("npy_file") or "capture_maxfps_15x15.npy")
    npy_path = src_dir / npy_name
    if not npy_path.exists():
        return None
    roi_meta = dict((original_meta.get("actual") or {}).get("roi") or {})
    if not all(k in roi_meta for k in ("x", "y", "w", "h", "cx", "cy")):
        return None
    if not original_meta.get("center_px"):
        return None

    ts = _timestamp_from_text(src_dir.name) or _timestamp_from_text(npy_path.name)
    if ts is None:
        return None
    bg = _nearest_background(profiles, ts)

    arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    roi_meta = dict(roi_meta)
    arr_use = _strip_phase_marker_frame(arr, roi_meta=roi_meta)
    if arr_use.ndim < 3:
        return None
    frame_shape = (int(arr_use.shape[1]), int(arr_use.shape[2]))
    crop = _background_crop(bg, roi_meta, frame_shape)
    win_raw = int(roi_meta.get("win_raw", int(roi_meta.get("w", frame_shape[1]))))

    xy_series: list[tuple[float, float]] = []
    phi_series: list[float] = []
    pass_fraction_series: list[float] = []
    min_after = math.inf
    max_after = -math.inf
    min_before_clip = math.inf
    negative_before_clip = 0
    total_pixels = 0
    for i in range(int(arr_use.shape[0])):
        delta = np.asarray(arr_use[i], dtype=np.float32) - crop
        if delta.size:
            min_before_clip = min(min_before_clip, float(np.min(delta)))
            negative_before_clip += int(np.sum(delta < 0.0))
            total_pixels += int(delta.size)
        corrected = np.clip(delta, 0.0, None)
        if corrected.size:
            min_after = min(min_after, float(np.min(corrected)))
            max_after = max(max_after, float(np.max(corrected)))
        xv, yv, phi, pass_fraction = _xy_phi_stats_from_frame(corrected, roi_meta=roi_meta, win_raw=win_raw)
        xy_series.append((float(xv), float(yv)))
        phi_series.append(float(phi))
        pass_fraction_series.append(float(pass_fraction))

    corrected_meta = dict(original_meta)
    corrected_meta["xy_series"] = [[float(a), float(b)] for a, b in xy_series]
    corrected_meta["phi_series"] = [float(v) for v in phi_series]
    corrected_meta["pass_fraction_series"] = [float(v) for v in pass_fraction_series]
    corrected_meta["xy_metrics"] = _series_metrics(xy_series)
    corrected_meta["background_subtraction"] = {
        "performed": True,
        "method": "after-the-fact ROI crop subtraction from nearest timestamped average background, clipped at zero",
        "background_profile": str(bg.path.resolve()),
        "background_timestamp": bg.timestamp.isoformat(sep=" "),
        "recording_timestamp": ts.isoformat(sep=" "),
        "absolute_time_delta_s": abs((bg.timestamp - ts).total_seconds()),
        "source_meta": str(meta_path.resolve()),
        "source_npy": str(npy_path.resolve()),
        "corrected_stack_saved": False,
        "min_pixel_before_zero_clip": None if min_before_clip is math.inf else float(min_before_clip),
        "negative_pixel_fraction_before_zero_clip": (
            float(negative_before_clip / total_pixels) if total_pixels > 0 else None
        ),
        "min_corrected_pixel": None if min_after is math.inf else float(min_after),
        "max_corrected_pixel": None if max_after == -math.inf else float(max_after),
    }
    corrected_meta["npy_file"] = npy_name

    shutil.copy2(meta_path, out_dir / "capture_maxfps_15x15_meta_original.json")
    src_top_meta = src_dir / "meta.json"
    if src_top_meta.exists():
        shutil.copy2(src_top_meta, out_dir / "meta.json")
    (out_dir / "source_recording.json").write_text(
        json.dumps(
            {
                "source_folder": str(src_dir.resolve()),
                "source_npy": str(npy_path.resolve()),
                "source_meta": str(meta_path.resolve()),
                "background_profile": str(bg.path.resolve()),
                "corrected_stack_saved": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "capture_maxfps_15x15_meta.json").write_text(json.dumps(corrected_meta, indent=2), encoding="utf-8")

    return {
        "source": rel_dir.parts[0] if rel_dir.parts else "",
        "rod": src_dir.name,
        "source_meta": str(meta_path),
        "derived_meta": str(out_dir / "capture_maxfps_15x15_meta.json"),
        "background_profile": str(bg.path),
        "recording_timestamp": ts.isoformat(sep=" "),
        "background_timestamp": bg.timestamp.isoformat(sep=" "),
        "absolute_time_delta_s": abs((bg.timestamp - ts).total_seconds()),
        "original_range_x": float((original_meta.get("xy_metrics") or {}).get("range_x", 0.0)),
        "original_range_y": float((original_meta.get("xy_metrics") or {}).get("range_y", 0.0)),
        "corrected_range_x": float(corrected_meta["xy_metrics"]["range_x"]),
        "corrected_range_y": float(corrected_meta["xy_metrics"]["range_y"]),
        "original_r_mean": float((original_meta.get("xy_metrics") or {}).get("r_mean", 0.0)),
        "corrected_r_mean": float(corrected_meta["xy_metrics"]["r_mean"]),
    }


def _phi_deg_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.degrees(np.mod(0.5 * np.arctan2(y, x), np.pi))


def _gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    sigma = float(max(0.0, sigma_bins))
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(4.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _gaussian_smooth_hist(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_bins)
    if kernel.size == 1:
        return counts.copy()
    return np.convolve(counts, kernel, mode="same")


def _fit_uniform_costheta_curve(
    r_values: np.ndarray,
    bins: int = 160,
    sigma_bins: float = 3.0,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_values, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("No finite r values to fit.")
    lo = float(np.percentile(r, lo_pct))
    hi = float(np.percentile(r, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid r fit range.")
    counts, edges = np.histogram(r, bins=int(bins), range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    smooth_density = _gaussian_smooth_hist(counts, sigma_bins=sigma_bins)
    smooth_density = np.maximum(smooth_density, 0.0)
    area = float(np.sum(smooth_density * widths))
    if area > 0.0:
        smooth_density = smooth_density / area
    cdf = np.cumsum(smooth_density * widths)
    cdf = np.clip(cdf, 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    theta_deg = np.maximum.accumulate(theta_deg)
    return centers, theta_deg


def _interp_monotonic_theta(r_centers: np.ndarray, theta_deg: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    if PchipInterpolator is not None:
        fn = PchipInterpolator(r_centers, theta_deg, extrapolate=True)
        out = fn(r_grid)
    else:
        out = np.interp(r_grid, r_centers, theta_deg, left=theta_deg[0], right=theta_deg[-1])
    out = np.maximum.accumulate(np.asarray(out, dtype=np.float64))
    return np.clip(out, 0.0, 90.0)


def _load_recordings(root: Path) -> list[Recording]:
    out: list[Recording] = []
    for source in ("pending", "good", "bad"):
        src_dir = root / source
        if not src_dir.exists():
            continue
        for rod_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            xy_series = payload.get("xy_series")
            if not isinstance(xy_series, list) or not xy_series:
                continue
            xy = np.asarray(xy_series, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            xy = xy[:, :2]
            valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[valid]
            n_total = int(xy.shape[0])
            r_all = np.hypot(xy[:, 0], xy[:, 1]) if n_total > 0 else np.asarray([], dtype=np.float64)
            r_ok = r_all <= float(R_MAX_PHYSICAL)
            n_r_gt_max = int(np.sum(~r_ok)) if r_ok.size else 0
            xy = xy[r_ok]
            if xy.size == 0:
                continue
            range_x = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
            range_y = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
            if not (range_x > RANGE_THRESHOLD and range_y > RANGE_THRESHOLD):
                continue
            out.append(
                Recording(
                    source=source,
                    rod=rod_dir.name,
                    x=xy[:, 0].copy(),
                    y=xy[:, 1].copy(),
                    n_points_total=n_total,
                    n_points_r_gt_max=n_r_gt_max,
                )
            )
    return out


def _sample_balanced_r_from_recordings(recordings: list[Recording], rng_seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.concatenate([rec.x for rec in recordings])
    y = np.concatenate([rec.y for rec in recordings])
    phi_deg = _phi_deg_from_xy(x, y)
    r = np.sqrt((x * x) + (y * y))
    edges = np.arange(0.0, 180.0 + PHI_BIN_DEG, PHI_BIN_DEG, dtype=np.float64)
    bin_ids = np.digitize(phi_deg, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    idx_by_bin = [np.flatnonzero(bin_ids == i) for i in range(len(edges) - 1)]
    counts = [int(idx.size) for idx in idx_by_bin]
    min_count = int(min(counts))
    n_per_bin = max(1, min_count // 2)
    rng = np.random.default_rng(int(rng_seed))
    sampled_idx = [np.asarray(rng.choice(idx, size=n_per_bin, replace=False), dtype=np.int64) for idx in idx_by_bin]
    sampled_idx_arr = np.concatenate(sampled_idx)
    return r[sampled_idx_arr], {
        "n_recordings": int(len(recordings)),
        "n_pooled_points": int(r.size),
        "r_max_physical_filter": float(R_MAX_PHYSICAL),
        "n_points_removed_r_gt_max": int(sum(rec.n_points_r_gt_max for rec in recordings)),
        "n_points_before_r_filter": int(sum(rec.n_points_total for rec in recordings)),
        "least_populated_bin_count": int(min_count),
        "sampled_per_bin": int(n_per_bin),
        "n_selected_points": int(sampled_idx_arr.size),
    }


def _recording_bootstrap_curve(recordings: list[Recording]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    r_sample, sample_meta = _sample_balanced_r_from_recordings(recordings, rng_seed=RNG_SEED_BALANCE)
    center_r, center_theta = _fit_uniform_costheta_curve(r_sample)
    r_grid = np.linspace(float(center_r[0]), float(center_r[-1]), 600)
    theta_center = _interp_monotonic_theta(center_r, center_theta, r_grid)

    rng = np.random.default_rng(int(BOOT_SEED))
    curves: list[np.ndarray] = []
    n_rec = len(recordings)
    for boot_i in range(int(N_BOOT)):
        idx = rng.integers(0, n_rec, size=n_rec)
        sampled_recs = [recordings[int(i)] for i in idx]
        try:
            r_boot, _meta = _sample_balanced_r_from_recordings(sampled_recs, rng_seed=RNG_SEED_BALANCE + boot_i + 1)
            rb, tb = _fit_uniform_costheta_curve(r_boot)
            curves.append(_interp_monotonic_theta(rb, tb, r_grid))
        except Exception:
            continue
    if not curves:
        raise RuntimeError("Recording bootstrap failed.")
    curves_arr = np.asarray(curves, dtype=np.float64)
    theta_lo = np.percentile(curves_arr, 16.0, axis=0)
    theta_hi = np.percentile(curves_arr, 84.0, axis=0)
    theta_std = np.std(curves_arr, axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros_like(theta_center)
    meta = {**sample_meta, "n_boot_completed": int(curves_arr.shape[0])}
    return r_grid, theta_center, theta_lo, theta_hi, theta_std, r_sample, meta


def _write_curve_csv(path: Path, r_grid: np.ndarray, center: np.ndarray, lo: np.ndarray, hi: np.ndarray, std: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["r", "theta_deg_center", "theta_deg_lo_1sigma", "theta_deg_hi_1sigma", "theta_deg_std"],
        )
        writer.writeheader()
        for rv, cv, lv, hv, sv in zip(r_grid, center, lo, hi, std):
            writer.writerow(
                {
                    "r": float(rv),
                    "theta_deg_center": float(cv),
                    "theta_deg_lo_1sigma": float(lv),
                    "theta_deg_hi_1sigma": float(hv),
                    "theta_deg_std": float(sv),
                }
            )


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_theta_compare(path: Path, dataset_name: str, original: dict[str, Any], corrected: dict[str, Any], colors: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(original["r"], original["lo"], original["hi"], color=colors["color_original"], alpha=0.16, label="original 1 sigma")
    ax.plot(original["r"], original["center"], color=colors["color_original"], lw=2.1, label="original")
    ax.fill_between(corrected["r"], corrected["lo"], corrected["hi"], color=colors["color_corrected"], alpha=0.22, label="background-subtracted 1 sigma")
    ax.plot(corrected["r"], corrected["center"], color=colors["color_corrected"], lw=2.3, label="background-subtracted")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"Theta(r), {dataset_name}: original vs background-subtracted")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_r_density_compare(path: Path, dataset_name: str, original_r: np.ndarray, corrected_r: np.ndarray, colors: dict[str, str]) -> None:
    hi = max(float(np.percentile(original_r, 99.5)), float(np.percentile(corrected_r, 99.5)))
    hi = min(1.25, max(0.05, hi))
    bins = np.linspace(0.0, hi, 90)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(original_r, bins=bins, density=True, histtype="step", lw=2.0, color=colors["color_original"], label="original")
    ax.hist(corrected_r, bins=bins, density=True, histtype="step", lw=2.0, color=colors["color_corrected"], label="background-subtracted")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.set_title(f"r density, {dataset_name}: balanced sampled points")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _analyze_dataset(root: Path) -> tuple[dict[str, Any], list[Recording]]:
    recordings = _load_recordings(root)
    if not recordings:
        raise RuntimeError(f"No recordings passed filters under {root}")
    r_grid, center, lo, hi, std, r_sample, meta = _recording_bootstrap_curve(recordings)
    return {
        "r": r_grid,
        "center": center,
        "lo": lo,
        "hi": hi,
        "std": std,
        "r_sample": r_sample,
        "meta": meta,
    }, recordings


def main() -> None:
    profiles = _load_background_profiles(Path.cwd() / BACKGROUND_ROOT)
    out_root = Path.cwd() / OUTPUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "output_root": str(out_root.resolve()),
        "background_profiles": [{"path": str(p.path.resolve()), "timestamp": p.timestamp.isoformat(sep=" ")} for p in profiles],
        "datasets": {},
    }

    for cfg in DATASETS:
        name = str(cfg["name"])
        source_root = Path.cwd() / cfg["root"]
        derived_root = out_root / name / "background_subtracted"
        original_subset_root = out_root / name / "original_metadata_subset"
        plots_dir = out_root / name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        for meta_path in sorted(source_root.rglob("capture_maxfps_15x15_meta.json")):
            try:
                row = _correct_recording(meta_path, source_root=source_root, derived_root=derived_root, profiles=profiles)
            except Exception as e:
                skipped.append(f"{meta_path}: {e}")
                continue
            if row is None:
                skipped.append(f"{meta_path}: missing usable center/ROI/raw stack metadata")
                continue
            rows.append(row)

            rel_dir = meta_path.parent.relative_to(source_root)
            out_dir = original_subset_root / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_path, out_dir / "capture_maxfps_15x15_meta.json")
            top_meta = meta_path.parent / "meta.json"
            if top_meta.exists():
                shutil.copy2(top_meta, out_dir / "meta.json")

        _write_manifest_csv(plots_dir / "background_subtraction_manifest.csv", rows)
        (plots_dir / "background_subtraction_skipped.txt").write_text("\n".join(skipped), encoding="utf-8")

        original_analysis, original_recordings = _analyze_dataset(original_subset_root)
        corrected_analysis, corrected_recordings = _analyze_dataset(derived_root)

        _write_curve_csv(
            plots_dir / f"{name}_theta_r_original.csv",
            original_analysis["r"],
            original_analysis["center"],
            original_analysis["lo"],
            original_analysis["hi"],
            original_analysis["std"],
        )
        _write_curve_csv(
            plots_dir / f"{name}_theta_r_background_subtracted.csv",
            corrected_analysis["r"],
            corrected_analysis["center"],
            corrected_analysis["lo"],
            corrected_analysis["hi"],
            corrected_analysis["std"],
        )
        _plot_theta_compare(
            plots_dir / f"{name}_theta_r_original_vs_background_subtracted.png",
            name,
            original_analysis,
            corrected_analysis,
            cfg,
        )
        _plot_r_density_compare(
            plots_dir / f"{name}_r_density_original_vs_background_subtracted.png",
            name,
            original_analysis["r_sample"],
            corrected_analysis["r_sample"],
            cfg,
        )

        summary["datasets"][name] = {
            "source_root": str(source_root.resolve()),
            "derived_background_subtracted_root": str(derived_root.resolve()),
            "original_metadata_subset_root": str(original_subset_root.resolve()),
            "plots_dir": str(plots_dir.resolve()),
            "recordings_with_center_metadata": int(len(rows)),
            "skipped_recordings": int(len(skipped)),
            "original_recordings_passing_filter": int(len(original_recordings)),
            "background_subtracted_recordings_passing_filter": int(len(corrected_recordings)),
            "original_analysis": original_analysis["meta"],
            "background_subtracted_analysis": corrected_analysis["meta"],
            "max_background_time_delta_hours": float(max((r["absolute_time_delta_s"] for r in rows), default=0.0) / 3600.0),
            "median_background_time_delta_hours": float(np.median([r["absolute_time_delta_s"] for r in rows]) / 3600.0) if rows else None,
        }

    (out_root / "background_subtracted_theta_r_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
