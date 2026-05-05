#!/usr/bin/env python3
"""
characterise_illumination.py

Build an illumination profile from a bright-field AVI by:
1) computing per-pixel median over time,
2) smoothing with a large Gaussian kernel,
3) normalizing to [0, 1].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# Convenience: if you run this file directly in an IDE without CLI args.
INPUT_STACK_PATH = r"C:\Polarcam Software\Polarcam_v3\frame_runs\frame_stack_20260413-154339.npy"
# Per-pixel statistic across time (50=median). Requested default: 75th centile.
PROFILE_CENTILE = 95.0
# Apply dark subtraction before X/Y map calculation: frame <- max(frame - scale*dark, 0).
DARK_BMP_PATH = r"C:\IDS recordings\Background characterisation\coverslip_2ms_130426.bmp"
XY_DARK_SCALE = 0.01
# Fit gamma for bright - gamma*dark by minimizing X/Y map spread.
FIT_XY_GAMMA = True
GAMMA_MIN = 0.0
GAMMA_MAX = 0.05

def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Got empty frame")
    if frame.ndim == 2:
        g = frame
    elif frame.ndim == 3:
        if frame.shape[2] == 3:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            g = frame[..., 0]
    else:
        raise ValueError(f"Unsupported frame ndim={frame.ndim}")

    if g.dtype == np.uint8:
        return np.ascontiguousarray(g)
    if np.issubdtype(g.dtype, np.integer):
        # Best-effort downscale; many cameras store 10/12-bit in uint16.
        maxv = int(g.max()) if g.size else 0
        if maxv <= 255:
            out = g.astype(np.uint8, copy=False)
        elif maxv <= 4095:
            out = (g.astype(np.uint16, copy=False) >> 4).astype(np.uint8, copy=False)
        else:
            out = (g.astype(np.uint32, copy=False) >> 8).astype(np.uint8, copy=False)
        return np.ascontiguousarray(out)

    # float
    maxv = float(np.nanmax(g)) if g.size else 0.0
    if not np.isfinite(maxv) or maxv <= 0.0:
        return np.zeros_like(g, dtype=np.uint8)
    if maxv <= 1.0:
        y = g * 255.0
    elif maxv <= 255.0:
        y = g
    else:
        y = (g / maxv) * 255.0
    return np.ascontiguousarray(np.clip(y, 0.0, 255.0).astype(np.uint8))


def _to_u8_percentile(img: np.ndarray, lo_pct: float, hi_pct: float, gamma: float = 1.0) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    vals = x[finite]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    if gamma and gamma != 1.0:
        y = np.power(y, float(gamma))
    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    u8[~finite] = 0
    return u8


def _to_u8_log_stretch(img: np.ndarray, lo_pct: float = 0.0, hi_pct: float = 99.9) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    vals = x[finite]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip(x - lo, 0.0, hi - lo)
    y = np.log1p(y)
    denom = float(np.log1p(hi - lo))
    if denom <= 0.0:
        return np.zeros_like(x, dtype=np.uint8)
    y = y / denom
    u8 = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    u8[~finite] = 0
    return u8


def _open_avi(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open AVI: {path}")
    return cap


def _copy_trim_memmap(src_path: Path, dst_path: Path, n: int) -> None:
    src = np.load(src_path, mmap_mode="r")
    if src.ndim != 3:
        raise ValueError(f"Expected (N,H,W) in {src_path}, got {src.shape}")
    if n <= 0:
        raise ValueError("n must be > 0 for trimming")
    n = min(int(n), int(src.shape[0]))
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=(n, src.shape[1], src.shape[2]))
    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        dst[i0:i1] = src[i0:i1]
    dst.flush()


def convert_avi_to_npy(
    avi_path: Path, out_dir: Path, save_frames: bool = True
) -> Tuple[Optional[Path], tuple[int, int]]:
    cap = _open_avi(avi_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame.")
    gray0 = _to_gray_u8(frame0)
    h, w = gray0.shape

    npy_path = None
    tmp_path = None
    frames_mm = None
    if save_frames:
        npy_path = out_dir / f"{avi_path.stem}_frames.npy"
        if frame_count > 0:
            tmp_path = out_dir / f"{avi_path.stem}_frames_tmp.npy"
            frames_mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(frame_count, h, w))
            frames_mm[0] = gray0
        else:
            # Unknown count; fall back to list (may be large).
            frames_list = [gray0]
    else:
        frames_list = None

    i = 1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = _to_gray_u8(frame)

        if save_frames:
            if frames_mm is not None:
                if i >= frames_mm.shape[0]:
                    # CAP_PROP_FRAME_COUNT was wrong; stop writing.
                    break
                frames_mm[i] = gray
            else:
                frames_list.append(gray)
        i += 1

    cap.release()
    actual_n = i

    if save_frames:
        if frames_mm is not None:
            frames_mm.flush()
            del frames_mm
            frames_mm = None
            if actual_n != frame_count:
                # Trim to actual_n frames without loading into RAM.
                if npy_path.exists():
                    npy_path.unlink()
                _copy_trim_memmap(tmp_path, npy_path, actual_n)
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            else:
                # Rename tmp -> final
                if npy_path.exists():
                    npy_path.unlink()
                tmp_path.rename(npy_path)
        else:
            arr = np.stack(frames_list, axis=0)
            np.save(npy_path, arr)

    return npy_path, (h, w)


def _profile_from_memmap(
    frames_path: Path, shape_hw: tuple[int, int], centile: float, row_block: int = 24
) -> np.ndarray:
    mm = np.load(frames_path, mmap_mode="r")
    if mm.ndim != 3:
        raise RuntimeError(f"Expected (N,H,W) stack, got {mm.shape}")
    n, h, w = int(mm.shape[0]), int(mm.shape[1]), int(mm.shape[2])
    if (h, w) != tuple(shape_hw):
        h, w = shape_hw
    if n <= 0:
        raise RuntimeError("No frames in stack.")

    out = np.zeros((h, w), dtype=np.float32)
    row_block = max(1, int(row_block))
    for y0 in range(0, h, row_block):
        y1 = min(h, y0 + row_block)
        block = np.asarray(mm[:, y0:y1, :], dtype=np.uint8)
        stat = np.percentile(block, q=float(centile), axis=0)
        out[y0:y1, :] = stat.astype(np.float32, copy=False)
    return out


def _profile_from_npy_stack(stack_path: Path, centile: float, row_block: int = 24) -> np.ndarray:
    mm = np.load(stack_path, mmap_mode="r")
    if mm.ndim not in (3, 4):
        raise RuntimeError(f"Expected NPY stack with ndim 3 or 4, got shape {mm.shape}")
    n = int(mm.shape[0])
    h = int(mm.shape[1])
    w = int(mm.shape[2])
    if n <= 0:
        raise RuntimeError("No frames in NPY stack.")

    out = np.zeros((h, w), dtype=np.float32)
    row_block = max(1, int(row_block))
    for y0 in range(0, h, row_block):
        y1 = min(h, y0 + row_block)
        block = np.asarray(mm[:, y0:y1, ...])
        if block.ndim == 4:
            block = block[..., 0]
        stat = np.percentile(block.astype(np.float32, copy=False), q=float(centile), axis=0)
        out[y0:y1, :] = stat.astype(np.float32, copy=False)
    return out


def _xy_profiles_from_npy_stack(
    stack_path: Path,
    centile: float,
    row_block: int = 24,
    dark_u8: np.ndarray | None = None,
    dark_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    mm = np.load(stack_path, mmap_mode="r")
    if mm.ndim not in (3, 4):
        raise RuntimeError(f"Expected NPY stack with ndim 3 or 4, got shape {mm.shape}")
    n = int(mm.shape[0])
    h = int(mm.shape[1])
    w = int(mm.shape[2])
    if n <= 0:
        raise RuntimeError("No frames in NPY stack.")
    if (h % 2) != 0 or (w % 2) != 0:
        raise RuntimeError(f"Frame shape must be even for polar mosaic, got {(h, w)}")
    if dark_u8 is not None and dark_u8.shape != (h, w):
        raise RuntimeError(f"Dark shape {dark_u8.shape} != frame shape {(h, w)}")

    h2 = h // 2
    w2 = w // 2
    x_out = np.zeros((h2, w2), dtype=np.float32)
    y_out = np.zeros((h2, w2), dtype=np.float32)
    row_block = max(1, int(row_block))
    eps = 1e-6
    dark_scaled = None if dark_u8 is None else (dark_u8.astype(np.float32, copy=False) * float(dark_scale))

    for yb0 in range(0, h2, row_block):
        yb1 = min(h2, yb0 + row_block)
        yr0 = 2 * yb0
        yr1 = 2 * yb1
        block = np.asarray(mm[:, yr0:yr1, ...])
        if block.ndim == 4:
            block = block[..., 0]
        bf = block.astype(np.float32, copy=False)
        if dark_scaled is not None:
            bf = bf - dark_scaled[yr0:yr1, :][None, :, :]
            np.maximum(bf, 0.0, out=bf)

        i0 = bf[:, 0::2, 0::2]
        i45 = bf[:, 0::2, 1::2]
        i135 = bf[:, 1::2, 0::2]
        i90 = bf[:, 1::2, 1::2]

        x = (i0 - i90) / (i0 + i90 + eps)
        y = (i45 - i135) / (i45 + i135 + eps)

        x_stat = np.percentile(x, q=float(centile), axis=0)
        y_stat = np.percentile(y, q=float(centile), axis=0)

        x_out[yb0:yb1, :] = x_stat.astype(np.float32, copy=False)
        y_out[yb0:yb1, :] = y_stat.astype(np.float32, copy=False)

    return x_out, y_out


def _load_dark_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read dark image: {path}")
    return _to_gray_u8(img)


def _prepare_xy_fit_samples(
    stack_path: Path,
    dark_u8: np.ndarray,
    frame_step: int = 5,
    pixel_step: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mm = np.load(stack_path, mmap_mode="r")
    if mm.ndim not in (3, 4):
        raise RuntimeError(f"Expected NPY stack with ndim 3 or 4, got shape {mm.shape}")
    n = int(mm.shape[0])
    h = int(mm.shape[1])
    w = int(mm.shape[2])
    if dark_u8.shape != (h, w):
        raise RuntimeError(f"Dark shape {dark_u8.shape} != frame shape {(h, w)}")

    frame_step = max(1, int(frame_step))
    pixel_step = max(1, int(pixel_step))
    idx = np.arange(0, n, frame_step, dtype=int)

    i0_list: list[np.ndarray] = []
    i45_list: list[np.ndarray] = []
    i135_list: list[np.ndarray] = []
    i90_list: list[np.ndarray] = []
    for i in idx:
        fr = np.asarray(mm[int(i)])
        if fr.ndim == 3:
            fr = fr[..., 0]
        ff = fr.astype(np.float32, copy=False)
        i0_list.append(ff[0::2, 0::2][::pixel_step, ::pixel_step])
        i45_list.append(ff[0::2, 1::2][::pixel_step, ::pixel_step])
        i135_list.append(ff[1::2, 0::2][::pixel_step, ::pixel_step])
        i90_list.append(ff[1::2, 1::2][::pixel_step, ::pixel_step])

    i0 = np.stack(i0_list, axis=0)
    i45 = np.stack(i45_list, axis=0)
    i135 = np.stack(i135_list, axis=0)
    i90 = np.stack(i90_list, axis=0)

    d = dark_u8.astype(np.float32, copy=False)
    d0 = d[0::2, 0::2][::pixel_step, ::pixel_step]
    d45 = d[0::2, 1::2][::pixel_step, ::pixel_step]
    d135 = d[1::2, 0::2][::pixel_step, ::pixel_step]
    d90 = d[1::2, 1::2][::pixel_step, ::pixel_step]
    return i0, i45, i135, i90, d0, d45, d135, d90


def _xy_maps_from_channels(
    i0: np.ndarray,
    i45: np.ndarray,
    i135: np.ndarray,
    i90: np.ndarray,
    centile: float,
    d0: np.ndarray | None = None,
    d45: np.ndarray | None = None,
    d135: np.ndarray | None = None,
    d90: np.ndarray | None = None,
    gamma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    a0 = i0.astype(np.float32, copy=False)
    a45 = i45.astype(np.float32, copy=False)
    a135 = i135.astype(np.float32, copy=False)
    a90 = i90.astype(np.float32, copy=False)

    if d0 is not None:
        g = float(gamma)
        a0 = a0 - g * d0[None, :, :]
        a45 = a45 - g * d45[None, :, :]
        a135 = a135 - g * d135[None, :, :]
        a90 = a90 - g * d90[None, :, :]
        np.maximum(a0, 0.0, out=a0)
        np.maximum(a45, 0.0, out=a45)
        np.maximum(a135, 0.0, out=a135)
        np.maximum(a90, 0.0, out=a90)

    x = (a0 - a90) / (a0 + a90 + eps)
    y = (a45 - a135) / (a45 + a135 + eps)
    x_map = np.percentile(x, q=float(centile), axis=0).astype(np.float32, copy=False)
    y_map = np.percentile(y, q=float(centile), axis=0).astype(np.float32, copy=False)
    return x_map, y_map


def _fit_gamma_for_xy(
    stack_path: Path,
    dark_u8: np.ndarray,
    centile: float,
    gmin: float,
    gmax: float,
) -> tuple[float, float]:
    i0, i45, i135, i90, d0, d45, d135, d90 = _prepare_xy_fit_samples(
        stack_path, dark_u8, frame_step=5, pixel_step=8
    )

    def objective(gamma: float) -> float:
        x_map, y_map = _xy_maps_from_channels(
            i0, i45, i135, i90, centile=centile, d0=d0, d45=d45, d135=d135, d90=d90, gamma=gamma
        )
        return float(np.std(x_map) + np.std(y_map))

    gmin = float(min(gmin, gmax))
    gmax = float(max(gmin, gmax))
    grid = np.linspace(gmin, gmax, 31)
    vals = np.asarray([objective(float(g)) for g in grid], dtype=np.float64)
    i_best = int(np.argmin(vals))
    g_best = float(grid[i_best])
    v_best = float(vals[i_best])

    step = float(grid[1] - grid[0]) if grid.size > 1 else max(1e-4, 0.1 * (gmax - gmin))
    lo = max(gmin, g_best - step)
    hi = min(gmax, g_best + step)
    fine = np.linspace(lo, hi, 41)
    fvals = np.asarray([objective(float(g)) for g in fine], dtype=np.float64)
    j_best = int(np.argmin(fvals))
    gf = float(fine[j_best])
    vf = float(fvals[j_best])
    if vf < v_best:
        return gf, vf
    return g_best, v_best


def _show_xy_distributions(
    x_before: np.ndarray,
    y_before: np.ndarray,
    x_after: np.ndarray,
    y_after: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt

        bins = np.linspace(-1.0, 1.0, 201)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), num=title)
        ax1.hist(x_before.ravel(), bins=bins, alpha=0.5, density=True, label="X before")
        ax1.hist(x_after.ravel(), bins=bins, alpha=0.5, density=True, label="X after")
        ax1.set_title("X distribution")
        ax1.set_xlabel("X")
        ax1.grid(alpha=0.2)
        ax1.legend()

        ax2.hist(y_before.ravel(), bins=bins, alpha=0.5, density=True, label="Y before")
        ax2.hist(y_after.ravel(), bins=bins, alpha=0.5, density=True, label="Y after")
        ax2.set_title("Y distribution")
        ax2.set_xlabel("Y")
        ax2.grid(alpha=0.2)
        ax2.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


def _normalize_01(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    hi = float(np.max(vals))
    if hi <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    y = x / hi
    y = np.clip(y, 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32, copy=False)


def _signed_to_u8(img: np.ndarray, clip_abs: float | None = None) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)
    if clip_abs is None:
        clip_abs = float(np.nanmax(np.abs(x[finite])))
    clip_abs = max(1e-9, float(clip_abs))
    y = np.clip((x / clip_abs + 1.0) * 0.5, 0.0, 1.0)
    y[~finite] = 0.5
    return (y * 255.0 + 0.5).astype(np.uint8)


def _show_popup(profile_01: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(title, figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(profile_01, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized intensity")
        plt.show()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default="", type=str, help="Path to .npy stack (preferred) or AVI")
    ap.add_argument("--sigma", type=float, default=50.0, help="Gaussian sigma in pixels (>=50 recommended).")
    ap.add_argument(
        "--keep-stack",
        action="store_true",
        help="Keep intermediate frame stack .npy (default: remove after profile build).",
    )
    args = ap.parse_args()

    inp = args.inp or INPUT_STACK_PATH
    if not inp:
        raise SystemExit("No input provided. Set INPUT_STACK_PATH at the top of the file or pass a path.")

    in_path = Path(inp).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")
    out_dir = in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma = float(max(50.0, args.sigma))
    centile = float(min(100.0, max(0.0, PROFILE_CENTILE)))
    if in_path.suffix.lower() == ".npy":
        frames_path = None
        stack_for_xy = in_path
        prof_img = _profile_from_npy_stack(in_path, centile=centile, row_block=24)
    else:
        frames_path, shape_hw = convert_avi_to_npy(in_path, out_dir=out_dir, save_frames=True)
        if frames_path is None:
            raise SystemExit("Failed to create intermediate frame stack.")
        stack_for_xy = frames_path
        prof_img = _profile_from_memmap(frames_path, shape_hw=shape_hw, centile=centile, row_block=24)
    smooth = cv2.GaussianBlur(
        prof_img.astype(np.float32, copy=False),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    profile_01 = _normalize_01(smooth)

    npy_out = out_dir / f"{in_path.stem}_illumination_profile_01.npy"
    png_out = out_dir / f"{in_path.stem}_illumination_profile_01.png"
    np.save(npy_out, profile_01.astype(np.float32, copy=False))
    Image.fromarray((profile_01 * 255.0 + 0.5).astype(np.uint8)).save(png_out)

    dark_path = Path(DARK_BMP_PATH)
    x_prof_before, y_prof_before = _xy_profiles_from_npy_stack(stack_for_xy, centile=centile, row_block=24)
    x_prof = x_prof_before
    y_prof = y_prof_before
    gamma_used = float(XY_DARK_SCALE)
    gamma_obj = float("nan")
    if dark_path.exists():
        dark_u8 = _load_dark_u8(dark_path)
        if bool(FIT_XY_GAMMA):
            gamma_used, gamma_obj = _fit_gamma_for_xy(
                stack_for_xy,
                dark_u8=dark_u8,
                centile=centile,
                gmin=float(GAMMA_MIN),
                gmax=float(GAMMA_MAX),
            )
        x_prof, y_prof = _xy_profiles_from_npy_stack(
            stack_for_xy,
            centile=centile,
            row_block=24,
            dark_u8=dark_u8,
            dark_scale=float(gamma_used),
        )
        print(f"Applied X/Y dark subtraction: frame <- max(frame - ({float(gamma_used):.6f} * dark), 0)")
        print(f"Dark source: {dark_path}")
    else:
        print(f"Dark source not found, X/Y computed without subtraction: {dark_path}")
    x_npy_out = out_dir / f"{in_path.stem}_x_profile.npy"
    y_npy_out = out_dir / f"{in_path.stem}_y_profile.npy"
    x_png_out = out_dir / f"{in_path.stem}_x_profile.png"
    y_png_out = out_dir / f"{in_path.stem}_y_profile.png"
    np.save(x_npy_out, x_prof.astype(np.float32, copy=False))
    np.save(y_npy_out, y_prof.astype(np.float32, copy=False))
    Image.fromarray(_signed_to_u8(x_prof)).save(x_png_out)
    Image.fromarray(_signed_to_u8(y_prof)).save(y_png_out)

    if (frames_path is not None) and (not args.keep_stack):
        try:
            frames_path.unlink()
        except Exception:
            pass

    print(f"Input: {in_path}")
    print(f"Profile centile: {centile:.1f}")
    print(f"Saved profile NPY: {npy_out}")
    print(f"Saved preview PNG: {png_out}")
    print(f"Saved X profile NPY: {x_npy_out}")
    print(f"Saved Y profile NPY: {y_npy_out}")
    print(f"Saved X profile PNG: {x_png_out}")
    print(f"Saved Y profile PNG: {y_png_out}")
    print(f"X std before/after correction: {float(np.std(x_prof_before)):.6f} / {float(np.std(x_prof)):.6f}")
    print(f"Y std before/after correction: {float(np.std(y_prof_before)):.6f} / {float(np.std(y_prof)):.6f}")
    print(f"Fitted gamma: {float(gamma_used):.8f}")
    if np.isfinite(gamma_obj):
        print(f"Gamma objective (stdX+stdY): {float(gamma_obj):.8f}")
    print(f"Profile min/max: {float(np.min(profile_01)):.6f} / {float(np.max(profile_01)):.6f}")
    _show_popup(profile_01, title=f"Illumination profile ({in_path.name})")
    _show_xy_distributions(
        x_prof_before,
        y_prof_before,
        x_prof,
        y_prof,
        title=f"X/Y distributions ({in_path.name})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
