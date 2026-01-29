#!/usr/bin/env python3
"""
characterise_illumination.py

Load an AVI, convert it to an NPY frame stack, and compute an illumination profile:
the per-pixel maximum over time.

Outputs (saved next to the AVI by default):
  - <stem>_frames.npy
  - background_profile.png
  - background_profile_stretched.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# Convenience: if you run this file directly in an IDE without CLI args,
# set this path and just press Run.
AVI_PATH2 = "/Users/hughbowman/Desktop/Oxford_RA/Polarcam_Software/polarcam-rewrite/offline/Spot_detection_alg_playgroun/background_profile2_280126.avi"  # e.g. "/Users/you/data/illumination_scan.avi"
AVI_PATH1 = "/Users/hughbowman/Desktop/Oxford_RA/Polarcam_Software/polarcam-rewrite/offline/Spot_detection_alg_playgroun/background_profile_280126.avi"
AVI_PATH = AVI_PATH2

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


def convert_avi_to_npy_and_max(
    avi_path: Path, out_dir: Path, save_frames: bool = True
) -> Tuple[Optional[Path], np.ndarray]:
    cap = _open_avi(avi_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame.")
    gray0 = _to_gray_u8(frame0)
    h, w = gray0.shape

    max_img = gray0.copy()

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
        np.maximum(max_img, gray, out=max_img)

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

    return npy_path, max_img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("avi", nargs="?", default="", type=str, help="Path to AVI file")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as AVI)",
    )
    ap.add_argument(
        "--no-save-npy",
        action="store_true",
        help="Do not write the frame stack .npy file (compute max only).",
    )
    args = ap.parse_args()

    avi_in = args.avi or AVI_PATH
    if not avi_in:
        raise SystemExit("No AVI provided. Set AVI_PATH at the top of the file or pass an AVI path.")

    avi_path = Path(avi_in).expanduser().resolve()
    if not avi_path.exists():
        raise SystemExit(f"File not found: {avi_path}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else avi_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path, max_img = convert_avi_to_npy_and_max(
        avi_path, out_dir=out_dir, save_frames=(not args.no_save_npy)
    )

    # Save background profile images.
    bg_path = out_dir / "background_profile.png"
    bg_stretch_path = out_dir / "background_profile_stretched.png"
    bg_npy_path = out_dir / "background_profile_gaussian_sigma20.npy"
    # Smooth the max-profile to suppress pixel-scale noise / dust.
    prof = max_img.astype(np.float32, copy=False)
    prof = cv2.GaussianBlur(prof, ksize=(0, 0), sigmaX=20.0, borderType=cv2.BORDER_REPLICATE)

    np.save(bg_npy_path, prof.astype(np.float32, copy=False))

    # Use mild percentile scaling for the "normal" output and log stretch for background visibility.
    u8 = _to_u8_percentile(prof, lo_pct=0.0, hi_pct=99.9, gamma=1.0)
    u8s = _to_u8_log_stretch(prof, lo_pct=0.0, hi_pct=99.9)
    Image.fromarray(u8).convert("RGB").save(bg_path)
    Image.fromarray(u8s).convert("RGB").save(bg_stretch_path)

    print(f"Wrote: {bg_path}")
    print(f"Wrote: {bg_stretch_path}")
    print(f"Wrote: {bg_npy_path}")
    if npy_path is not None:
        print(f"Wrote: {npy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
