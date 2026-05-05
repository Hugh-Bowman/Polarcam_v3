# pol_basic_player_min_throttled_display_with_qu_single_decode_process_all.py
import time
import json
import os
import shutil
import traceback
from pathlib import Path
import threading
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

try:
    from scipy.signal import welch as _welch  # type: ignore
    from scipy.signal import csd as _csd  # type: ignore
except Exception:  # pragma: no cover
    _welch = None
    _csd = None

try:
    mpl_cfg = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
    import matplotlib

    matplotlib.use("Agg")  # offscreen rendering for Tkinter image labels
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
except Exception:  # pragma: no cover
    Figure = None
    FigureCanvas = None

import Detection_alg_offline as detect_spinners
from pol_reconstruction import make_qu_reconstructor


def _append_timing_log(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        path = Path.cwd() / "spot_analysis_timing.txt"
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass


def _welch_fallback(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal Welch PSD fallback using NumPy only (two-sided).
    Returns empty arrays on failure.
    """
    x = np.asarray(x)
    if x.ndim != 1 or nperseg <= 0 or x.size < nperseg:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    fs = float(fs) if fs and fs > 0.0 else 1.0
    noverlap = int(noverlap) if noverlap is not None else 0
    step = max(1, int(nperseg) - max(0, noverlap))
    if step <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    if window in ("hann", "hanning"):
        win = np.hanning(nperseg)
    else:
        win = np.ones(nperseg, dtype=np.float64)

    scale = fs * float(np.sum(win * win))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    nseg = 1 + (x.size - nperseg) // step
    if nseg <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    acc = None
    for i in range(nseg):
        start = i * step
        seg = x[start : start + nperseg]
        if detrend == "constant":
            seg = seg - np.mean(seg)
        seg = seg * win
        fft = np.fft.fft(seg, n=nperseg)
        p = (np.abs(fft) ** 2) / scale
        acc = p if acc is None else (acc + p)

    psd = acc / float(nseg)
    freqs = np.fft.fftfreq(nperseg, d=1.0 / fs)
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def _csd_fallback(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal cross spectral density fallback using NumPy only (two-sided).
    Returns empty arrays on failure.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size < nperseg or y.size < nperseg:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.complex128)

    fs = float(fs) if fs and fs > 0.0 else 1.0
    noverlap = int(noverlap) if noverlap is not None else 0
    step = max(1, int(nperseg) - max(0, noverlap))
    if step <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.complex128)

    if window in ("hann", "hanning"):
        win = np.hanning(nperseg)
    else:
        win = np.ones(nperseg, dtype=np.float64)

    scale = fs * float(np.sum(win * win))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    nseg = 1 + (x.size - nperseg) // step
    if nseg <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.complex128)

    acc = None
    for i in range(nseg):
        start = i * step
        seg_x = x[start : start + nperseg]
        seg_y = y[start : start + nperseg]
        if detrend == "constant":
            seg_x = seg_x - np.mean(seg_x)
            seg_y = seg_y - np.mean(seg_y)
        seg_x = seg_x * win
        seg_y = seg_y * win
        fft_x = np.fft.fft(seg_x, n=nperseg)
        fft_y = np.fft.fft(seg_y, n=nperseg)
        c = (fft_x * np.conj(fft_y)) / scale
        acc = c if acc is None else (acc + c)

    pxy = acc / float(nseg)
    freqs = np.fft.fftfreq(nperseg, d=1.0 / fs)
    return np.asarray(freqs, dtype=np.float64), np.asarray(pxy, dtype=np.complex128)


def _safe_welch(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _welch is not None:
        try:
            return _welch(
                x,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                return_onesided=False,
                scaling="density",
            )
        except Exception:
            pass
    return _welch_fallback(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
    )


def _safe_csd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _csd is not None:
        try:
            return _csd(
                x,
                y,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                return_onesided=False,
                scaling="density",
            )
        except Exception:
            pass
    return _csd_fallback(
        x,
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
    )


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """
    Return a contiguous (H,W) uint8 frame.

    AVI frames arrive as BGR; NPY frames might be (H,W), (H,W,C), uint8/uint16/float.
    For non-uint8 inputs, apply a simple *global* downscale heuristic (no per-frame
    contrast normalization) so downstream processing stays consistent over time.
    """
    if frame is None:
        return None
    if frame.ndim == 3:
        try:
            frame = cv2.extractChannel(frame, 0)  # preserves dtype
        except Exception:
            frame = frame[..., 0]

    x = np.asarray(frame)
    if x.ndim != 2:
        return None

    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)

    if np.issubdtype(x.dtype, np.integer):
        maxv = int(x.max()) if x.size else 0
        if maxv <= 255:
            out = x.astype(np.uint8, copy=False)
        elif maxv <= 4095:
            out = (x.astype(np.uint16, copy=False) >> 4).astype(np.uint8, copy=False)
        else:
            out = (x.astype(np.uint32, copy=False) >> 8).astype(np.uint8, copy=False)
        return np.ascontiguousarray(out)

    if np.issubdtype(x.dtype, np.floating):
        # Avoid large temporary copies (x[np.isfinite(x)]) on big frames.
        maxv = float(np.nanmax(x)) if x.size else 0.0
        if not np.isfinite(maxv):
            finite_mask = np.isfinite(x)
            maxv = float(x[finite_mask].max()) if finite_mask.any() else 0.0
        if maxv <= 1.0:
            y = x * 255.0
        elif maxv <= 255.0:
            y = x
        elif maxv <= 4095.0:
            y = x / 16.0
        else:
            y = x / 256.0
        out = np.clip(y, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(out)

    return np.ascontiguousarray(x.astype(np.uint8))


class BasicVideoPlayer:
    DEFAULT_RING_SCORE_MIN = 0.0
    ABS_RANGE_MIN = 0.50
    AUTO_INSPECT_TOP_N = 5
    AUTO_INSPECT_FRAMES = 3125
    AUTO_INSPECT_FPS = 1500.0
    AUTO_INSPECT_ROI_RAW = 11
    MIN_RING_FRAMES = 20
    S_MAP_SMOOTH_K = int(detect_spinners.S_MAP_SMOOTH_K)
    EDGE_EXCLUDE_PX = int(detect_spinners.EDGE_EXCLUDE_PX)
    PLAYBACK_FPS = 20.0
    # Overview S-map display scale. 0.5 is "half-res".
    S_MAP_DISPLAY_SCALE = 0.4
    S_MAP_RING_R = 14          # full-res pixels (slightly larger rings for visibility)
    S_MAP_DISPLAY_GAMMA = 1.25  # less aggressive than log; suppresses background noise
    # Directionality analysis
    MIN_DIR_FRAMES = 32
    DIR_EXCLUDE_DC_HZ = 0.1
    DIR_BAND_HALF_WIDTH_BINS = 2
    # Directionality integration:
    # Use symmetric bounds: integrate all negative freqs (-fmax..0) and all positive (0..+fmax).
    # This is effectively (-inf..0) and (0..inf) within Welch's available band.
    DIR_PSD_THRESHOLD_FRAC = 0.02  # retained (no longer used for integration bounds)
    DIR_FILTER_B_MIN = 0.4
    FLAT_FIELD_FILENAME = "background_profile_gaussian_sigma20.npy"
    STATIONARY_R_MIN_DEFAULT = 0.35
    STATIONARY_BRIGHT_MIN_DEFAULT = 20.0
    STATIONARY_MOTION_MAX_DEFAULT = 0.12
    STATIONARY_MIN_FRAMES = 8
    STATIONARY_SEED_BRIGHT_PCT = 92.0
    STATIONARY_SEED_MIN_AREA = 2
    STATIONARY_MAX_CANDIDATES = 300
    STATIONARY_REC_77_FPS_DEFAULT = 77.0
    STATIONARY_REC_77_EXP_MS_DEFAULT = 0.02
    STATIONARY_REC_77_DURATION_S_DEFAULT = 5.0
    STATIONARY_REC_77_ROI_RAW_DEFAULT = 15
    STATIONARY_REC_MAX_EXP_MS_DEFAULT = 0.02
    STATIONARY_REC_MAX_DURATION_S_DEFAULT = 2.0
    STATIONARY_REC_MAX_FPS_EST_DEFAULT = 2000.0
    STATIONARY_REC_MAX_ROI_RAW = 11
    STATIONARY_REC_ALL_77_FPS = 77.0
    STATIONARY_REC_ALL_77_EXP_MS = 0.02
    STATIONARY_REC_ALL_77_DURATION_S = 5.0
    STATIONARY_REC_ALL_77_ROI_RAW = 15
    STATIONARY_REC_ALL_MAX_EXP_MS = 0.02
    STATIONARY_REC_ALL_MAX_DURATION_S = 2.0
    STATIONARY_REC_ALL_MAX_ROI_RAW = 11
    STATIONARY_DATASET_DIRNAME = "stationary_rod_dataset"
    STATIONARY_DATASET_PENDING_DIR = "pending"
    STATIONARY_DATASET_GOOD_DIR = "good"
    RECORDINGS_ROOT_DIRNAME = "recordings"
    RECORDINGS_WIDEFIELD_DIRNAME = "widefield_frames"
    RECORDINGS_SPOT_DIRNAME = "spots"

    def _find_centers_on_s_map(self, s_map_full: np.ndarray) -> list[tuple[float, float]]:
        """
        DoG-based spot finding on the full-resolution S-map.

        This is intended for "3-15px diameter" blobs (tune sigmas/area/k as needed).
        Edge exclusion is applied by cropping before running the detector.
        """
        if s_map_full is None:
            return []

        edge = int(self.EDGE_EXCLUDE_PX)
        h, w = s_map_full.shape
        if edge > 0 and (2 * edge) < min(h, w):
            work = s_map_full[edge : h - edge, edge : w - edge]
            offset = edge
        else:
            work = s_map_full
            offset = 0

        centers = detect_spinners.find_spot_centers_dog(
            work,
            sigma_small=float(detect_spinners.DOG_SIGMA_SMALL),
            sigma_large=float(detect_spinners.DOG_SIGMA_LARGE),
            k_std=float(self._dog_k_std),
            min_area=int(self._spot_min_area),
            max_area=int(self._spot_max_area) if self._spot_max_area is not None else None,
            connectivity=int(detect_spinners.DOG_CONNECTIVITY),
        )
        if offset:
            centers = [(cx + offset, cy + offset) for (cx, cy) in centers]
        return centers

    def _show_st2_popup(self, s_map: np.ndarray):
        """
        Save S_map images with detected spots.
        """
        # Diagnostics disabled: no S_map or frame image saving.
        return

    def _anisotropy_range_s_map(
        self,
        min_x: np.ndarray,
        max_x: np.ndarray,
        min_y: np.ndarray,
        max_y: np.ndarray,
        raw_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        # S: squared sum of the anisotropy ranges (unnormalised; no intensity weighting).
        rx = (max_x.astype(np.float32) - min_x.astype(np.float32))
        ry = (max_y.astype(np.float32) - min_y.astype(np.float32))
        s_int = (rx * rx) + (ry * ry)

        # Expand to raw pixel grid so spot coordinates remain in full-res space.
        h, w = raw_shape
        si_h, si_w = s_int.shape
        if (si_h, si_w) == (h, w):
            s_full = s_int
        elif (si_h, si_w) == (h - 1, w - 1):
            # Full-resolution intersection grid: preserve detail and just pad to sensor extents.
            s_full = np.pad(s_int, ((0, 1), (0, 1)), mode="edge")
        elif cv2 is not None:
            s_full = cv2.resize(s_int, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            s_full = np.repeat(np.repeat(s_int, 2, axis=0), 2, axis=1)
            s_full = s_full[:h, :w]
        return (s_full.astype(np.float32, copy=False), s_int.astype(np.float32, copy=False))

    def _ring_likeness_score(
        self, xy_series: list[tuple[float, float]], eps: float = 1e-12
    ) -> float:
        # High when points form a tight annulus (circular/elliptical) in XY.
        if len(xy_series) < int(self.MIN_RING_FRAMES):
            return 0.0

        arr = np.asarray(xy_series, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return 0.0
        x = arr[:, 0]
        y = arr[:, 1]
        if x.size < 3:
            return 0.0

        mx = float(np.mean(x))
        my = float(np.mean(y))
        zx = x - mx
        zy = y - my

        # covariance of centered points (2x2), ddof=1
        a = float(np.var(zx, ddof=1))
        c = float(np.var(zy, ddof=1))
        b = float(np.cov(zx, zy, ddof=1)[0, 1])
        C = np.array([[a + eps, b], [b, c + eps]], dtype=np.float64)

        try:
            w, V = np.linalg.eigh(C)
        except Exception:
            return 0.0
        w = np.maximum(w, eps)
        W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        Z = np.stack([zx, zy], axis=0)  # (2, N)
        U = W @ Z
        r = np.sqrt(U[0] * U[0] + U[1] * U[1])

        q25 = float(np.quantile(r, 0.25))
        q75 = float(np.quantile(r, 0.75))
        score = q25 / (q75 + eps)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return float(score)

    def _apply_ring_filter(self, force: bool = False) -> None:
        # Final filter: keep only spots whose XY scatter is annulus-like.
        # IMPORTANT: do not hold the analysis lock while computing ring scores (can be slow and blocks UI).
        with self._analysis_lock:
            if not self._spot_centers_all or not self._spot_xy_series_all:
                return
            ring_thr = float(self._ring_score_min)
            abs_thr = float(self.ABS_RANGE_MIN)
            abs_enabled = bool(self._abs_range_filter_enabled)
            centers_all = list(self._spot_centers_all)
            phi_all = list(self._spot_phi_series_all) if self._spot_phi_series_all else []
            xy_all = list(self._spot_xy_series_all)
            spot_win = int(self._spot_window_size)

        if ring_thr <= 0.0:
            # Threshold disabled => keep everything without computing the expensive score.
            keep = list(range(len(centers_all)))
        else:
            keep = []
            for i, series in enumerate(xy_all):
                if self._ring_likeness_score(series) >= ring_thr:
                    keep.append(i)
        ring_keep_n = len(keep)

        if abs_enabled:
            keep = [i for i in keep if self._spot_xy_max_axis_range(xy_all[i]) > abs_thr]

        prev_n = len(centers_all)
        if not keep:
            if abs_enabled:
                msg = (
                    f"Filters kept 0/{prev_n} "
                    f"(ring {ring_keep_n}/{prev_n}, max(range(X),range(Y))>{abs_thr:.2f})."
                )
            else:
                msg = f"Ring filter kept 0/{prev_n} (min={ring_thr:.2f})."
            self._ui_call(self.bottom_var.set, msg)
            if not force:
                return
            with self._analysis_lock:
                self._spot_centers = []
                self._spot_phi_series = []
                self._spot_xy_series = []
                self._spot_idx = 0
                self._spot_window_cache = []
                self._spot_window_cache_size = None
                self._spot_view_cache = []
            self._ui_call(self._rebuild_smap_overlay)
            return

        # Build filtered lists.
        filt_centers = [centers_all[i] for i in keep]
        filt_xy = [xy_all[i] for i in keep]
        filt_phi = [phi_all[i] for i in keep] if phi_all else []

        # Sort by XY range (strongest first) using the *spot-averaged* series,
        # but do the heavy part outside the lock so UI can keep updating.
        order = list(range(len(filt_centers)))
        if filt_xy:
            scores = [self._spot_xy_range_score(series) for series in filt_xy]
            if not all(score == float("-inf") for score in scores):
                order = sorted(order, key=lambda i: scores[i], reverse=True)
        filt_centers = [filt_centers[i] for i in order]
        filt_xy = [filt_xy[i] for i in order]
        if filt_phi:
            filt_phi = [filt_phi[i] for i in order]

        with self._analysis_lock:
            self._spot_centers = filt_centers
            self._spot_xy_series = filt_xy
            self._spot_phi_series = filt_phi
            # Snapshot the "post-hollowness" list for the directionality toggle.
            # Store list objects (not deep copies) so ongoing frame appends still update.
            self._dir_filter_base = (self._spot_centers, self._spot_xy_series, self._spot_phi_series)
            self._spot_idx = 0
            self._spot_window_cache = [None for _ in self._spot_centers]
            self._spot_window_cache_size = spot_win
            self._spot_view_cache = [
                {
                    "spot_u8": None,
                    "spot_win": None,
                    "phi_len": None,
                    "xy_len": None,
                    "fft": None,
                    "phi": None,
                    "xy": None,
                    "dir_len": None,
                    "dir_psd": None,
                    "dir_hand": None,
                    "dir_B": None,
                }
                for _ in self._spot_centers
            ]

        if abs_enabled:
            msg = (
                f"Filters kept {len(keep)}/{prev_n} "
                f"(ring {ring_keep_n}/{prev_n}, max(range(X),range(Y))>{abs_thr:.2f})."
            )
        else:
            msg = f"Ring filter kept {len(keep)}/{prev_n} (min={ring_thr:.2f})."
        self._ui_call(self.bottom_var.set, msg)
        self._ui_call(self._rebuild_smap_overlay)
        # If enabled, apply directionality filtering after hollowness filtering.
        if bool(getattr(self, "_dir_filter_enabled", False)):
            self._ui_call(self._apply_directionality_filter, True)

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def _start_ui_pump(self) -> None:
        # Tkinter must only be touched from the main thread on macOS.
        # Worker threads enqueue UI work into _ui_queue and the main thread drains it.
        if getattr(self, "_ui_pump_after_id", None) is not None:
            return

        def _tick():
            try:
                while True:
                    fn, args, kwargs = self._ui_queue.get_nowait()
                    try:
                        fn(*args, **kwargs)
                    except Exception:
                        # Don't let a single UI callback kill the pump.
                        pass
            except queue.Empty:
                pass
            self._ui_pump_after_id = self.root.after(30, _tick)

        self._ui_pump_after_id = self.root.after(30, _tick)

    def _ui_call(self, fn, *args, **kwargs) -> None:
        # Safe UI scheduling from any thread.
        if threading.get_ident() == self._ui_thread_id:
            self.root.after(0, lambda: fn(*args, **kwargs))
        else:
            self._ui_queue.put((fn, args, kwargs))

    def _find_flat_field_path(self, base_dir: Optional[Path]) -> Optional[Path]:
        # Prefer the directory containing the currently loaded video.
        candidates = []
        if base_dir is not None:
            candidates.append(base_dir / self.FLAT_FIELD_FILENAME)
        # Also check the working directory and this script's directory.
        candidates.append(Path.cwd() / self.FLAT_FIELD_FILENAME)
        try:
            candidates.append(Path(__file__).resolve().parent / self.FLAT_FIELD_FILENAME)
        except Exception:
            pass
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return None

    def _ensure_flat_field_loaded(self, shape: tuple[int, int], base_dir: Optional[Path]) -> bool:
        """
        Load and cache the flat-field correction (inverse profile) for the given shape.
        Returns True if correction can be applied.
        """
        if not bool(self._flat_field_enabled):
            return False
        if bool(getattr(self, "_flat_warned_mismatch", False)):
            return False

        with self._analysis_lock:
            inv = self._flat_inv
            prof_path = self._flat_profile_path
        if inv is not None and tuple(inv.shape) == tuple(shape) and prof_path is not None:
            return True

        p = self._find_flat_field_path(base_dir)
        if p is None:
            tried = []
            if base_dir is not None:
                tried.append(str((base_dir / self.FLAT_FIELD_FILENAME).resolve()))
            tried.append(str((Path.cwd() / self.FLAT_FIELD_FILENAME).resolve()))
            try:
                tried.append(str((Path(__file__).resolve().parent / self.FLAT_FIELD_FILENAME).resolve()))
            except Exception:
                pass
            self._ui_call(
                messagebox.showwarning,
                "Flat-field",
                "Could not find flat-field profile. Tried:\n" + "\n".join(tried),
            )
            return False

        try:
            prof = np.load(p, allow_pickle=True)
        except Exception as e:
            self._ui_call(messagebox.showerror, "Flat-field", f"Could not load profile: {e}")
            return False

        prof = np.asarray(prof)
        if prof.ndim != 2:
            self._ui_call(messagebox.showerror, "Flat-field", f"Profile must be 2D. Got {prof.shape}.")
            return False
        if tuple(prof.shape) != tuple(shape):
            if not getattr(self, "_flat_warned_mismatch", False):
                self._flat_warned_mismatch = True
                self._flat_field_enabled = False
                self._ui_call(
                    messagebox.showwarning,
                    "Flat-field",
                    f"Flat-field disabled: profile shape {prof.shape} does not match frame shape {shape}.",
                )
                self._ui_call(self._flat_field_enabled_var.set, False)
            return False

        prof_f = prof.astype(np.float32, copy=False)
        # New correction rule:
        # - any profile value <= 1 is treated as "no illumination" and set to 255
        # - frames are divided by this profile, so /255 suppresses those regions.
        prof_adj = prof_f.copy()
        prof_adj[~np.isfinite(prof_adj)] = 255.0
        prof_adj[prof_adj <= 7.0] = 255.0
        inv_f = 1.0 / prof_adj

        with self._analysis_lock:
            self._flat_profile = prof_adj
            self._flat_inv = inv_f
            self._flat_profile_path = p
            self._flat_field_enabled = True
        return True

    def _apply_flat_field(self, gray_u8: np.ndarray, base_dir: Optional[Path]) -> np.ndarray:
        """
        Apply flat-field correction to a grayscale uint8 mosaic frame:
          - values in the profile <= 1 are treated as 255 (suppresses unilluminated regions)
          - output = gray / profile
        Returns uint8 frame.
        """
        if gray_u8 is None or gray_u8.ndim != 2:
            return gray_u8
        if not bool(self._flat_field_enabled):
            return gray_u8
        if not self._ensure_flat_field_loaded(tuple(gray_u8.shape), base_dir):
            return gray_u8
        with self._analysis_lock:
            inv = self._flat_inv
        if inv is None:
            return gray_u8
        out = gray_u8.astype(np.float32, copy=False) * inv
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _parse_float(self, text: str) -> Optional[float]:
        try:
            return float(str(text).strip())
        except Exception:
            return None

    def _parse_int(self, text: str) -> Optional[int]:
        try:
            return int(round(float(str(text).strip())))
        except Exception:
            return None

    def _sync_fetch_from(self, changed: str) -> None:
        if getattr(self, "_fetch_sync_lock", False):
            return
        self._fetch_sync_lock = True
        try:
            fps = self._parse_float(self._fetch_fps_var.get())
            n_frames = self._parse_int(self._fetch_n_var.get())
            duration = self._parse_float(self._fetch_dur_var.get())

            if fps is None or fps <= 0.0:
                return

            if changed == "fps":
                if n_frames is not None and n_frames > 0:
                    dur = float(n_frames) / float(fps)
                    self._fetch_dur_var.set(f"{dur:.4f}")
            elif changed == "frames":
                if n_frames is not None and n_frames > 0:
                    dur = float(n_frames) / float(fps)
                    self._fetch_dur_var.set(f"{dur:.4f}")
            elif changed == "duration":
                if duration is not None and duration >= 0.0:
                    n = max(1, int(round(float(duration) * float(fps))))
                    self._fetch_n_var.set(str(n))
        finally:
            self._fetch_sync_lock = False

    def _set_fetch_busy(self, busy: bool) -> None:
        self._fetch_busy = bool(busy)
        if getattr(self, "_fetch_btn", None) is not None:
            if busy:
                self._fetch_btn.state(["disabled"])
            else:
                self._fetch_btn.state(["!disabled"])
                try:
                    self._fetch_btn.configure(state=tk.NORMAL)
                except Exception:
                    pass

    def _capture_frames_to_npy(
        self,
        out_path: Path,
        n_frames: int,
        fps: Optional[float],
        exp_ms: Optional[float],
    ) -> tuple[Path, Optional[float]]:
        script = Path(__file__).resolve().parent / "fetch_frames.py"
        out_dir = out_path.parent
        args = [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--n-frames",
            str(n_frames),
            "--stop-after",
            str(n_frames),
            "--json",
        ]
        if fps is not None:
            args.extend(["--fps", str(float(fps))])
        if exp_ms is not None:
            args.extend(["--exp-ms", str(float(exp_ms))])

        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or "fetch_frames.py failed.")

        payload = (proc.stdout or "").strip().splitlines()
        if not payload:
            raise RuntimeError("fetch_frames.py returned no output.")
        try:
            data = json.loads(payload[-1])
            path = Path(str(data.get("path", "")))
            actual_fps = data.get("actual_fps")
            actual_fps = float(actual_fps) if actual_fps is not None else None
        except Exception as e:
            raise RuntimeError(f"Could not parse fetch_frames output: {e}")
        if not path.exists():
            raise RuntimeError("fetch_frames.py did not produce an output file.")
        return path, actual_fps


    def _fetch_frames_worker(self, fps: float, exp_ms: float, n_frames: int) -> None:
        try:
            out_dir = self._recordings_subdir(self.RECORDINGS_WIDEFIELD_DIRNAME)
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = out_dir / f"frame_stack_{ts}.npy"
            self._ui_call(self.bottom_var.set, f"Fetching {n_frames} frame(s)...")
            saved_path, actual_fps = self._capture_frames_to_npy(out_path, n_frames, fps, exp_ms)
        except Exception as e:
            self._ui_call(messagebox.showerror, "Fetch frames", str(e))
            self._ui_call(self.bottom_var.set, "Fetch frames failed.")
        else:
            def _load_and_start():
                self._close_video()
                if self._load_npy_source(str(saved_path)):
                    use_fps = actual_fps if actual_fps and actual_fps > 0.0 else fps
                    if use_fps and use_fps > 0.0:
                        self.source_fps = float(use_fps)
                        if actual_fps and actual_fps > 0.0:
                            self._fetch_fps_var.set(f"{float(actual_fps):.3f}")
                    name = Path(saved_path).name
                    self.status_var.set(
                        f"Loaded: {name}  src≈{self.source_fps:.2f}fps  Q/U: 0%"
                    )
                    self.bottom_var.set(f"Fetch frames complete: {name}")
                self._set_fetch_busy(False)
            self._ui_call(_load_and_start)
        finally:
            self._ui_call(self._set_fetch_busy, False)

    def _on_fetch_frames(self) -> None:
        if getattr(self, "_fetch_busy", False):
            return

        fps = self._parse_float(self._fetch_fps_var.get())
        exp_ms = self._parse_float(self._fetch_exp_ms_var.get())
        n_frames = self._parse_int(self._fetch_n_var.get())

        if fps is None or fps <= 0.0:
            messagebox.showerror("Fetch frames", "Frame rate must be > 0.")
            return
        if exp_ms is None or exp_ms <= 0.0:
            messagebox.showerror("Fetch frames", "Exposure time must be > 0 ms.")
            return
        if n_frames is None or n_frames < 1:
            messagebox.showerror("Fetch frames", "Number of frames must be >= 1.")
            return

        # Reset analysis state before fetching a fresh stack.
        self._close_video()
        self._set_fetch_busy(True)
        t = threading.Thread(
            target=self._fetch_frames_worker,
            args=(float(fps), float(exp_ms), int(n_frames)),
            daemon=True,
        )
        t.start()

    def _on_flat_field_toggle(self) -> None:
        # When toggled, restart analysis for the currently loaded source to keep everything consistent.
        self._flat_field_enabled = bool(self._flat_field_enabled_var.get())
        # Drop cached profile so it is reloaded for the current video's folder/shape.
        with self._analysis_lock:
            self._flat_profile = None
            self._flat_inv = None
            self._flat_profile_path = None
            self._flat_warned_mismatch = False
        if not self.video_path:
            return
        path = self.video_path
        # Reload same source with new setting.
        self._close_video()
        self._load_source(path)

    def _clear_all_caches(self) -> None:
        """
        Hard-reset any caches / large references that can build up across runs.
        Intended to be called when a *new* source is opened so performance matches
        the first run.
        """
        # Cancel scheduled UI work that may reference old images/widgets.
        try:
            while True:
                self._ui_queue.get_nowait()
        except Exception:
            pass

        # Stop playback timers + clear playback buffers.
        self._stop_spot_playback()
        self._play_raw_u8 = []
        self._play_s_u8 = []
        self._play_frame_i = 0
        self._play_raw_ref = None
        self._play_s_ref = None
        self._dir_psd_ref = None
        self._dir_hand_ref = None
        self._dir_error_last = None

        # Clear all per-spot rendered images (PhotoImage refs keep memory alive in Tk).
        self._spot_img_ref = None
        self._fft_img_ref = None
        self._phi_img_ref = None
        self._xy_img_ref = None

        # Clear spot caches.
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._stationary_candidates = []
        self._stationary_idx = 0
        self._set_selected_center_override(None, source="analysis")

        # Clear decoded frame cache (AVI only; NPY isn't cached anyway).
        with self._gray_lock:
            self._gray_frames = []

        # Clear directionality filter base snapshot.
        self._dir_filter_base = ([], [], [])
        self._dir_filter_enabled = bool(self._dir_filter_enabled_var.get())
        self._abs_range_filter_enabled = bool(self._abs_range_filter_enabled_var.get())
        self._auto_inspect_enabled = bool(self._auto_inspect_enabled_var.get())
        with self._auto_inspect_start_lock:
            self._auto_inspect_running = False
        self._analysis_finished = False
        self._spot_inspect_overrides = {}
        # Clear flat-field cache (profile depends on input frame shape).
        self._flat_profile = None
        self._flat_inv = None
        self._flat_profile_path = None
        self._flat_warned_mismatch = False
        self._flat_field_enabled = bool(self._flat_field_enabled_var.get())
        self._reset_live_tracking(keep_shift=False)
        self._update_stationary_view()

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AVI/NPY Spot Analysis (DoG spot detection)")
        self._ui_thread_id = threading.get_ident()
        self._ui_queue: "queue.Queue[tuple[object, tuple, dict]]" = queue.Queue()
        self._ui_pump_after_id = None
        self._dir_filter_enabled_var = tk.BooleanVar(value=False)
        self._dir_filter_enabled = False
        self._dir_filter_base = ([], [], [])
        self._abs_range_filter_enabled_var = tk.BooleanVar(value=True)
        self._abs_range_filter_enabled = True
        self._auto_inspect_enabled_var = tk.BooleanVar(value=False)
        self._auto_inspect_enabled = False
        self._auto_inspect_chk = None
        self._auto_inspect_running = False
        self._auto_inspect_start_lock = threading.Lock()
        self._analysis_finished = False
        # Flat-field / illumination correction
        self._flat_field_enabled_var = tk.BooleanVar(value=False)
        self._flat_field_enabled = False
        self._flat_profile = None  # float32 (H,W)
        self._flat_inv = None      # float32 (H,W) : scale/profile, zeros where profile==0
        self._flat_profile_path = None
        self._flat_warned_mismatch = False
        # Fetch-frames controls
        self._fetch_busy = False
        self._fetch_sync_lock = False
        self._fetch_exp_ms_var = tk.StringVar(value="0.02")
        self._fetch_fps_var = tk.StringVar(value="78")
        self._fetch_n_var = tk.StringVar(value="150")
        self._fetch_dur_var = tk.StringVar(value="")
        self._fetch_btn = None
        self._sync_fetch_from("fps")
        self._fetch_fps_var.trace_add("write", lambda *_: self._sync_fetch_from("fps"))
        self._fetch_n_var.trace_add("write", lambda *_: self._sync_fetch_from("frames"))
        self._fetch_dur_var.trace_add("write", lambda *_: self._sync_fetch_from("duration"))
        # Live view state
        self._live_running = False
        self._live_controller = None
        self._live_app = None
        self._live_queue = queue.Queue(maxsize=2)
        self._live_after_id = None
        self._live_img_ref = None
        self._live_img_label = None
        self._live_zoom_label = None
        self._live_zoom_blank_ref = None
        self._live_left_frame = None
        # Spot capture (tab 3)
        self._spotrec_running = False
        self._spotrec_controller = None
        self._spotrec_app = None
        self._spotrec_queue = queue.Queue(maxsize=4)
        self._spotrec_after_id = None
        self._spotrec_xy_series = []
        self._spotrec_phi_series = []
        self._spotrec_frames = []
        self._spotrec_actual_fps = None
        self._spotrec_tmp_path = None
        self._spotrec_proc = None
        self._spotrec_stop_flag = None
        self._spotrec_out_path = None
        self._spotrec_roi_meta = None
        self._spotrec_fps_var = tk.StringVar(value="2000")
        self._spotrec_exp_ms_var = tk.StringVar(value="0.02")
        self._spotrec_size_var = tk.StringVar(value="11")
        self._spotrec_spot_var = tk.StringVar(value="Spot - / -")
        self._spotrec_status_var = tk.StringVar(value="Idle")
        self._spotrec_progress_var = tk.StringVar(value="")
        self._spotrec_start_btn = None
        self._spotrec_stop_btn = None
        self._spotrec_save_btn = None
        self._spotrec_discard_btn = None
        self._spotrec_tmp_path = None
        self._spotrec_preview_label = None
        self._spotrec_preview_scale = 10
        self._spotrec_preview_size = 19
        self._spotrec_preview_after_id = None
        self._spotrec_preview_interval_ms = 100  # ~10 fps
        self._spotrec_preview_frame_i = 0
        self._spotrec_preview_path = None
        self._spotrec_preview_mtime = None
        self._spotrec_preview_every = 100
        self._spotrec_preview_last_frame = None
        self._spot_play_btn = None
        self._spotrec_center_offset = (0, 0)
        self._spotrec_xy_label = None
        self._spotrec_phi_label = None
        self._spotrec_fft_label = None
        self._spotrec_dir_label = None
        self._spotrec_hand_label = None
        self._spotrec_dir_var = tk.StringVar(value="B: -")
        self._spotrec_brownian_var = tk.StringVar(value="Brownian: -")
        self._spotrec_xy_ref = None
        self._spotrec_phi_ref = None
        self._spotrec_fft_ref = None
        self._spotrec_dir_ref = None
        self._spotrec_hand_ref = None
        self._live_start_btn = None
        self._live_stop_btn = None
        self._live_exp_ms_var = tk.StringVar(value="0.05")
        self._live_gain_var = tk.StringVar(value="20")
        self._live_status_var = tk.StringVar(value="Live feed stopped")
        self._live_mag_enabled_var = tk.BooleanVar(value=False)
        self._live_zoom_var = tk.StringVar(value="3.0")
        self._live_zoom_center = None  # (x,y) in source frame pixels
        self._live_last_frame = None
        self._live_disp_scale = 1.0
        self._live_disp_offset = (0, 0)
        self._live_zoom_output_px = 240
        # Live frame registration/tracking state (center-template translation fit).
        self._live_track_prev = None  # previous blurred uint8 frame
        self._live_track_shift = (0.0, 0.0)  # cumulative dx,dy in full-res pixels
        self._live_track_resp = 0.0
        self._live_track_interval = 1  # track every N displayed frames
        self._live_track_counter = 0
        self._live_track_roi_size = 1200
        self._live_track_template_size = 800
        self._live_track_blur_sigma = 1.2
        self._live_track_bg_sigma = 8.0
        self._live_track_bright_pct = 92.0
        self._live_track_score_min = 0.05
        self._live_track_axis_lock = False
        self._live_track_min_axis_step = 0.25

        # Single capture (decoded once)
        self.cap = None
        self.video_path = None
        self.source_kind = None
        self.npy_frames = None
        self.npy_has_frames_dim = False
        self._source_shape = None

        # Initial S-map metric (first N frames)
        # legacy buffers (kept for now)
        self._q_buf = []
        self._u_buf = []
        self._st2_frames = max(2, int(detect_spinners.S_MAP_FRAMES))
        self._st_popup_done = False
        self._st_popup_img_ref = None  # keep PhotoImage alive
        self._st_popup_label = None
        self._overlay_base_frame = None
        self._s_map = None
        self._s_map_int = None
        self._spot_centers_all = []
        self._spot_centers = []
        self._spot_idx = 0
        self._selected_center_override = None  # Optional[(cx, cy)] from non-analysis selectors.
        self._selected_center_source = "analysis"
        self._spot_phi_series_all = []
        self._spot_phi_series = []
        self._phi_frames_processed = 0
        self._spot_xy_series_all = []
        self._spot_xy_series = []
        self._xy_frames_processed = 0
        self._spot_bounds_int_all = []
        self._spot_window_size = 19
        self._spot_scale = 10
        self._dog_k_std = float(detect_spinners.DOG_K_STD)
        self._spot_min_area = int(detect_spinners.DOG_MIN_AREA)
        self._spot_max_area = int(detect_spinners.DOG_MAX_AREA)
        self._ring_score_min = float(self.DEFAULT_RING_SCORE_MIN)
        self._spot_img_ref = None
        self._fft_img_ref = None
        self._phi_img_ref = None
        self._xy_img_ref = None
        self._dir_psd_ref = None
        self._dir_hand_ref = None
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._spot_inspect_overrides = {}
        # S-map overview canvas (background + overlay items)
        self._smap_canvas = None
        self._smap_bg_ref = None
        self._smap_canvas_img_id = None
        self._smap_spot_ring_ids = []
        self._smap_spot_text_ids = []
        self._smap_overlay_after_id = None
        self._smap_overlay_pending = []
        self._stationary_candidates = []
        self._stationary_idx = 0
        self._stationary_status_var = tk.StringVar(value="Stationary 0 / 0")
        self._stationary_metrics_var = tk.StringVar(value="")
        self._stationary_brightness_min_var = tk.StringVar(
            value=f"{self.STATIONARY_BRIGHT_MIN_DEFAULT:.1f}"
        )
        self._stationary_r_min_var = tk.StringVar(value=f"{self.STATIONARY_R_MIN_DEFAULT:.2f}")
        self._stationary_motion_max_var = tk.StringVar(
            value=f"{self.STATIONARY_MOTION_MAX_DEFAULT:.2f}"
        )
        self._stationary_prev_btn = None
        self._stationary_next_btn = None
        self._stationary_capture_77_fps_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_77_FPS_DEFAULT:.2f}"
        )
        self._stationary_capture_77_exp_ms_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_77_EXP_MS_DEFAULT:.3f}"
        )
        self._stationary_capture_77_duration_s_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_77_DURATION_S_DEFAULT:.2f}"
        )
        self._stationary_capture_77_roi_var = tk.StringVar(
            value=str(int(self.STATIONARY_REC_77_ROI_RAW_DEFAULT))
        )
        self._stationary_capture_max_exp_ms_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_MAX_EXP_MS_DEFAULT:.3f}"
        )
        self._stationary_capture_max_duration_s_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_MAX_DURATION_S_DEFAULT:.2f}"
        )
        self._stationary_capture_max_fps_est_var = tk.StringVar(
            value=f"{self.STATIONARY_REC_MAX_FPS_EST_DEFAULT:.1f}"
        )
        self._stationary_capture_status_var = tk.StringVar(value="Stationary capture idle.")
        self._stationary_capture_running = False
        self._stationary_capture_lock = threading.Lock()
        self._stationary_capture_selected_btn = None
        self._stationary_capture_all_btn = None
        self._stationary_spot_img_label = None
        self._stationary_xy_label = None
        self._stationary_phi_label = None
        self._stationary_spot_ref = None
        self._stationary_xy_ref = None
        self._stationary_phi_ref = None
        self._stationary_review_status_var = tk.StringVar(value="Pending 0 | Good 0")
        self._stationary_review_detail_var = tk.StringVar(value="No stationary recordings yet.")
        self._stationary_review_theta_var = tk.StringVar(value="")
        self._stationary_review_include_var = tk.BooleanVar(value=False)
        self._stationary_review_include_sync = False
        self._stationary_review_idx = 0
        self._stationary_review_items: list[tuple[str, Path]] = []
        self._stationary_review_prev_btn = None
        self._stationary_review_next_btn = None
        self._stationary_review_mark_chk = None
        self._stationary_review_xy77_label = None
        self._stationary_review_xymax_label = None
        self._stationary_review_hist_label = None
        self._stationary_review_xy77_ref = None
        self._stationary_review_xymax_ref = None
        self._stationary_review_hist_ref = None
        self._stationary_dataset_tab = None
        self._play_popup = None
        self._play_running = False
        self._play_after_id = None
        self._play_frame_i = 0
        self._play_raw_u8 = []
        self._play_s_u8 = []
        self._play_raw_label = None
        self._play_s_label = None
        self._play_status_var = tk.StringVar(value="")
        self._play_raw_max_var = tk.StringVar(value="")
        self._play_raw_ref = None
        self._play_s_ref = None
        self._gray_frames = []
        self._gray_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._text_font = self._load_font(size=14)
        # Cache decoded grayscale frames only for AVI sources.
        # NPY sources already store frames on disk; caching them again can blow RAM.
        self._cache_gray_frames = True

        self.frame_count = 0
        self.source_fps = 30.0  # metadata only; display is independent
        self.current_idx = 0

        # Latest decoded frame (kept for debugging / optional image export)
        self.last_frame_gray = None

        # Threads + coordination
        self.decode_thread = None
        self.recon_thread = None
        self.stop_event = threading.Event()

        # EOF indicator for decoder
        self.decode_done = False

        # Processing progress
        self.proc_done = 0  # frames processed by recon thread

        # Queue: bounded to avoid RAM blow-up, BLOCKING put => no frame drops
        self.frame_q = queue.Queue(maxsize=16)

        self._build_ui()
        self._update_spotrec_label()
        self._start_ui_pump()
        self._spotrec_size_var.trace_add("write", lambda *_: self._spotrec_update_preview())
        self._stationary_review_include_var.trace_add(
            "write", lambda *_: self._on_stationary_review_include_toggle()
        )

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._notebook = notebook

        self._live_tab = ttk.Frame(notebook)
        self._analysis_tab = ttk.Frame(notebook)
        self._stationary_tab = ttk.Frame(notebook)
        self._stationary_dataset_tab = ttk.Frame(notebook)
        self._spotrec_tab = ttk.Frame(notebook)
        notebook.add(self._live_tab, text="Live video")
        notebook.add(self._analysis_tab, text="Spot analysis")
        notebook.add(self._stationary_tab, text="Stationary rods")
        notebook.add(self._stationary_dataset_tab, text="Stationary dataset")
        notebook.add(self._spotrec_tab, text="Spot examine")
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_live_ui(self._live_tab)
        self._build_analysis_ui(self._analysis_tab)
        self._build_stationary_ui(self._stationary_tab)
        self._build_stationary_dataset_ui(self._stationary_dataset_tab)
        self._build_spotrec_ui(self._spotrec_tab)

    def _build_live_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self._live_start_btn = ttk.Button(top, text="Start live feed", command=self._start_live_feed)
        self._live_start_btn.pack(side=tk.LEFT)
        self._live_stop_btn = ttk.Button(top, text="Stop live feed", command=self._stop_live_feed)
        self._live_stop_btn.state(["disabled"])
        self._live_stop_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="FPS 20").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, text="Exp (ms)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._live_exp_ms_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Gain").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._live_gain_var, width=7).pack(side=tk.LEFT)
        ttk.Button(top, text="Apply", command=self._apply_live_settings).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(top, text="Magnifier", variable=self._live_mag_enabled_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, text="Zoom x").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Entry(top, textvariable=self._live_zoom_var, width=5).pack(side=tk.LEFT)

        ttk.Label(top, textvariable=self._live_status_var).pack(side=tk.RIGHT)

        view = ttk.Frame(parent, padding=8)
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        view.columnconfigure(0, weight=1)
        view.columnconfigure(1, weight=0, minsize=self._live_zoom_output_px + 10)
        view.rowconfigure(0, weight=1)

        left = ttk.Frame(view)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)
        self._live_left_frame = left
        self._live_img_label = tk.Label(left, bg="black")
        self._live_img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._live_img_label.bind("<Button-1>", self._on_live_click)

        right = ttk.Frame(view, width=self._live_zoom_output_px, height=self._live_zoom_output_px)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))
        right.grid_propagate(False)
        self._live_zoom_label = tk.Label(right, bg="black")
        self._live_zoom_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        blank = Image.new("L", (self._live_zoom_output_px, self._live_zoom_output_px), 0)
        self._live_zoom_blank_ref = ImageTk.PhotoImage(blank)
        self._live_zoom_label.configure(image=self._live_zoom_blank_ref)

    def _build_analysis_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Select AVI/NPY", command=self.open_video).pack(side=tk.LEFT)
        self._fetch_btn = ttk.Button(top, text="Fetch frames", command=self._on_fetch_frames)
        self._fetch_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(top, text="Exp (ms)").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(top, textvariable=self._fetch_exp_ms_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="FPS").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(top, textvariable=self._fetch_fps_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Frames").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(top, textvariable=self._fetch_n_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Dur (s)").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(top, textvariable=self._fetch_dur_var, width=8).pack(side=tk.LEFT)
        ttk.Checkbutton(
            top,
            text=f"Flat-field ({self.FLAT_FIELD_FILENAME})",
            variable=self._flat_field_enabled_var,
            command=self._on_flat_field_toggle,
        ).pack(side=tk.LEFT, padx=(10, 0))
        self.status_var = tk.StringVar(value="No video loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(parent, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        content = ttk.Frame(main)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Middle panel: scrollable so the plots fit on smaller screens.
        right_outer = ttk.Frame(content)
        right_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        right_canvas = tk.Canvas(right_outer, highlightthickness=0)
        right_canvas.configure(width=360)
        right_scroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        right = ttk.Frame(right_canvas)
        right_window_id = right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _right_on_configure(_evt=None):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            # Keep the embedded frame width matched to the canvas width.
            right_canvas.itemconfigure(right_window_id, width=right_canvas.winfo_width())

        right.bind("<Configure>", _right_on_configure)
        right_canvas.bind("<Configure>", _right_on_configure)

        # Right-most panel: directionality analysis (separate from the main analysis panel).
        ttk.Separator(content, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(10, 10))
        dir_outer = ttk.Frame(content)
        dir_outer.pack(side=tk.LEFT, fill=tk.Y)
        dir_canvas = tk.Canvas(dir_outer, highlightthickness=0)
        dir_canvas.configure(width=360)
        dir_scroll = ttk.Scrollbar(dir_outer, orient=tk.VERTICAL, command=dir_canvas.yview)
        dir_canvas.configure(yscrollcommand=dir_scroll.set)
        dir_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        dir_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        dir_panel = ttk.Frame(dir_canvas)
        dir_window_id = dir_canvas.create_window((0, 0), window=dir_panel, anchor="nw")

        def _dir_on_configure(_evt=None):
            dir_canvas.configure(scrollregion=dir_canvas.bbox("all"))
            dir_canvas.itemconfigure(dir_window_id, width=dir_canvas.winfo_width())

        dir_panel.bind("<Configure>", _dir_on_configure)
        dir_canvas.bind("<Configure>", _dir_on_configure)

        # Left: main S_map overview (S_smoothed, scaled by 1/2) + overlay circles.
        self._smap_canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self._smap_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._smap_canvas.bind("<Button-1>", self._on_smap_click)

        # Right: controls + plots.
        controls = ttk.Frame(right, padding=(0, 0, 0, 10))
        controls.pack(side=tk.TOP, fill=tk.X)

        nav = ttk.Frame(controls)
        nav.pack(side=tk.TOP, fill=tk.X)
        self.spot_prev_btn = ttk.Button(nav, text="<", command=self._prev_spot, width=3)
        self.spot_prev_btn.pack(side=tk.LEFT)
        self.spot_next_btn = ttk.Button(nav, text=">", command=self._next_spot, width=3)
        self.spot_next_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._spot_status_var = tk.StringVar(value="Spot 0 / 0")
        ttk.Label(nav, textvariable=self._spot_status_var).pack(side=tk.LEFT, padx=(8, 0))
        self._spot_play_btn = ttk.Button(nav, text="Play spot", command=self._open_spot_playback)
        self._spot_play_btn.pack(side=tk.RIGHT)

        params = ttk.Frame(controls)
        params.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(params, text="DoG k").grid(row=0, column=0, sticky="w")
        self._dog_k_var = tk.StringVar(value=f"{self._dog_k_std:.2f}")
        ttk.Entry(params, textvariable=self._dog_k_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(params, text="Phi window").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._spot_win_var = tk.StringVar(value=str(self._spot_window_size))
        ttk.Entry(params, textvariable=self._spot_win_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(params, text="Min Hollowness Score").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._ring_score_min_var = tk.StringVar(value=f"{self._ring_score_min:.2f}")
        ttk.Entry(params, textvariable=self._ring_score_min_var, width=10).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Checkbutton(
            params,
            text=f"Filter max(range(X), range(Y)) > {self.ABS_RANGE_MIN:.2f}",
            variable=self._abs_range_filter_enabled_var,
            command=self._on_abs_range_filter_toggle,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._auto_inspect_chk = ttk.Checkbutton(
            params,
            text=f"Auto inspect top {self.AUTO_INSPECT_TOP_N} spots after analysis",
            variable=self._auto_inspect_enabled_var,
            command=self._on_auto_inspect_toggle,
        )
        self._auto_inspect_chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._auto_inspect_chk.state(["disabled"])

        self._spot_update_btn = ttk.Button(controls, text="Update analysis", command=self._apply_spot_params)
        self._spot_update_btn.pack(side=tk.TOP, anchor="w", pady=(8, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(right, text="S-map window").pack(side=tk.TOP, anchor="w")
        self._spot_img_label = ttk.Label(right)
        self._spot_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="X/Y scatter").pack(side=tk.TOP, anchor="w")
        self._xy_img_label = ttk.Label(right)
        self._xy_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="Phi(t)").pack(side=tk.TOP, anchor="w")
        self._phi_img_label = ttk.Label(right)
        self._phi_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="Phi FFT").pack(side=tk.TOP, anchor="w")
        self._fft_img_label = ttk.Label(right)
        self._fft_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 0))

        # Directionality panel (right-most).
        ttk.Label(dir_panel, text="Rotation directionality").pack(side=tk.TOP, anchor="w")
        ttk.Checkbutton(
            dir_panel,
            text=f"Filter unidirectional (|B| > {self.DIR_FILTER_B_MIN:.2f})",
            variable=self._dir_filter_enabled_var,
            command=self._on_dir_filter_toggle,
        ).pack(side=tk.TOP, anchor="w", pady=(4, 8))
        self._dir_var = tk.StringVar(value="B: -")
        ttk.Label(dir_panel, textvariable=self._dir_var).pack(side=tk.TOP, anchor="w", pady=(2, 8))

        ttk.Label(dir_panel, text="Two-sided PSD of Z=X+iY").pack(side=tk.TOP, anchor="w")
        self._dir_psd_label = ttk.Label(dir_panel)
        self._dir_psd_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(dir_panel, text="Handedness spectrum Im{CSD(X,Y)}").pack(side=tk.TOP, anchor="w")
        self._dir_hand_label = ttk.Label(dir_panel)
        self._dir_hand_label.pack(side=tk.TOP, anchor="w", pady=(2, 0))

        self.bottom_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.bottom_var).pack(side=tk.BOTTOM, anchor="w")

    def _build_stationary_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Min brightness").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._stationary_brightness_min_var, width=8).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(top, text="Min mean r").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._stationary_r_min_var, width=8).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(top, text="Max range XY").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._stationary_motion_max_var, width=8).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Button(top, text="Find stationary rods", command=self._refresh_stationary_candidates).pack(
            side=tk.LEFT
        )
        ttk.Label(top, textvariable=self._stationary_status_var).pack(side=tk.RIGHT)

        nav = ttk.Frame(parent, padding=(8, 0, 8, 0))
        nav.pack(side=tk.TOP, fill=tk.X)
        self._stationary_prev_btn = ttk.Button(
            nav, text="<", width=3, command=self._prev_stationary_spot
        )
        self._stationary_prev_btn.pack(side=tk.LEFT)
        self._stationary_next_btn = ttk.Button(
            nav, text=">", width=3, command=self._next_stationary_spot
        )
        self._stationary_next_btn.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(nav, textvariable=self._stationary_metrics_var).pack(side=tk.LEFT, padx=(10, 0))

        capture = ttk.Frame(parent, padding=(8, 6, 8, 0))
        capture.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(capture, text="77fps mode: FPS").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_77_fps_var, width=6).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(capture, text="Exp (ms)").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_77_exp_ms_var, width=6).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(capture, text="Duration (s)").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_77_duration_s_var, width=6).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(capture, text="FOV px").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_77_roi_var, width=6).pack(
            side=tk.LEFT, padx=(4, 16)
        )
        ttk.Label(capture, text="Max-FPS mode: 11x11, Exp (ms)").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_max_exp_ms_var, width=6).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(capture, text="Duration (s)").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_max_duration_s_var, width=6).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(capture, text="FPS estimate").pack(side=tk.LEFT)
        ttk.Entry(capture, textvariable=self._stationary_capture_max_fps_est_var, width=7).pack(
            side=tk.LEFT, padx=(4, 12)
        )
        self._stationary_capture_selected_btn = ttk.Button(
            capture, text="Record selected (both modes)", command=self._stationary_capture_selected
        )
        self._stationary_capture_selected_btn.pack(side=tk.LEFT)
        self._stationary_capture_all_btn = ttk.Button(
            capture, text="Record all stationary", command=self._stationary_capture_all
        )
        self._stationary_capture_all_btn.pack(side=tk.LEFT, padx=(6, 0))

        capture_status = ttk.Frame(parent, padding=(8, 2, 8, 2))
        capture_status.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(capture_status, textvariable=self._stationary_capture_status_var).pack(
            side=tk.LEFT, anchor="w"
        )

        plots = ttk.Frame(parent, padding=8)
        plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        plots.rowconfigure(1, weight=1)

        ttk.Label(plots, text="S-map window").grid(row=0, column=0, sticky="w")
        ttk.Label(plots, text="X/Y scatter").grid(row=0, column=1, sticky="w")
        self._stationary_spot_img_label = ttk.Label(plots)
        self._stationary_spot_img_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self._stationary_xy_label = ttk.Label(plots)
        self._stationary_xy_label.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        phi_row = ttk.Frame(parent, padding=(8, 0, 8, 8))
        phi_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(phi_row, text="Phi(t)").pack(side=tk.TOP, anchor="w")
        self._stationary_phi_label = ttk.Label(phi_row)
        self._stationary_phi_label.pack(side=tk.TOP, anchor="w")

        self._update_stationary_view()

    def _build_stationary_dataset_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Refresh dataset", command=self._stationary_review_refresh).pack(
            side=tk.LEFT
        )
        ttk.Label(top, textvariable=self._stationary_review_status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, textvariable=self._stationary_review_theta_var).pack(side=tk.RIGHT)

        nav = ttk.Frame(parent, padding=(8, 0, 8, 0))
        nav.pack(side=tk.TOP, fill=tk.X)
        self._stationary_review_prev_btn = ttk.Button(
            nav, text="<", width=3, command=self._stationary_review_prev
        )
        self._stationary_review_prev_btn.pack(side=tk.LEFT)
        self._stationary_review_next_btn = ttk.Button(
            nav, text=">", width=3, command=self._stationary_review_next
        )
        self._stationary_review_next_btn.pack(side=tk.LEFT, padx=(4, 8))
        self._stationary_review_mark_chk = ttk.Checkbutton(
            nav,
            text="Good enough (tick to move into good folder)",
            variable=self._stationary_review_include_var,
        )
        self._stationary_review_mark_chk.pack(side=tk.LEFT)

        ttk.Label(
            parent,
            textvariable=self._stationary_review_detail_var,
            padding=(8, 6, 8, 2),
            justify=tk.LEFT,
        ).pack(side=tk.TOP, anchor="w", fill=tk.X)

        plots = ttk.Frame(parent, padding=8)
        plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        plots.rowconfigure(1, weight=1)

        ttk.Label(plots, text="XY trace (77fps mode)").grid(row=0, column=0, sticky="w")
        ttk.Label(plots, text="XY trace (max-fps 11x11 mode)").grid(row=0, column=1, sticky="w")
        self._stationary_review_xy77_label = ttk.Label(plots)
        self._stationary_review_xy77_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self._stationary_review_xymax_label = ttk.Label(plots)
        self._stationary_review_xymax_label.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        hist_row = ttk.Frame(parent, padding=(8, 0, 8, 8))
        hist_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(hist_row, text="Accepted rods: theta coverage").pack(side=tk.TOP, anchor="w")
        self._stationary_review_hist_label = ttk.Label(hist_row)
        self._stationary_review_hist_label.pack(side=tk.TOP, anchor="w")

        self._stationary_review_refresh()

    def _build_spotrec_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, textvariable=self._spotrec_spot_var).pack(side=tk.LEFT)
        self._spotrec_start_btn = ttk.Button(top, text="Start recording", command=self._start_spotrec)
        self._spotrec_start_btn.pack(side=tk.LEFT, padx=(10, 0))
        self._spotrec_stop_btn = ttk.Button(top, text="Stop recording", command=self._stop_spotrec)
        self._spotrec_stop_btn.state(["disabled"])
        self._spotrec_stop_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="FPS").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._spotrec_fps_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Exp (ms)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._spotrec_exp_ms_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="ROI size (sensor px)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._spotrec_size_var, width=5).pack(side=tk.LEFT)

        self._spotrec_save_btn = ttk.Button(top, text="Save recording", command=self._spotrec_save)
        self._spotrec_save_btn.state(["disabled"])
        self._spotrec_save_btn.pack(side=tk.LEFT, padx=(12, 0))
        self._spotrec_discard_btn = ttk.Button(top, text="Discard", command=self._spotrec_discard)
        self._spotrec_discard_btn.state(["disabled"])
        self._spotrec_discard_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, textvariable=self._spotrec_status_var).pack(side=tk.RIGHT)
        ttk.Label(top, textvariable=self._spotrec_progress_var).pack(side=tk.RIGHT, padx=(0, 12))

        preview = ttk.Frame(parent, padding=(8, 0, 8, 8))
        preview.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(preview, text="Pick ROI centre").pack(side=tk.LEFT)
        self._spotrec_preview_label = tk.Label(preview, bg="black")
        self._spotrec_preview_label.pack(side=tk.LEFT, padx=(10, 0))
        self._spotrec_preview_label.bind("<Button-1>", self._on_spotrec_preview_click)

        plots = ttk.Frame(parent, padding=8)
        plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.columnconfigure(2, weight=1)
        plots.rowconfigure(0, weight=1)
        plots.rowconfigure(1, weight=1)

        self._spotrec_xy_label = ttk.Label(plots)
        self._spotrec_xy_label.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self._spotrec_phi_label = ttk.Label(plots)
        self._spotrec_phi_label.grid(row=0, column=1, sticky="nsew", padx=(8, 8), pady=(0, 8))

        self._spotrec_fft_label = ttk.Label(plots)
        self._spotrec_fft_label.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=(0, 8))
        dir_frame = ttk.Frame(plots)
        dir_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        ttk.Label(dir_frame, textvariable=self._spotrec_dir_var).pack(side=tk.TOP, anchor="w")
        ttk.Label(
            dir_frame,
            textvariable=self._spotrec_brownian_var,
            wraplength=320,
            justify="left",
        ).pack(side=tk.TOP, anchor="w", pady=(2, 6))
        self._spotrec_dir_label = ttk.Label(dir_frame)
        self._spotrec_dir_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._spotrec_hand_label = ttk.Label(plots)
        self._spotrec_hand_label.grid(row=1, column=1, sticky="nsew", padx=(8, 8), pady=(0, 8))
        ttk.Frame(plots).grid(row=1, column=2, sticky="nsew")

    def _on_tab_changed(self, _event=None) -> None:
        # Keep live feed active for live + spot examine tabs.
        if not hasattr(self, "_notebook"):
            return
        try:
            current = self._notebook.select()
        except Exception:
            return
        if (current != str(getattr(self, "_live_tab", ""))) and (current != str(getattr(self, "_spotrec_tab", ""))):
            self._stop_live_feed()
        if current != str(getattr(self, "_spotrec_tab", "")):
            self._stop_spotrec()
            self._stop_spotrec_preview_loop()
        else:
            if (not self._spotrec_running) and (self._spotrec_proc is None) and (not self._live_running):
                self._start_live_feed()
            self._start_spotrec_preview_loop()

    def _set_selected_center_override(
        self, center: Optional[tuple[float, float]], source: str = "analysis"
    ) -> None:
        if center is None:
            self._selected_center_override = None
            self._selected_center_source = "analysis"
            return
        cx, cy = center
        self._selected_center_override = (float(cx), float(cy))
        self._selected_center_source = str(source)

    def _stationary_local_brightness(
        self, gray_frame: np.ndarray, center: tuple[float, float], size: int = 7
    ) -> float:
        if gray_frame is None or gray_frame.ndim != 2:
            return 0.0
        win = self._extract_window(gray_frame, center[0], center[1], int(size))
        if win is None or win.size == 0:
            return 0.0
        return float(np.mean(win))

    def _stationary_seed_candidates_pre_dog(self) -> list[tuple[float, float]]:
        """
        Build a broad candidate list directly from bright regions in the raw frame,
        i.e. independent of DoG center detection.
        """
        gray = None
        if self._overlay_base_frame is not None and getattr(self._overlay_base_frame, "ndim", 0) == 2:
            gray = self._overlay_base_frame
        elif self.last_frame_gray is not None and getattr(self.last_frame_gray, "ndim", 0) == 2:
            gray = self.last_frame_gray
        if gray is None:
            return []

        work = np.asarray(gray, dtype=np.uint8)
        if work.size == 0:
            return []
        if cv2 is not None:
            try:
                work = cv2.GaussianBlur(
                    work,
                    (0, 0),
                    sigmaX=1.0,
                    sigmaY=1.0,
                    borderType=cv2.BORDER_REPLICATE,
                )
            except Exception:
                pass

        try:
            thr = float(np.percentile(work, float(self.STATIONARY_SEED_BRIGHT_PCT)))
        except Exception:
            return []

        mask = (work >= thr).astype(np.uint8)
        if cv2 is not None:
            try:
                kernel = np.ones((3, 3), dtype=np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            except Exception:
                pass

        h, w = gray.shape
        edge = int(self.EDGE_EXCLUDE_PX)
        candidates: list[tuple[float, float]] = []
        if cv2 is not None:
            try:
                n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
            except Exception:
                n_labels = 0
                stats = None
                centroids = None
            if n_labels > 1 and stats is not None and centroids is not None:
                for i in range(1, int(n_labels)):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < int(self.STATIONARY_SEED_MIN_AREA):
                        continue
                    cx = float(centroids[i, 0])
                    cy = float(centroids[i, 1])
                    if not np.isfinite(cx) or not np.isfinite(cy):
                        continue
                    if edge > 0 and (
                        cx < float(edge)
                        or cy < float(edge)
                        or cx > float(w - 1 - edge)
                        or cy > float(h - 1 - edge)
                    ):
                        continue
                    candidates.append((cx, cy))

        # Fallback: brightest non-overlapping pixels if connected-components was too sparse.
        if not candidates:
            ys, xs = np.where(mask > 0)
            if xs.size > 0:
                vals = work[ys, xs].astype(np.float32)
                order = np.argsort(vals)[::-1]
                min_sep = max(2.0, 0.5 * float(self._spot_window_size))
                min_sep2 = float(min_sep * min_sep)
                for j in order:
                    x = float(xs[j])
                    y = float(ys[j])
                    if edge > 0 and (
                        x < float(edge)
                        or y < float(edge)
                        or x > float(w - 1 - edge)
                        or y > float(h - 1 - edge)
                    ):
                        continue
                    too_close = False
                    for (cx, cy) in candidates:
                        dx = cx - x
                        dy = cy - y
                        if (dx * dx) + (dy * dy) < min_sep2:
                            too_close = True
                            break
                    if too_close:
                        continue
                    candidates.append((x, y))
                    if len(candidates) >= int(self.STATIONARY_MAX_CANDIDATES):
                        break

        if len(candidates) > int(self.STATIONARY_MAX_CANDIDATES):
            scored = [
                (self._stationary_local_brightness(gray, c, size=7), c)
                for c in candidates
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            candidates = [c for _b, c in scored[: int(self.STATIONARY_MAX_CANDIDATES)]]
        return candidates

    def _stationary_iter_gray_frames(self):
        if self.source_kind == "avi":
            with self._gray_lock:
                frames = list(self._gray_frames)
            for g in frames:
                if g is not None and getattr(g, "ndim", 0) == 2:
                    yield g
            return

        if self.source_kind != "npy" or self.npy_frames is None:
            return

        base_dir = Path(self.video_path).parent if self.video_path else None
        if self.npy_has_frames_dim:
            n = int(self._playback_frame_count())
            for i in range(max(0, n)):
                gray = _to_gray_u8(self.npy_frames[i])
                if gray is None:
                    continue
                if self._source_shape is not None and tuple(gray.shape) != tuple(self._source_shape):
                    continue
                yield self._apply_flat_field(gray, base_dir=base_dir)
            return

        gray = _to_gray_u8(self.npy_frames)
        if gray is None:
            return
        if self._source_shape is not None and tuple(gray.shape) != tuple(self._source_shape):
            return
        yield self._apply_flat_field(gray, base_dir=base_dir)

    def _stationary_compute_xy_phi_for_centers(
        self, centers: list[tuple[float, float]]
    ) -> tuple[list[list[tuple[float, float]]], list[list[float]], int]:
        xy_all = [[] for _ in centers]
        phi_all = [[] for _ in centers]
        if not centers:
            return (xy_all, phi_all, 0)

        shape = self._source_shape
        if shape is None:
            if self.last_frame_gray is not None and getattr(self.last_frame_gray, "ndim", 0) == 2:
                shape = self.last_frame_gray.shape
            elif self._overlay_base_frame is not None and getattr(self._overlay_base_frame, "ndim", 0) == 2:
                shape = self._overlay_base_frame.shape
        if shape is None:
            return (xy_all, phi_all, 0)

        h, w = int(shape[0]), int(shape[1])
        ih = h // 2
        iw = w // 2
        win = max(1, int(round(self._spot_window_size / 2.0)))
        if (win % 2) == 0:
            win += 1
        half = win // 2

        bounds: list[tuple[int, int, int, int]] = []
        for cx, cy in centers:
            ix = int(round(float(cx) / 2.0))
            iy = int(round(float(cy) / 2.0))
            x0 = max(0, ix - half)
            x1 = min(iw, ix + half + 1)
            y0 = max(0, iy - half)
            y1 = min(ih, iy + half + 1)
            bounds.append((x0, x1, y0, y1))

        eps = 1e-6
        frames_used = 0
        for gray in self._stationary_iter_gray_frames():
            I0 = gray[0::2, 0::2]
            I45 = gray[0::2, 1::2]
            I135 = gray[1::2, 0::2]
            I90 = gray[1::2, 1::2]

            for i, (x0, x1, y0, y1) in enumerate(bounds):
                a0 = I0[y0:y1, x0:x1]
                a90 = I90[y0:y1, x0:x1]
                a45 = I45[y0:y1, x0:x1]
                a135 = I135[y0:y1, x0:x1]
                m0 = float(a0.mean()) if a0.size else 0.0
                m90 = float(a90.mean()) if a90.size else 0.0
                m45 = float(a45.mean()) if a45.size else 0.0
                m135 = float(a135.mean()) if a135.size else 0.0
                x = (m0 - m90) / (m0 + m90 + eps)
                y = (m45 - m135) / (m45 + m135 + eps)
                xy_all[i].append((float(x), float(y)))
                phi_all[i].append(float(0.5 * np.arctan2(y, x)))
            frames_used += 1
        return (xy_all, phi_all, frames_used)

    def _stationary_series_metrics(self, series: list[tuple[float, float]]) -> Optional[dict]:
        if not series:
            return None
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
            return None
        x = arr[:, 0]
        y = arr[:, 1]
        r = np.sqrt((x * x) + (y * y))
        return {
            "n_frames": int(arr.shape[0]),
            "r_mean": float(np.mean(r)),
            "r_std": float(np.std(r)),
            "motion": float(self._spot_xy_max_axis_range(series)),
        }

    def _refresh_stationary_candidates(self, preserve_selection: bool = True) -> None:
        bright_min = self._parse_float(self._stationary_brightness_min_var.get())
        r_min = self._parse_float(self._stationary_r_min_var.get())
        motion_max = self._parse_float(self._stationary_motion_max_var.get())
        if bright_min is None or r_min is None or motion_max is None:
            messagebox.showerror(
                "Stationary rods",
                "Brightness / mean r / max XY range must all be numeric.",
            )
            return
        if bright_min < 0.0:
            messagebox.showerror("Stationary rods", "Min brightness must be >= 0.")
            return
        if r_min < 0.0:
            messagebox.showerror("Stationary rods", "Min mean r must be >= 0.")
            return
        if motion_max < 0.0:
            messagebox.showerror("Stationary rods", "Max XY range must be >= 0.")
            return

        seed_centers = self._stationary_seed_candidates_pre_dog()
        frame_ref = self._overlay_base_frame
        if frame_ref is None or getattr(frame_ref, "ndim", 0) != 2:
            frame_ref = self.last_frame_gray
        if frame_ref is None or getattr(frame_ref, "ndim", 0) != 2:
            frame_ref = np.zeros((1, 1), dtype=np.uint8)

        bright_centers: list[tuple[float, float]] = []
        bright_vals: list[float] = []
        for center in seed_centers:
            b = self._stationary_local_brightness(frame_ref, center, size=7)
            if float(b) >= float(bright_min):
                bright_centers.append(center)
                bright_vals.append(float(b))

        xy_all, phi_all, frames_used = self._stationary_compute_xy_phi_for_centers(bright_centers)

        candidates = []
        keep_r = 0
        for i, center in enumerate(bright_centers):
            xy_series = list(xy_all[i]) if i < len(xy_all) else []
            m = self._stationary_series_metrics(xy_series)
            if m is None:
                continue
            if int(m["n_frames"]) < int(self.STATIONARY_MIN_FRAMES):
                continue
            if float(m["r_mean"]) < float(r_min):
                continue
            keep_r += 1
            if float(m["motion"]) > float(motion_max):
                continue
            candidates.append(
                {
                    "center": center,
                    "brightness": float(bright_vals[i]) if i < len(bright_vals) else 0.0,
                    "xy": xy_series,
                    "phi": list(phi_all[i]) if i < len(phi_all) else [],
                    "r_mean": float(m["r_mean"]),
                    "r_std": float(m["r_std"]),
                    "motion": float(m["motion"]),
                    "n_frames": int(m["n_frames"]),
                }
            )

        candidates.sort(
            key=lambda c: (
                float(c["motion"]),
                -float(c["r_mean"]),
                -float(c["brightness"]),
            )
        )
        self._stationary_candidates = candidates
        self._stationary_idx = 0

        if hasattr(self, "bottom_var"):
            self.bottom_var.set(
                f"Stationary pipeline: seeds {len(seed_centers)}, bright {len(bright_centers)}, "
                f"r {keep_r}, final {len(candidates)} (frames={frames_used})."
            )

        if not candidates:
            if self._selected_center_source == "stationary":
                self._set_selected_center_override(None, source="analysis")
            self._update_stationary_view()
            self._update_spotrec_label()
            self._spotrec_update_preview(gray_frame=self._live_last_frame)
            return

        target_idx = 0
        if preserve_selection and self._selected_center_source == "stationary":
            ov = self._selected_center_override
            if ov is not None:
                ov_key = self._spot_center_key(ov)
                for i, c in enumerate(candidates):
                    if self._spot_center_key(c["center"]) == ov_key:
                        target_idx = i
                        break

        self._select_stationary_spot(target_idx)

    def _update_stationary_view(self) -> None:
        n = len(self._stationary_candidates)
        if n <= 0:
            self._stationary_status_var.set("Stationary 0 / 0")
            self._stationary_metrics_var.set("No stationary candidates found.")
            if self._stationary_prev_btn is not None:
                self._stationary_prev_btn.configure(state=tk.DISABLED)
            if self._stationary_next_btn is not None:
                self._stationary_next_btn.configure(state=tk.DISABLED)
            if self._stationary_spot_img_label is not None:
                self._stationary_spot_img_label.configure(image="")
            if self._stationary_xy_label is not None:
                self._stationary_xy_label.configure(image="")
            if self._stationary_phi_label is not None:
                self._stationary_phi_label.configure(image="")
            self._stationary_spot_ref = None
            self._stationary_xy_ref = None
            self._stationary_phi_ref = None
            return

        self._stationary_idx = max(0, min(int(self._stationary_idx), n - 1))
        c = self._stationary_candidates[self._stationary_idx]
        self._stationary_status_var.set(f"Stationary {self._stationary_idx + 1} / {n}")
        self._stationary_metrics_var.set(
            f"bright={float(c.get('brightness', 0.0)):.1f}  "
            f"mean r={float(c['r_mean']):.3f}  std(r)={float(c['r_std']):.3f}  "
            f"max range XY={float(c['motion']):.3f}  frames={int(c['n_frames'])}"
        )
        if self._stationary_prev_btn is not None:
            self._stationary_prev_btn.configure(state=tk.NORMAL)
        if self._stationary_next_btn is not None:
            self._stationary_next_btn.configure(state=tk.NORMAL)

        if self._s_map is not None and self._stationary_spot_img_label is not None:
            cx, cy = c["center"]
            win = self._extract_window(self._s_map, cx, cy, int(self._spot_window_size))
            win_u8 = detect_spinners.to_u8_preview(win, lo_pct=0.0, hi_pct=100.0)
            size_px = int(self._spot_window_size) * int(self._spot_scale)
            img = Image.fromarray(win_u8).resize((size_px, size_px), resample=Image.NEAREST).convert("RGB")
            img_tk = ImageTk.PhotoImage(img)
            self._stationary_spot_img_label.configure(image=img_tk)
            self._stationary_spot_ref = img_tk
        elif self._stationary_spot_img_label is not None:
            self._stationary_spot_img_label.configure(image="")
            self._stationary_spot_ref = None

        if self._stationary_xy_label is not None:
            xy_img = self._make_xy_scatter_image(c["xy"])
            xy_tk = ImageTk.PhotoImage(xy_img)
            self._stationary_xy_label.configure(image=xy_tk)
            self._stationary_xy_ref = xy_tk

        if self._stationary_phi_label is not None:
            phi_img = self._make_phi_plot_image(c["phi"], self.source_fps)
            phi_tk = ImageTk.PhotoImage(phi_img)
            self._stationary_phi_label.configure(image=phi_tk)
            self._stationary_phi_ref = phi_tk

    def _select_stationary_spot(self, idx: int) -> None:
        if not self._stationary_candidates:
            self._update_stationary_view()
            return
        self._stationary_idx = max(0, min(int(idx), len(self._stationary_candidates) - 1))
        c = self._stationary_candidates[self._stationary_idx]
        center = c["center"]
        self._set_selected_center_override(center, source="stationary")

        # If this spot is present in the currently filtered analysis list,
        # keep both views aligned on the same index.
        key = self._spot_center_key(center)
        hit_idx = None
        for i, pt in enumerate(self._spot_centers):
            if self._spot_center_key(pt) == key:
                hit_idx = i
                break
        if hit_idx is not None:
            self._spot_idx = int(hit_idx)
            self._update_spot_view()
            self._rebuild_smap_overlay()

        self._update_stationary_view()
        self._update_spotrec_label()
        self._spotrec_update_preview(gray_frame=self._live_last_frame)

    def _prev_stationary_spot(self) -> None:
        if not self._stationary_candidates:
            return
        self._select_stationary_spot((int(self._stationary_idx) - 1) % len(self._stationary_candidates))

    def _next_stationary_spot(self) -> None:
        if not self._stationary_candidates:
            return
        self._select_stationary_spot((int(self._stationary_idx) + 1) % len(self._stationary_candidates))

    def _recordings_day_dir(self) -> Path:
        day = time.strftime("%Y-%m-%d")
        out = Path.cwd() / self.RECORDINGS_ROOT_DIRNAME / day
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _recordings_subdir(self, kind: str) -> Path:
        out = self._recordings_day_dir() / str(kind)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _stationary_dataset_paths(self) -> tuple[Path, Path, Path]:
        root = Path.cwd() / self.STATIONARY_DATASET_DIRNAME
        pending = root / self.STATIONARY_DATASET_PENDING_DIR
        good = root / self.STATIONARY_DATASET_GOOD_DIR
        pending.mkdir(parents=True, exist_ok=True)
        good.mkdir(parents=True, exist_ok=True)
        return root, pending, good

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(payload, indent=2, ensure_ascii=True)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _theta_from_model(self, r_value: float, a: float, b: float, c: float) -> Optional[float]:
        try:
            r = float(r_value)
            den = float(b) - (float(c) * r)
            if (not np.isfinite(r)) or (not np.isfinite(den)) or den <= 0.0:
                return None
            ratio = (float(a) * r) / den
            if (not np.isfinite(ratio)) or ratio < 0.0 or ratio > 1.0:
                return None
            th = float(np.arcsin(np.sqrt(ratio)))
            if not np.isfinite(th):
                return None
            return th
        except Exception:
            return None

    def _theta_models_from_r(self, r_value: float) -> dict:
        th_nohole = self._theta_from_model(
            r_value, a=0.1895779531, b=0.6256149990, c=0.4867530374
        )
        th_hole = self._theta_from_model(
            r_value, a=0.1865937176, b=0.5576753053, c=0.4215514426
        )
        out = {
            "theta_nohole_rad": th_nohole,
            "theta_nohole_deg": (float(np.degrees(th_nohole)) if th_nohole is not None else None),
            "theta_hole_fresnel_rad": th_hole,
            "theta_hole_fresnel_deg": (float(np.degrees(th_hole)) if th_hole is not None else None),
        }
        return out

    def _stationary_xy_phi_from_stack(
        self, arr: np.ndarray, roi_meta: dict
    ) -> tuple[list[tuple[float, float]], list[float]]:
        xy_series: list[tuple[float, float]] = []
        phi_series: list[float] = []
        if arr is None or arr.ndim < 3:
            return (xy_series, phi_series)
        total = int(arr.shape[0])
        if total <= 0:
            return (xy_series, phi_series)

        roi_raw = int(roi_meta.get("win_raw", int(roi_meta.get("w", 11))))
        roi_raw = max(1, int(roi_raw))
        if (roi_raw % 2) == 0:
            roi_raw += 1
        for i in range(total):
            try:
                xv, yv, phi = self._xy_phi_from_frame(gray=np.asarray(arr[i]), roi_meta=roi_meta, win_raw=roi_raw)
            except Exception:
                continue
            xy_series.append((float(xv), float(yv)))
            phi_series.append(float(phi))
        return (xy_series, phi_series)

    def _xy_phi_from_frame(self, gray: np.ndarray, roi_meta: Optional[dict], win_raw: int) -> tuple[float, float, float]:
        # Shared reduction path used by spot inspection and stationary-capture post-analysis.
        if gray.ndim != 2:
            g = gray[..., 0]
        else:
            g = gray

        px = 0
        py = 0
        if roi_meta is not None:
            try:
                px = int(roi_meta.get("phase_x", int(roi_meta.get("x", 0)) % 2)) % 2
                py = int(roi_meta.get("phase_y", int(roi_meta.get("y", 0)) % 2)) % 2
            except Exception:
                px = 0
                py = 0

        I0 = g[py::2, px::2]
        I45 = g[py::2, (1 - px) :: 2]
        I135 = g[(1 - py) :: 2, px::2]
        I90 = g[(1 - py) :: 2, (1 - px) :: 2]

        ih, iw = I0.shape
        win_raw = max(1, int(win_raw))
        if (win_raw % 2) == 0:
            win_raw += 1
        win = max(1, int(round(win_raw / 2.0)))
        if win % 2 == 0:
            win += 1
        half = win // 2

        cx = iw // 2
        cy = ih // 2
        if roi_meta is not None:
            try:
                cx_rel = float(roi_meta["cx"]) - float(roi_meta["x"])
                cy_rel = float(roi_meta["cy"]) - float(roi_meta["y"])
                cx = int(round(cx_rel / 2.0))
                cy = int(round(cy_rel / 2.0))
            except Exception:
                pass

        x0 = max(0, cx - half)
        x1 = min(iw, cx + half + 1)
        y0 = max(0, cy - half)
        y1 = min(ih, cy + half + 1)

        eps = 1e-6
        a0 = I0[y0:y1, x0:x1]
        a90 = I90[y0:y1, x0:x1]
        a45 = I45[y0:y1, x0:x1]
        a135 = I135[y0:y1, x0:x1]

        m0 = float(a0.mean()) if a0.size else 0.0
        m90 = float(a90.mean()) if a90.size else 0.0
        m45 = float(a45.mean()) if a45.size else 0.0
        m135 = float(a135.mean()) if a135.size else 0.0
        x = (m0 - m90) / (m0 + m90 + eps)
        y = (m45 - m135) / (m45 + m135 + eps)
        phi = float(0.5 * np.arctan2(y, x))
        return float(x), float(y), phi

    def _stationary_series_full_metrics(self, series: list[tuple[float, float]]) -> dict:
        if not series:
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
        arr = np.asarray(series, dtype=np.float32)
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

    def _roi_from_target_center(self, center: tuple[float, float], roi_raw: int) -> dict:
        # Shared camera-ROI placement rule used by Spot inspection and stationary capture.
        cx, cy = center
        roi_raw = max(1, int(roi_raw))
        if (roi_raw % 2) == 0:
            roi_raw += 1
        w_cam = int(roi_raw)
        h_cam = int(roi_raw)
        x = int(round(float(cx))) - (w_cam // 2)
        y = int(round(float(cy))) - (h_cam // 2)
        if (x % 2) != 0:
            x -= 1
        if (y % 2) != 0:
            y -= 1
        x = max(0, x)
        y = max(0, y)
        return {
            "x": int(x),
            "y": int(y),
            "w": int(w_cam),
            "h": int(h_cam),
            "roi_raw": int(roi_raw),
            "cx": float(cx),
            "cy": float(cy),
            "phase_x": int(x) % 2,
            "phase_y": int(y) % 2,
        }

    def _capture_stationary_mode(
        self,
        center: tuple[float, float],
        out_path: Path,
        roi_raw: int,
        n_frames: int,
        exp_ms: Optional[float],
        requested_fps: Optional[float],
        mode_name: str,
        requested_duration_s: Optional[float],
    ) -> dict:
        roi_req = self._roi_from_target_center(center=center, roi_raw=int(roi_raw))
        cx = float(roi_req["cx"])
        cy = float(roi_req["cy"])
        x = int(roi_req["x"])
        y = int(roi_req["y"])
        w_cam = int(roi_req["w"])
        h_cam = int(roi_req["h"])
        roi_raw = int(roi_req["roi_raw"])
        n_frames = max(1, int(n_frames))
        script = Path(__file__).resolve().parent / "fetch_frames.py"

        def _run_fetch(req_fps: Optional[float]) -> tuple[Path, Optional[float], object]:
            args = [
                sys.executable,
                str(script),
                "--out-dir",
                str(out_path.parent),
                "--out-path",
                str(out_path),
                "--roi",
                str(x),
                str(y),
                str(w_cam),
                str(h_cam),
                "--json",
                "--n-frames",
                str(n_frames),
                "--stop-after",
                str(n_frames),
            ]
            if req_fps is not None and float(req_fps) > 0.0:
                args.extend(["--fps", str(float(req_fps))])
            if exp_ms is not None and float(exp_ms) > 0.0:
                args.extend(["--exp-ms", str(float(exp_ms))])
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(err or f"{mode_name} capture failed.")
            payload = (proc.stdout or "").strip().splitlines()
            if not payload:
                raise RuntimeError(f"{mode_name} recorder produced no output.")
            try:
                data = json.loads(payload[-1])
                path_l = Path(str(data.get("path", "")))
                actual_fps_l = data.get("actual_fps")
                actual_fps_l = float(actual_fps_l) if actual_fps_l is not None else None
                roi_l = data.get("roi")
                return path_l, actual_fps_l, roi_l
            except Exception as e:
                raise RuntimeError(f"{mode_name} output parse failed: {e}")

        used_fps_fallback = False
        try:
            path, actual_fps, roi = _run_fetch(requested_fps)
        except Exception:
            if requested_fps is None or float(requested_fps) <= 0.0:
                raise
            # If requested fps is too high for this ROI/exposure, retry in auto/max-fps mode.
            path, actual_fps, roi = _run_fetch(None)
            used_fps_fallback = True

        if not path.exists():
            raise RuntimeError(f"{mode_name} recording file missing.")
        try:
            if path.resolve() != out_path.resolve():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, out_path)
                path = out_path
        except Exception:
            # Best effort: keep using the produced path if move/replace fails.
            pass

        arr = np.load(path, allow_pickle=False)
        total = int(arr.shape[0]) if arr.ndim >= 3 else 0
        if total <= 0:
            if (not used_fps_fallback) and requested_fps is not None and float(requested_fps) > 0.0:
                path, actual_fps, roi = _run_fetch(None)
                used_fps_fallback = True
                arr = np.load(path, allow_pickle=False)
                total = int(arr.shape[0]) if arr.ndim >= 3 else 0
            if total <= 0:
                raise RuntimeError(f"{mode_name} capture produced 0 frames.")
        roi_meta = {
            "x": int(x),
            "y": int(y),
            "w": int(w_cam),
            "h": int(h_cam),
            "cx": float(cx),
            "cy": float(cy),
            "win_raw": int(roi_raw),
            "phase_x": int(x) % 2,
            "phase_y": int(y) % 2,
        }
        if isinstance(roi, dict):
            try:
                rx = roi.get("OffsetX", roi.get("x"))
                ry = roi.get("OffsetY", roi.get("y"))
                rw = roi.get("Width", roi.get("w"))
                rh = roi.get("Height", roi.get("h"))
                if rx is not None:
                    roi_meta["x"] = int(round(float(rx)))
                if ry is not None:
                    roi_meta["y"] = int(round(float(ry)))
                if rw is not None:
                    roi_meta["w"] = int(round(float(rw)))
                if rh is not None:
                    roi_meta["h"] = int(round(float(rh)))
                roi_meta["phase_x"] = int(roi_meta.get("x", 0)) % 2
                roi_meta["phase_y"] = int(roi_meta.get("y", 0)) % 2
            except Exception:
                pass

        xy_series, phi_series = self._stationary_xy_phi_from_stack(arr, roi_meta)
        stats = self._stationary_series_full_metrics(xy_series)
        theta_stats = self._theta_models_from_r(float(stats.get("r_mean", 0.0)))
        fps_for_duration = (
            float(actual_fps)
            if actual_fps is not None and actual_fps > 0.0
            else (float(requested_fps) if requested_fps is not None and requested_fps > 0.0 else None)
        )
        duration_s = (
            (float(stats["n_frames"]) / float(fps_for_duration))
            if fps_for_duration is not None and fps_for_duration > 0.0
            else None
        )

        mode_meta = {
            "mode_name": mode_name,
            "npy_file": path.name,
            "center_px": {"x": float(cx), "y": float(cy)},
            "requested": {
                "fps": (float(requested_fps) if requested_fps is not None else None),
                "exp_ms": (float(exp_ms) if exp_ms is not None else None),
                "duration_s": (
                    float(requested_duration_s) if requested_duration_s is not None else None
                ),
                "target_frames": int(n_frames),
                "fov_raw_px": int(roi_raw),
            },
            "actual": {
                "fps": (float(actual_fps) if actual_fps is not None else None),
                "fps_request_fallback_to_auto": bool(used_fps_fallback),
                "frames": int(stats["n_frames"]),
                "duration_s": (float(duration_s) if duration_s is not None else None),
                "roi": roi_meta,
            },
            "xy_metrics": stats,
            "theta_estimate": theta_stats,
            "xy_series": [[float(a), float(b)] for (a, b) in xy_series],
            "phi_series": [float(v) for v in phi_series],
        }
        meta_path = path.with_name(path.stem + "_meta.json")
        self._write_json_atomic(meta_path, mode_meta)
        return mode_meta

    def _stationary_capture_parse_config(self) -> Optional[dict]:
        fps77 = self._parse_float(self._stationary_capture_77_fps_var.get())
        exp77 = self._parse_float(self._stationary_capture_77_exp_ms_var.get())
        dur77 = self._parse_float(self._stationary_capture_77_duration_s_var.get())
        roi77 = self._parse_int(self._stationary_capture_77_roi_var.get())
        exp_max = self._parse_float(self._stationary_capture_max_exp_ms_var.get())
        dur_max = self._parse_float(self._stationary_capture_max_duration_s_var.get())
        max_fps_est = self._parse_float(self._stationary_capture_max_fps_est_var.get())

        if fps77 is None or fps77 <= 0.0:
            messagebox.showerror("Stationary capture", "77fps-mode FPS must be > 0.")
            return None
        if exp77 is None or exp77 <= 0.0:
            messagebox.showerror("Stationary capture", "77fps-mode exposure must be > 0 ms.")
            return None
        if dur77 is None or dur77 <= 0.0:
            messagebox.showerror("Stationary capture", "77fps-mode duration must be > 0 s.")
            return None
        if roi77 is None or roi77 < 3:
            messagebox.showerror("Stationary capture", "77fps-mode FOV must be an odd integer >= 3.")
            return None
        if (int(roi77) % 2) == 0:
            roi77 = int(roi77) + 1
            self._stationary_capture_77_roi_var.set(str(int(roi77)))

        if exp_max is None or exp_max <= 0.0:
            messagebox.showerror("Stationary capture", "Max-fps-mode exposure must be > 0 ms.")
            return None
        if dur_max is None or dur_max <= 0.0:
            messagebox.showerror("Stationary capture", "Max-fps-mode duration must be > 0 s.")
            return None
        if max_fps_est is None or max_fps_est <= 0.0:
            messagebox.showerror("Stationary capture", "Max-fps estimate must be > 0.")
            return None

        n77 = max(1, int(round(float(dur77) * float(fps77))))
        nmax = max(1, int(round(float(dur_max) * float(max_fps_est))))
        return {
            "fps77": float(fps77),
            "exp77": float(exp77),
            "dur77": float(dur77),
            "roi77": int(roi77),
            "n77": int(n77),
            "exp_max": float(exp_max),
            "dur_max": float(dur_max),
            "max_fps_est": float(max_fps_est),
            "nmax": int(nmax),
            "roi_max": int(self.STATIONARY_REC_MAX_ROI_RAW),
        }

    def _set_stationary_capture_busy(self, busy: bool) -> None:
        if self._stationary_capture_selected_btn is not None:
            self._stationary_capture_selected_btn.configure(
                state=(tk.DISABLED if busy else tk.NORMAL)
            )
        if self._stationary_capture_all_btn is not None:
            self._stationary_capture_all_btn.configure(
                state=(tk.DISABLED if busy else tk.NORMAL)
            )

    def _stationary_capture_selected(self) -> None:
        if not self._stationary_candidates:
            messagebox.showerror("Stationary capture", "No stationary rod selected.")
            return
        idx = max(0, min(int(self._stationary_idx), len(self._stationary_candidates) - 1))
        target = dict(self._stationary_candidates[idx])
        cfg = self._stationary_capture_parse_config()
        if cfg is None:
            return
        self._stationary_capture_targets([target], cfg, capture_label="selected")

    def _stationary_capture_all(self) -> None:
        if not self._stationary_candidates:
            messagebox.showerror("Stationary capture", "No stationary rods available.")
            return
        max_fps_est = self._parse_float(self._stationary_capture_max_fps_est_var.get())
        if max_fps_est is None or max_fps_est <= 0.0:
            messagebox.showerror(
                "Stationary capture", "Max-fps estimate must be > 0 for Record all stationary."
            )
            return
        n77 = max(1, int(round(self.STATIONARY_REC_ALL_77_DURATION_S * self.STATIONARY_REC_ALL_77_FPS)))
        nmax = max(1, int(round(self.STATIONARY_REC_ALL_MAX_DURATION_S * float(max_fps_est))))
        cfg = {
            "fps77": float(self.STATIONARY_REC_ALL_77_FPS),
            "exp77": float(self.STATIONARY_REC_ALL_77_EXP_MS),
            "dur77": float(self.STATIONARY_REC_ALL_77_DURATION_S),
            "roi77": int(self.STATIONARY_REC_ALL_77_ROI_RAW),
            "n77": int(n77),
            "exp_max": float(self.STATIONARY_REC_ALL_MAX_EXP_MS),
            "dur_max": float(self.STATIONARY_REC_ALL_MAX_DURATION_S),
            "max_fps_est": float(max_fps_est),
            "nmax": int(nmax),
            "roi_max": int(self.STATIONARY_REC_ALL_MAX_ROI_RAW),
        }
        targets = [dict(c) for c in self._stationary_candidates]
        self._stationary_capture_targets(targets, cfg, capture_label="all")

    def _stationary_capture_targets(
        self, targets: list[dict], cfg: dict, capture_label: str
    ) -> None:
        with self._stationary_capture_lock:
            if self._stationary_capture_running:
                messagebox.showinfo("Stationary capture", "A stationary capture run is already active.")
                return
            self._stationary_capture_running = True
        if self._spotrec_running or self._spotrec_proc is not None:
            with self._stationary_capture_lock:
                self._stationary_capture_running = False
            messagebox.showerror("Stationary capture", "Stop Spot examine recording before capture.")
            return
        if self._live_running:
            self._stop_live_feed()

        self._set_stationary_capture_busy(True)
        total = len(targets)
        self._stationary_capture_status_var.set(
            f"Starting stationary capture ({capture_label}) for {total} rod(s)..."
        )

        def _worker():
            n_ok = 0
            n_fail = 0
            try:
                _, pending_dir, _ = self._stationary_dataset_paths()
                for i, cand in enumerate(targets, start=1):
                    center = cand.get("center")
                    if (
                        center is None
                        or not isinstance(center, (tuple, list))
                        or len(center) < 2
                    ):
                        n_fail += 1
                        continue
                    cx = float(center[0])
                    cy = float(center[1])
                    off_x, off_y = self._spotrec_center_offset
                    cx_cap = float(cx) + float(off_x)
                    cy_cap = float(cy) + float(off_y)
                    key = self._spot_center_key((cx_cap, cy_cap))
                    token = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}"
                    rod_id = f"rod_x{key[0]}_y{key[1]}_{token}"
                    rod_dir = pending_dir / rod_id
                    rod_dir.mkdir(parents=True, exist_ok=True)

                    self._ui_call(
                        self._stationary_capture_status_var.set,
                        f"Capturing rod {i}/{total}: {rod_id}",
                    )
                    mode77_path = rod_dir / "capture_77fps.npy"
                    mode_max_path = rod_dir / "capture_maxfps_11x11.npy"
                    try:
                        mode77 = self._capture_stationary_mode(
                            center=(cx_cap, cy_cap),
                            out_path=mode77_path,
                            roi_raw=int(cfg["roi77"]),
                            n_frames=int(cfg["n77"]),
                            exp_ms=float(cfg["exp77"]),
                            requested_fps=float(cfg["fps77"]),
                            mode_name="77fps",
                            requested_duration_s=float(cfg["dur77"]),
                        )
                        # Match Spot inspection behavior exactly:
                        # each recording mode derives ROI directly from the same target center.
                        cx_max = float(cx_cap)
                        cy_max = float(cy_cap)
                        mode_max = self._capture_stationary_mode(
                            center=(cx_max, cy_max),
                            out_path=mode_max_path,
                            roi_raw=int(cfg["roi_max"]),
                            n_frames=int(cfg["nmax"]),
                            exp_ms=float(cfg["exp_max"]),
                            # Request high fps; camera/controller should clamp to max achievable.
                            requested_fps=float(cfg["max_fps_est"]),
                            mode_name="maxfps_11x11",
                            requested_duration_s=float(cfg["dur_max"]),
                        )
                        theta_nohole_deg = mode77.get("theta_estimate", {}).get("theta_nohole_deg")
                        theta_hole_deg = mode77.get("theta_estimate", {}).get("theta_hole_fresnel_deg")
                        rod_meta = {
                            "rod_id": rod_id,
                            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_path": str(self.video_path) if self.video_path else None,
                            "capture_type": "stationary_rod_dual_mode",
                            "center_px": {"x": float(cx), "y": float(cy)},
                            "capture_center_px": {
                                "x": float(cx_cap),
                                "y": float(cy_cap),
                                "spotrec_offset_x": float(off_x),
                                "spotrec_offset_y": float(off_y),
                            },
                            "candidate_metrics": {
                                "brightness": float(cand.get("brightness", 0.0)),
                                "r_mean": float(cand.get("r_mean", 0.0)),
                                "r_std": float(cand.get("r_std", 0.0)),
                                "max_range_xy": float(cand.get("motion", 0.0)),
                                "n_frames": int(cand.get("n_frames", 0)),
                            },
                            "modes": {
                                "capture_77fps_meta": "capture_77fps_meta.json",
                                "capture_maxfps_11x11_meta": "capture_maxfps_11x11_meta.json",
                                "capture_77fps_summary": {
                                    "actual_fps": mode77.get("actual", {}).get("fps"),
                                    "frames": mode77.get("actual", {}).get("frames"),
                                    "duration_s": mode77.get("actual", {}).get("duration_s"),
                                    "fov_raw_px": mode77.get("requested", {}).get("fov_raw_px"),
                                    "r_mean": mode77.get("xy_metrics", {}).get("r_mean"),
                                    "range_x": mode77.get("xy_metrics", {}).get("range_x"),
                                    "range_y": mode77.get("xy_metrics", {}).get("range_y"),
                                },
                                "capture_maxfps_11x11_summary": {
                                    "actual_fps": mode_max.get("actual", {}).get("fps"),
                                    "frames": mode_max.get("actual", {}).get("frames"),
                                    "duration_s": mode_max.get("actual", {}).get("duration_s"),
                                    "fov_raw_px": mode_max.get("requested", {}).get("fov_raw_px"),
                                    "r_mean": mode_max.get("xy_metrics", {}).get("r_mean"),
                                    "range_x": mode_max.get("xy_metrics", {}).get("range_x"),
                                    "range_y": mode_max.get("xy_metrics", {}).get("range_y"),
                                },
                            },
                            "theta_nohole_deg": theta_nohole_deg,
                            "theta_hole_fresnel_deg": theta_hole_deg,
                        }
                        self._write_json_atomic(rod_dir / "meta.json", rod_meta)
                        n_ok += 1
                        self._ui_call(
                            self._stationary_capture_status_var.set,
                            f"Captured rod {i}/{total}: theta(nohole)={theta_nohole_deg}",
                        )
                    except Exception as e:
                        n_fail += 1
                        err_meta = {
                            "rod_id": rod_id,
                            "center_px": {"x": float(cx), "y": float(cy)},
                            "error": str(e),
                            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        try:
                            self._write_json_atomic(rod_dir / "error.json", err_meta)
                        except Exception:
                            pass
                        self._ui_call(
                            self._stationary_capture_status_var.set,
                            f"Capture failed for rod {i}/{total}: {e}",
                        )
            finally:
                msg = f"Stationary capture finished: {n_ok} saved, {n_fail} failed."
                self._ui_call(self._stationary_capture_status_var.set, msg)
                self._ui_call(self.bottom_var.set, msg)
                self._ui_call(self._stationary_review_refresh, True)
                self._ui_call(self._set_stationary_capture_busy, False)
                with self._stationary_capture_lock:
                    self._stationary_capture_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _stationary_review_load_json(self, path: Path) -> Optional[dict]:
        try:
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _stationary_review_load_mode_meta(
        self, rod_dir: Path, key: str, fallback_name: str
    ) -> Optional[dict]:
        root_meta = self._stationary_review_load_json(rod_dir / "meta.json") or {}
        modes = root_meta.get("modes", {})
        meta_name = None
        if isinstance(modes, dict):
            meta_name = modes.get(key)
        if not meta_name:
            meta_name = fallback_name
        return self._stationary_review_load_json(rod_dir / str(meta_name))

    def _stationary_review_refresh(self, preserve_selection: bool = False) -> None:
        _, pending_dir, good_dir = self._stationary_dataset_paths()
        pending_dirs = sorted([p for p in pending_dir.iterdir() if p.is_dir()])
        good_dirs = sorted([p for p in good_dir.iterdir() if p.is_dir()])
        prev_name = None
        prev_state = None
        if preserve_selection and self._stationary_review_items:
            idx = max(
                0,
                min(int(self._stationary_review_idx), len(self._stationary_review_items) - 1),
            )
            prev_state, prev_dir = self._stationary_review_items[idx]
            prev_name = prev_dir.name
        # Review pane shows only pending rods; accepted ("good") rods are hidden.
        items: list[tuple[str, Path]] = [("pending", p) for p in pending_dirs]
        self._stationary_review_items = items
        self._stationary_review_status_var.set(
            f"Pending {len(pending_dirs)} | Good {len(good_dirs)} (hidden)"
        )
        if prev_name:
            hit = None
            for i, (state, p) in enumerate(items):
                if p.name == prev_name and state == prev_state:
                    hit = i
                    break
            if hit is None:
                for i, (_, p) in enumerate(items):
                    if p.name == prev_name:
                        hit = i
                        break
            if hit is not None:
                self._stationary_review_idx = int(hit)
        self._stationary_review_update_theta_panel(good_dirs)
        self._stationary_review_update_view()

    def _stationary_review_update_theta_panel(self, good_dirs: list[Path]) -> None:
        theta_vals: list[float] = []
        for d in good_dirs:
            meta = self._stationary_review_load_json(d / "meta.json") or {}
            t = meta.get("theta_nohole_deg")
            try:
                if t is not None:
                    t_f = float(t)
                    if np.isfinite(t_f):
                        theta_vals.append(t_f)
            except Exception:
                continue

        if theta_vals:
            arr = np.asarray(theta_vals, dtype=np.float64)
            bins = np.arange(0.0, 100.0, 10.0)
            hist, edges = np.histogram(arr, bins=bins)
            parts = []
            for i in range(len(hist)):
                a = int(round(edges[i]))
                b = int(round(edges[i + 1]))
                parts.append(f"{a}-{b}:{int(hist[i])}")
            self._stationary_review_theta_var.set(
                f"Accepted theta(nohole, deg): n={arr.size}  " + " | ".join(parts)
            )
        else:
            self._stationary_review_theta_var.set("Accepted theta(nohole, deg): n=0")

        img = self._stationary_review_theta_image(theta_vals)
        if self._stationary_review_hist_label is not None:
            photo = ImageTk.PhotoImage(img)
            self._stationary_review_hist_label.configure(image=photo)
            self._stationary_review_hist_ref = photo

    def _stationary_review_theta_image(self, theta_deg: list[float]) -> Image.Image:
        if Figure is not None and FigureCanvas is not None:
            fig = Figure(figsize=(3.2, 2.0), dpi=100)
            ax = fig.add_subplot(111)
            if theta_deg:
                bins = np.arange(0.0, 100.0, 10.0)
                ax.hist(theta_deg, bins=bins, color="tab:blue", alpha=0.85, edgecolor="black")
            else:
                ax.text(0.5, 0.5, "No accepted rods yet", ha="center", va="center", fontsize=9)
                ax.set_xlim(0.0, 90.0)
            ax.set_xlabel("theta (deg)", fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.tick_params(labelsize=8)
            fig.tight_layout(pad=0.5)
            return self._mpl_fig_to_image(fig)

        w, h = 320, 200
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w - 1, h - 1), outline=(0, 0, 0))
        if not theta_deg:
            draw.text((12, 14), "No accepted rods yet", fill=(0, 0, 0))
            return img
        draw.text((12, 14), f"n={len(theta_deg)} theta values", fill=(0, 0, 0))
        bins = np.arange(0.0, 100.0, 10.0)
        hist, _ = np.histogram(np.asarray(theta_deg, dtype=np.float64), bins=bins)
        y = 36
        for i, c in enumerate(hist):
            draw.text((12, y), f"{int(bins[i]):02d}-{int(bins[i+1]):02d}: {int(c)}", fill=(0, 0, 0))
            y += 14
        return img

    def _stationary_review_prev(self) -> None:
        if not self._stationary_review_items:
            return
        n = len(self._stationary_review_items)
        self._stationary_review_idx = (int(self._stationary_review_idx) - 1) % n
        self._stationary_review_update_view()

    def _stationary_review_next(self) -> None:
        if not self._stationary_review_items:
            return
        n = len(self._stationary_review_items)
        self._stationary_review_idx = (int(self._stationary_review_idx) + 1) % n
        self._stationary_review_update_view()

    def _stationary_review_update_view(self) -> None:
        n = len(self._stationary_review_items)
        if n <= 0:
            self._stationary_review_detail_var.set("No stationary recordings.")
            if self._stationary_review_prev_btn is not None:
                self._stationary_review_prev_btn.configure(state=tk.DISABLED)
            if self._stationary_review_next_btn is not None:
                self._stationary_review_next_btn.configure(state=tk.DISABLED)
            if self._stationary_review_mark_chk is not None:
                self._stationary_review_mark_chk.configure(state=tk.DISABLED)
            if self._stationary_review_xy77_label is not None:
                self._stationary_review_xy77_label.configure(image="")
            if self._stationary_review_xymax_label is not None:
                self._stationary_review_xymax_label.configure(image="")
            self._stationary_review_xy77_ref = None
            self._stationary_review_xymax_ref = None
            self._stationary_review_include_sync = True
            self._stationary_review_include_var.set(False)
            self._stationary_review_include_sync = False
            return

        self._stationary_review_idx = max(0, min(int(self._stationary_review_idx), n - 1))
        entry_state, rod_dir = self._stationary_review_items[self._stationary_review_idx]
        is_pending = entry_state == "pending"
        if self._stationary_review_prev_btn is not None:
            self._stationary_review_prev_btn.configure(state=tk.NORMAL)
        if self._stationary_review_next_btn is not None:
            self._stationary_review_next_btn.configure(state=tk.NORMAL)
        if self._stationary_review_mark_chk is not None:
            self._stationary_review_mark_chk.configure(
                state=(tk.NORMAL if is_pending else tk.DISABLED)
            )

        meta = self._stationary_review_load_json(rod_dir / "meta.json") or {}
        m77 = self._stationary_review_load_mode_meta(
            rod_dir, "capture_77fps_meta", "capture_77fps_meta.json"
        ) or {}
        mmax = self._stationary_review_load_mode_meta(
            rod_dir, "capture_maxfps_11x11_meta", "capture_maxfps_11x11_meta.json"
        ) or {}
        xy77_raw = m77.get("xy_series", [])
        xymax_raw = mmax.get("xy_series", [])
        xy77 = [(float(v[0]), float(v[1])) for v in xy77_raw if isinstance(v, (list, tuple)) and len(v) >= 2]
        xymax = [(float(v[0]), float(v[1])) for v in xymax_raw if isinstance(v, (list, tuple)) and len(v) >= 2]

        if self._stationary_review_xy77_label is not None:
            img77 = self._make_xy_scatter_image(xy77)
            p77 = ImageTk.PhotoImage(img77)
            self._stationary_review_xy77_label.configure(image=p77)
            self._stationary_review_xy77_ref = p77
        if self._stationary_review_xymax_label is not None:
            imgm = self._make_xy_scatter_image(xymax)
            pm = ImageTk.PhotoImage(imgm)
            self._stationary_review_xymax_label.configure(image=pm)
            self._stationary_review_xymax_ref = pm

        t_nohole = meta.get("theta_nohole_deg")
        t_hole = meta.get("theta_hole_fresnel_deg")
        c = meta.get("center_px", {})
        c_x = c.get("x")
        c_y = c.get("y")
        m77m = m77.get("xy_metrics", {})
        fps77 = m77.get("actual", {}).get("fps")
        fpsm = mmax.get("actual", {}).get("fps")
        status_txt = "Pending" if is_pending else "Good"
        detail = (
            f"{status_txt} {self._stationary_review_idx + 1}/{n}  |  {rod_dir.name}\n"
            f"center=({c_x}, {c_y})  theta_nohole={t_nohole} deg  theta_hole+fresnel={t_hole} deg\n"
            f"77fps: fps={fps77} frames={m77.get('actual', {}).get('frames')} "
            f"fov={m77.get('requested', {}).get('fov_raw_px')} "
            f"rangeX={m77m.get('range_x')} rangeY={m77m.get('range_y')} r_mean={m77m.get('r_mean')}\n"
            f"maxfps-11x11: fps={fpsm} frames={mmax.get('actual', {}).get('frames')}"
        )
        self._stationary_review_detail_var.set(detail)

        self._stationary_review_include_sync = True
        self._stationary_review_include_var.set(False)
        self._stationary_review_include_sync = False

    def _on_stationary_review_include_toggle(self) -> None:
        if self._stationary_review_include_sync:
            return
        if bool(self._stationary_review_include_var.get()):
            self._stationary_review_mark_current_good()

    def _stationary_review_mark_current_good(self) -> None:
        if not self._stationary_review_items:
            self._stationary_review_include_sync = True
            self._stationary_review_include_var.set(False)
            self._stationary_review_include_sync = False
            return
        idx = max(0, min(int(self._stationary_review_idx), len(self._stationary_review_items) - 1))
        entry_state, src = self._stationary_review_items[idx]
        if entry_state != "pending":
            self._stationary_review_include_sync = True
            self._stationary_review_include_var.set(False)
            self._stationary_review_include_sync = False
            return
        _, _, good_dir = self._stationary_dataset_paths()
        dst = good_dir / src.name
        if dst.exists():
            dst = good_dir / f"{src.name}_{time.time_ns()}"
        try:
            shutil.move(str(src), str(dst))
            self.bottom_var.set(f"Moved to good: {dst.name}")
        except Exception as e:
            messagebox.showerror("Stationary review", f"Could not move folder: {e}")
        self._stationary_review_include_sync = True
        self._stationary_review_include_var.set(False)
        self._stationary_review_include_sync = False
        self._stationary_review_refresh(True)

    def _reset_live_tracking(self, keep_shift: bool = False) -> None:
        self._live_track_prev = None
        self._live_track_resp = 0.0
        self._live_track_counter = 0
        if not keep_shift:
            self._live_track_shift = (0.0, 0.0)

    def _prepare_tracking_frame(self, frame8: np.ndarray) -> np.ndarray:
        h, w = int(frame8.shape[0]), int(frame8.shape[1])
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), dtype=np.uint8)
        img = np.asarray(frame8, dtype=np.uint8)
        # Remove smooth illumination profile so static shading does not dominate matching.
        try:
            bg_sigma = float(getattr(self, "_live_track_bg_sigma", 8.0))
            if bg_sigma > 0.01:
                bg = cv2.GaussianBlur(img, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma, borderType=cv2.BORDER_REPLICATE)
                img = cv2.subtract(img, bg)
        except Exception:
            pass
        sigma = float(getattr(self, "_live_track_blur_sigma", 0.0))
        if sigma > 0.01:
            try:
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
            except Exception:
                pass
        # Keep only the brightest spikes (e.g., 97th percentile and above).
        try:
            bright_pct = float(getattr(self, "_live_track_bright_pct", 97.0))
            bright_pct = max(80.0, min(99.9, bright_pct))
            thr = float(np.percentile(img, bright_pct))
            mask = (img >= thr).astype(np.uint8) * 255
            # Ensure non-trivial support for matching.
            nnz = int(np.count_nonzero(mask))
            min_nnz = max(16, int(0.0002 * float(mask.size)))
            if nnz < min_nnz:
                thr2 = float(np.percentile(img, max(80.0, bright_pct - 2.0)))
                mask = (img >= thr2).astype(np.uint8) * 255
            return mask
        except Exception:
            return img

    def _update_live_tracking(self, frame8: np.ndarray) -> None:
        if frame8 is None or getattr(frame8, "ndim", 0) != 2:
            return
        selected_center = self._get_selected_spot_center(tracked=False)
        if selected_center is None:
            self._reset_live_tracking(keep_shift=False)
            return
        interval = max(1, int(self._live_track_interval))
        self._live_track_counter = (int(self._live_track_counter) + 1) % interval
        if self._live_track_counter != 0:
            return
        try:
            h, w = int(frame8.shape[0]), int(frame8.shape[1])
            roi = int(getattr(self, "_live_track_roi_size", 400))
            tpl = int(getattr(self, "_live_track_template_size", 100))
            roi = max(64, min(roi, h - 4, w - 4))
            if roi % 2 != 0:
                roi -= 1
            tpl = max(24, min(tpl, roi - 8))
            if tpl % 2 != 0:
                tpl += 1
            tpl = min(tpl, roi - 4)
            cx = w // 2
            cy = h // 2
            half_r = roi // 2
            rx0 = max(0, min(w - roi, cx - half_r))
            ry0 = max(0, min(h - roi, cy - half_r))
            cur_roi_raw = frame8[ry0 : ry0 + roi, rx0 : rx0 + roi]
            cur = self._prepare_tracking_frame(cur_roi_raw)
            prev = self._live_track_prev
            if prev is not None and prev.shape == cur.shape:
                half_t = tpl // 2
                tx0 = max(0, min(roi - tpl, (roi // 2) - half_t))
                ty0 = max(0, min(roi - tpl, (roi // 2) - half_t))
                prev_tpl = prev[ty0 : ty0 + tpl, tx0 : tx0 + tpl]
                if prev_tpl.size > 0 and cur.size >= prev_tpl.size:
                    # Overlap score for sparse bright spikes.
                    res = cv2.matchTemplate(cur, prev_tpl, cv2.TM_CCORR_NORMED)
                    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
                    center_r = 0.5 * float(roi)
                    match_cx = float(max_loc[0]) + (0.5 * float(tpl))
                    match_cy = float(max_loc[1]) + (0.5 * float(tpl))
                    dx = float(match_cx - center_r)
                    dy = float(match_cy - center_r)
                    if bool(getattr(self, "_live_track_axis_lock", True)):
                        # Stage moves only along one axis per frame: keep dominant axis.
                        if abs(dx) >= abs(dy):
                            dy = 0.0
                        else:
                            dx = 0.0
                    max_step = max(8.0, (0.5 * float(roi - tpl)) - 2.0)
                    min_step = max(0.0, float(getattr(self, "_live_track_min_axis_step", 0.25)))
                    if (
                        np.isfinite(dx)
                        and np.isfinite(dy)
                        and np.isfinite(max_val)
                        and float(max_val) >= float(getattr(self, "_live_track_score_min", 0.08))
                        and (abs(dx) >= min_step or abs(dy) >= min_step)
                        and abs(dx) <= max_step
                        and abs(dy) <= max_step
                    ):
                        sx, sy = self._live_track_shift
                        self._live_track_shift = (float(sx) + dx, float(sy) + dy)
                        self._live_track_resp = float(max_val)
            self._live_track_prev = cur
        except Exception:
            return

    def _get_selected_spot_center(self, tracked: bool = True) -> Optional[tuple[float, float]]:
        if self._selected_center_override is not None:
            cx, cy = self._selected_center_override
        else:
            if not self._spot_centers:
                return None
            idx = max(0, min(int(self._spot_idx), len(self._spot_centers) - 1))
            cx, cy = self._spot_centers[idx]
        cx = float(cx)
        cy = float(cy)
        if tracked:
            dx, dy = self._live_track_shift
            cx += float(dx)
            cy += float(dy)
        shape = None
        if self._live_last_frame is not None and getattr(self._live_last_frame, "ndim", 0) == 2:
            shape = self._live_last_frame.shape
        elif self._source_shape is not None:
            shape = self._source_shape
        if shape is not None:
            h, w = int(shape[0]), int(shape[1])
            if w > 0 and h > 0:
                cx = max(0.0, min(float(w - 1), cx))
                cy = max(0.0, min(float(h - 1), cy))
        return (cx, cy)

    def _live_on_frame(self, arr_obj: object) -> None:
        if not self._live_running:
            return
        try:
            frame16 = np.asarray(arr_obj, dtype=np.uint16, copy=False)
            frame8 = (frame16 >> 4).astype(np.uint8, copy=False)
        except Exception:
            return
        self._live_last_frame = frame8
        try:
            self._live_queue.put_nowait(frame8)
        except queue.Full:
            try:
                self._live_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._live_queue.put_nowait(frame8)
            except queue.Full:
                pass

    def _apply_live_settings(self) -> None:
        if not self._live_controller:
            return
        exp_ms = self._parse_float(self._live_exp_ms_var.get())
        gain = self._parse_float(self._live_gain_var.get())
        if exp_ms is not None and exp_ms > 0.0:
            self._live_controller.set_timing(20.0, float(exp_ms))
        if gain is not None:
            self._live_controller.set_gains(float(gain), None)

    def _start_live_feed(self) -> None:
        if self._live_running:
            return
        try:
            from PySide6.QtWidgets import QApplication
            from Controlling.controller.controller import Controller
        except Exception as e:
            messagebox.showerror("Live feed", f"Could not start live feed: {e}")
            return

        exp_ms = self._parse_float(self._live_exp_ms_var.get())
        gain = self._parse_float(self._live_gain_var.get())
        if exp_ms is None or exp_ms <= 0.0:
            messagebox.showerror("Live feed", "Exposure time must be > 0 ms.")
            return
        if gain is None:
            gain = 0.0

        self._reset_live_tracking(keep_shift=False)
        self._live_app = QApplication.instance() or QApplication([])
        self._live_queue = queue.Queue(maxsize=2)
        self._live_controller = Controller()
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        if self._live_left_frame is not None:
            try:
                w = max(100, int(self._live_left_frame.winfo_width()))
                h = max(100, int(self._live_left_frame.winfo_height()))
                self._live_left_frame.configure(width=w, height=h)
                self._live_left_frame.grid_propagate(False)
            except Exception:
                pass
        try:
            self._live_controller.open()
            self._live_controller.full_sensor()
            self._live_controller.set_timing(20.0, float(exp_ms))
            self._live_controller.set_gains(float(gain), None)
            self._live_controller.start()
            self._live_controller.cam.frame.connect(self._live_on_frame)
        except Exception as e:
            try:
                self._live_controller.close()
            except Exception:
                pass
            self._live_controller = None
            messagebox.showerror("Live feed", f"Could not start live feed: {e}")
            return

        self._live_running = True
        if self._live_start_btn is not None:
            self._live_start_btn.state(["disabled"])
        if self._live_stop_btn is not None:
            self._live_stop_btn.state(["!disabled"])
        self._live_status_var.set("Live feed running (20 fps)")
        self._live_tick()

    def _stop_live_feed(self) -> None:
        if not self._live_running and self._live_controller is None:
            if self._live_start_btn is not None:
                self._live_start_btn.state(["!disabled"])
            if self._live_stop_btn is not None:
                self._live_stop_btn.state(["disabled"])
            self._live_status_var.set("Live feed stopped")
            return

        self._live_running = False
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        if self._live_controller is not None:
            try:
                self._live_controller.cam.frame.disconnect(self._live_on_frame)
            except Exception:
                pass
            try:
                self._live_controller.stop()
            except Exception:
                pass
            try:
                self._live_controller.close()
            except Exception:
                pass
        self._live_controller = None
        if self._live_start_btn is not None:
            self._live_start_btn.state(["!disabled"])
        if self._live_stop_btn is not None:
            self._live_stop_btn.state(["disabled"])
        self._live_status_var.set("Live feed stopped")

    def _live_tick(self) -> None:
        if not self._live_running:
            return
        try:
            if self._live_app is not None:
                self._live_app.processEvents()
        except Exception:
            pass

        frame = None
        try:
            while True:
                frame = self._live_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None and self._live_img_label is not None:
            try:
                self._update_live_tracking(frame)
                img = Image.fromarray(frame)
                try:
                    resample = Image.Resampling.BILINEAR
                    zoom_resample = Image.Resampling.NEAREST
                except Exception:
                    resample = Image.BILINEAR
                    zoom_resample = Image.NEAREST
                w = int(self._live_img_label.winfo_width())
                h = int(self._live_img_label.winfo_height())
                if w > 10 and h > 10:
                    src_w, src_h = img.size
                    scale = min(float(w) / float(src_w), float(h) / float(src_h))
                    disp_w = max(1, int(round(src_w * scale)))
                    disp_h = max(1, int(round(src_h * scale)))
                    img = img.resize((disp_w, disp_h), resample=resample)
                    self._live_disp_scale = scale
                    off_x = max(0, int((w - disp_w) // 2))
                    off_y = max(0, int((h - disp_h) // 2))
                    self._live_disp_offset = (off_x, off_y)

                    mag_on = bool(self._live_mag_enabled_var.get())
                    # Keep an RGB canvas so we can always draw spot overlays in color.
                    canvas_mode = "RGB"
                    canvas = Image.new(canvas_mode, (w, h), 0)
                    canvas.paste(img.convert("RGB"), (off_x, off_y))

                    # Overlay the currently selected spot from analysis/spot-examine tabs.
                    spot_xy = self._get_selected_spot_center(tracked=True)
                    if spot_xy is not None:
                        try:
                            cx, cy = spot_xy
                            if 0.0 <= cx < float(src_w) and 0.0 <= cy < float(src_h):
                                ring_r = 8
                                px = int(round(off_x + cx * scale))
                                py = int(round(off_y + cy * scale))
                                draw = ImageDraw.Draw(canvas)
                                draw.ellipse(
                                    [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                                    outline=(0, 255, 0),
                                    width=2,
                                )
                        except Exception:
                            pass

                    if mag_on:
                        z = self._parse_float(self._live_zoom_var.get())
                        zoom = float(z) if z and z > 0.1 else 1.0
                        zoom = max(1.0, zoom)
                        out_sz = int(self._live_zoom_output_px)
                        win = max(8, int(round(float(out_sz) / float(zoom))))
                        win = min(win, int(src_w), int(src_h))
                        win = max(1, int(win))
                        cx, cy = None, None
                        if self._live_zoom_center is not None:
                            cx, cy = self._live_zoom_center
                        if cx is None or cy is None:
                            cx = src_w / 2.0
                            cy = src_h / 2.0
                        half = win // 2
                        x0 = int(round(cx)) - half
                        y0 = int(round(cy)) - half
                        x0 = max(0, min(int(src_w) - win, x0))
                        y0 = max(0, min(int(src_h) - win, y0))
                        x1 = x0 + win
                        y1 = y0 + win

                        draw = ImageDraw.Draw(canvas)
                        dx0 = int(round(off_x + x0 * scale))
                        dy0 = int(round(off_y + y0 * scale))
                        dx1 = int(round(off_x + x1 * scale))
                        dy1 = int(round(off_y + y1 * scale))
                        draw.rectangle([dx0, dy0, dx1, dy1], outline=(255, 0, 0), width=2)

                        if self._live_zoom_label is not None:
                            crop = Image.fromarray(frame[y0:y1, x0:x1])
                            zoom_img = crop.resize((out_sz, out_sz), resample=zoom_resample)
                            zoom_photo = ImageTk.PhotoImage(zoom_img)
                            self._live_zoom_label.configure(image=zoom_photo)
                            self._live_zoom_label.image = zoom_photo
                    else:
                        if self._live_zoom_label is not None and self._live_zoom_blank_ref is not None:
                            self._live_zoom_label.configure(image=self._live_zoom_blank_ref)

                    img = canvas

                photo = ImageTk.PhotoImage(img)
                self._live_img_label.configure(image=photo)
                self._live_img_ref = photo
            except Exception:
                pass

        self._live_after_id = self.root.after(50, self._live_tick)

    def _on_live_click(self, event) -> None:
        if self._live_last_frame is None:
            return
        try:
            off_x, off_y = self._live_disp_offset
            scale = float(self._live_disp_scale) if self._live_disp_scale else 1.0
            x = float(event.x) - float(off_x)
            y = float(event.y) - float(off_y)
            if x < 0 or y < 0:
                return
            src_h, src_w = self._live_last_frame.shape
            fx = x / scale
            fy = y / scale
            if fx < 0 or fy < 0 or fx >= src_w or fy >= src_h:
                return
            self._live_zoom_center = (float(fx), float(fy))
        except Exception:
            return

    def _update_spotrec_label(self) -> None:
        center = self._get_selected_spot_center(tracked=False)
        if center is None:
            self._spotrec_spot_var.set("Spot - / -")
            if self._spotrec_start_btn is not None and not self._spotrec_running:
                self._spotrec_start_btn.state(["disabled"])
            self._spotrec_update_preview()
            return

        if self._selected_center_source == "stationary" and self._stationary_candidates:
            n = len(self._stationary_candidates)
            idx = max(0, min(int(self._stationary_idx), n - 1))
            self._spotrec_spot_var.set(f"Stationary {idx + 1} / {n}")
        elif self._spot_centers:
            n = len(self._spot_centers)
            idx = max(0, min(int(self._spot_idx), n - 1))
            self._spotrec_spot_var.set(f"Spot {idx + 1} / {n}")
        else:
            self._spotrec_spot_var.set("Spot selected")

        if self._spotrec_start_btn is not None and not self._spotrec_running:
            self._spotrec_start_btn.state(["!disabled"])
        self._spotrec_update_preview()

    def _start_spotrec_preview_loop(self) -> None:
        if self._spotrec_preview_after_id is not None:
            return
        self._spotrec_preview_tick()

    def _stop_spotrec_preview_loop(self) -> None:
        if self._spotrec_preview_after_id is None:
            return
        try:
            self.root.after_cancel(self._spotrec_preview_after_id)
        except Exception:
            pass
        self._spotrec_preview_after_id = None

    def _spotrec_preview_tick(self) -> None:
        self._spotrec_preview_after_id = None
        try:
            if hasattr(self, "_notebook"):
                current = self._notebook.select()
                if current != str(getattr(self, "_spotrec_tab", "")):
                    return
        except Exception:
            return

        frame = None
        if self._spotrec_running and self._spotrec_preview_path is not None:
            try:
                p = Path(self._spotrec_preview_path)
                if p.exists():
                    mtime = p.stat().st_mtime
                    if self._spotrec_preview_mtime != mtime:
                        arr = np.load(p, allow_pickle=False)
                        if arr is not None and getattr(arr, "ndim", 0) >= 2:
                            frame = _to_gray_u8(arr)
                            self._spotrec_preview_mtime = mtime
                            self._spotrec_preview_last_frame = frame
            except Exception:
                pass
            if frame is None and self._spotrec_preview_last_frame is not None:
                frame = self._spotrec_preview_last_frame
        if frame is None and (not self._spotrec_running) and self._live_running and self._live_last_frame is not None:
            frame = self._live_last_frame
        if frame is None:
            self._spotrec_preview_label.configure(image="")
            self._spotrec_preview_after_id = self.root.after(
                int(self._spotrec_preview_interval_ms),
                self._spotrec_preview_tick,
            )
            return
        self._spotrec_update_preview(gray_frame=frame)
        self._spotrec_preview_after_id = self.root.after(
            int(self._spotrec_preview_interval_ms),
            self._spotrec_preview_tick,
        )

    def _spotrec_update_preview(self, gray_frame: Optional[np.ndarray] = None) -> None:
        if self._spotrec_preview_label is None:
            return
        center = self._get_selected_spot_center(tracked=False)
        if center is None:
            self._spotrec_preview_label.configure(image="")
            return
        cx, cy = center

        base_win = int(self._spotrec_preview_size)
        src = gray_frame if (gray_frame is not None and getattr(gray_frame, "ndim", 0) == 2) else None
        if src is None:
            self._spotrec_preview_label.configure(image="")
            return
        # If preview frames come from a ROI capture, convert full-res center to ROI-local coords.
        if self._spotrec_running and gray_frame is not None:
            meta = getattr(self, "_spotrec_roi_meta", None)
            if isinstance(meta, dict):
                try:
                    rx = float(meta.get("x", 0))
                    ry = float(meta.get("y", 0))
                    cx = float(cx) - rx
                    cy = float(cy) - ry
                    h_src, w_src = src.shape
                    if cx < 0.0 or cy < 0.0 or cx >= float(w_src) or cy >= float(h_src):
                        self._spotrec_preview_label.configure(image="")
                        return
                except Exception:
                    pass
            # During recording: crop (n-1)x(n-1) around the ROI-local spot, then place it on a
            # fixed 18x18 canvas at the start-record red-box offset (not always centered).
            try:
                req_n = int(self._parse_int(self._spotrec_size_var.get()) or 7)
                req_n = max(1, int(req_n))
                crop_n = max(1, int(req_n - 1))
                h_src, w_src = src.shape
                crop_n = min(crop_n, int(min(h_src, w_src)))
                if crop_n < 1:
                    self._spotrec_preview_label.configure(image="")
                    return
                crop = self._extract_window(src, cx, cy, crop_n)
                target_n = 18
                canvas = np.zeros((target_n, target_n), dtype=crop.dtype)
                off_x_anchor = int(round(self._spotrec_center_offset[0]))
                off_y_anchor = int(round(self._spotrec_center_offset[1]))
                if isinstance(meta, dict):
                    try:
                        off_x_anchor = int(round(float(meta.get("off_x_start", off_x_anchor))))
                        off_y_anchor = int(round(float(meta.get("off_y_start", off_y_anchor))))
                    except Exception:
                        pass
                half_t = target_n // 2
                anchor_x = half_t + off_x_anchor
                anchor_y = half_t + off_y_anchor
                x0 = int(round(anchor_x - (crop_n // 2)))
                y0 = int(round(anchor_y - (crop_n // 2)))
                x1 = x0 + crop_n
                y1 = y0 + crop_n

                dst_x0 = max(0, x0)
                dst_y0 = max(0, y0)
                dst_x1 = min(target_n, x1)
                dst_y1 = min(target_n, y1)
                if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                    self._spotrec_preview_label.configure(image="")
                    return
                src_x0 = max(0, -x0)
                src_y0 = max(0, -y0)
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                canvas[dst_y0:dst_y1, dst_x0:dst_x1] = crop[src_y0:src_y1, src_x0:src_x1]
                window = canvas
                win = int(window.shape[0])
            except Exception:
                window = self._extract_window(src, cx, cy, base_win)
                win = int(base_win)
        else:
            window = self._extract_window(src, cx, cy, base_win)
            win = int(base_win)
        # Stronger contrast stretch for very dim live frames.
        u8 = detect_spinners.to_u8_preview(window, lo_pct=0.0, hi_pct=99.5)

        roi = self._parse_int(self._spotrec_size_var.get()) or 7
        roi = max(1, int(roi))
        if self._spotrec_running and gray_frame is not None:
            roi = max(1, int(roi - 1))
        elif roi % 2 == 0:
            roi += 1
        if roi > win:
            roi = win
        half_roi = roi // 2
        half_win = win // 2

        off_x, off_y = self._spotrec_center_offset
        if self._spotrec_running and gray_frame is not None and isinstance(meta, dict):
            try:
                off_x = int(round(float(meta.get("off_x_start", off_x))))
                off_y = int(round(float(meta.get("off_y_start", off_y))))
            except Exception:
                pass
            off_x = max(-half_win + half_roi, min(half_win - half_roi, int(round(off_x))))
            off_y = max(-half_win + half_roi, min(half_win - half_roi, int(round(off_y))))
        else:
            off_x = max(-half_win + half_roi, min(half_win - half_roi, int(round(off_x))))
            off_y = max(-half_win + half_roi, min(half_win - half_roi, int(round(off_y))))
            self._spotrec_center_offset = (off_x, off_y)

        x0 = half_win + off_x - half_roi
        y0 = half_win + off_y - half_roi
        x1 = x0 + roi - 1
        y1 = y0 + roi - 1

        disp_size = int(self._spotrec_preview_size) * int(self._spotrec_preview_scale)
        disp_size = max(10, disp_size)
        scale = float(disp_size) / float(win) if win > 0 else 1.0
        img = Image.fromarray(u8).convert("RGB")
        img = img.resize((disp_size, disp_size), resample=Image.NEAREST)
        # Draw ROI box on the scaled preview so the border doesn't expand with zoom.
        sx0 = int(round(x0 * scale))
        sy0 = int(round(y0 * scale))
        sx1 = int(round((x1 + 1) * scale - 1))
        sy1 = int(round((y1 + 1) * scale - 1))
        draw = ImageDraw.Draw(img)
        draw.rectangle([sx0, sy0, sx1, sy1], outline=(255, 0, 0), width=1)
        # Debug overlay: actual ROI + phase.
        meta = getattr(self, "_spotrec_roi_meta", None)
        if meta is not None:
            try:
                px = int(meta.get("phase_x", int(meta.get("x", 0)) % 2))
                py = int(meta.get("phase_y", int(meta.get("y", 0)) % 2))
                txt = f"ROI {int(meta.get('w', 0))}x{int(meta.get('h', 0))} @ ({int(meta.get('x', 0))},{int(meta.get('y', 0))})  phase {px},{py}"
                draw2 = ImageDraw.Draw(img)
                draw2.rectangle([0, img.height - 14, img.width, img.height], fill=(0, 0, 0))
                draw2.text((2, img.height - 12), txt, fill=(255, 255, 255))
            except Exception:
                pass
        photo = ImageTk.PhotoImage(img)
        self._spotrec_preview_label.configure(image=photo)
        self._spotrec_preview_label.image = photo

    def _set_spot_playback_enabled(self, enabled: bool) -> None:
        if self._spot_play_btn is None:
            return
        if enabled:
            self._spot_play_btn.configure(state=tk.NORMAL)
        else:
            self._spot_play_btn.configure(state=tk.DISABLED)

    def _on_spotrec_preview_click(self, event) -> None:
        if self._get_selected_spot_center(tracked=False) is None:
            return
        win = int(self._spotrec_preview_size)
        scale = int(self._spotrec_preview_scale)
        if scale <= 0:
            return
        try:
            x = int(event.x // scale)
            y = int(event.y // scale)
        except Exception:
            return
        half_win = win // 2
        off_x = x - half_win
        off_y = y - half_win
        self._spotrec_center_offset = (off_x, off_y)
        self._spotrec_update_preview()

    def _spotrec_on_frame(self, arr_obj: object) -> None:
        if not self._spotrec_running:
            return
        try:
            frame16 = np.asarray(arr_obj, dtype=np.uint16, copy=True)
        except Exception:
            return
        # No frame dropping: record every frame.
        self._spotrec_frames.append(frame16)
        x, y, phi = self._spotrec_compute_xy_phi(frame16)
        self._spotrec_xy_series.append((x, y))
        self._spotrec_phi_series.append(phi)

    def _spotrec_on_timing(self, payload: object) -> None:
        try:
            d = dict(payload or {})
            rf = d.get("resulting_fps")
            if rf is None:
                rf = d.get("fps")
            if rf is not None:
                self._spotrec_actual_fps = float(rf)
        except Exception:
            return

    def _spotrec_compute_xy_phi(self, gray: np.ndarray) -> tuple[float, float, float]:
        meta = getattr(self, "_spotrec_roi_meta", None)
        win_raw = self._parse_int(self._spotrec_size_var.get()) or 7
        return self._xy_phi_from_frame(gray=np.asarray(gray), roi_meta=meta, win_raw=int(win_raw))

    def _spotrec_tick(self) -> None:
        if not self._spotrec_running:
            return
        try:
            if self._spotrec_app is not None:
                self._spotrec_app.processEvents()
        except Exception:
            pass

        self._spotrec_status_var.set(f"Recording... frames={len(self._spotrec_phi_series)}")
        self._spotrec_after_id = self.root.after(30, self._spotrec_tick)

    def _start_spotrec(self) -> None:
        if self._spotrec_running:
            return
        center = self._get_selected_spot_center(tracked=False)
        if center is None:
            messagebox.showerror("Spot examine", "No spot selected.")
            return
        fps = self._parse_float(self._spotrec_fps_var.get())
        exp_ms = self._parse_float(self._spotrec_exp_ms_var.get())
        if fps is None or fps <= 0.0:
            messagebox.showerror("Spot examine", "Frame rate must be > 0.")
            return
        if exp_ms is None or exp_ms <= 0.0:
            messagebox.showerror("Spot examine", "Exposure time must be > 0 ms.")
            return

        cx, cy = center
        # Ensure live feed is stopped so the camera is free.
        self._stop_live_feed()
        preview_every = int(max(1, int(self._spotrec_preview_every)))
        off_x, off_y = self._spotrec_center_offset
        w_user = self._parse_int(self._spotrec_size_var.get()) or 7
        w_user = max(1, int(w_user))
        if w_user % 2 == 0:
            w_user += 1
        cx2 = float(cx) + float(off_x)
        cy2 = float(cy) + float(off_y)
        roi_req = self._roi_from_target_center(center=(cx2, cy2), roi_raw=int(w_user))
        x = int(roi_req["x"])
        y = int(roi_req["y"])
        w_cam = int(roi_req["w"])
        h_cam = int(roi_req["h"])
        w_user = int(roi_req["roi_raw"])

        out_dir = self._recordings_subdir(self.RECORDINGS_SPOT_DIRNAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"spotrec_{ts}.npy"
        preview_path = out_dir / f"spotrec_preview_{ts}.npy"
        stop_flag = out_dir / f"spotrec_{ts}.stop"
        try:
            stop_flag.unlink(missing_ok=True)
        except Exception:
            pass

        script = Path(__file__).resolve().parent / "fetch_frames.py"
        args = [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--out-path",
            str(out_path),
            "--roi",
            str(x),
            str(y),
            str(w_cam),
            str(h_cam),
            "--fps",
            str(float(fps)),
            "--exp-ms",
            str(float(exp_ms)),
            "--stop-flag",
            str(stop_flag),
            "--preview-path",
            str(preview_path),
            "--preview-every",
            str(int(preview_every)),
            "--json",
            "--n-frames",
            "1000000",
        ]

        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            messagebox.showerror("Spot examine", f"Could not start recording: {e}")
            return

        self._spotrec_proc = proc
        self._spotrec_stop_flag = stop_flag
        self._spotrec_out_path = out_path
        self._spotrec_preview_path = preview_path
        self._spotrec_preview_mtime = None
        self._spotrec_preview_every = int(preview_every)
        self._spotrec_preview_last_frame = None
        self._spotrec_roi_meta = {
            "x": int(x),
            "y": int(y),
            "w": int(w_cam),
            "h": int(h_cam),
            "cx": float(cx2),
            "cy": float(cy2),
            "off_x_start": int(off_x),
            "off_y_start": int(off_y),
            "win_raw": int(w_user),
            "phase_x": int(x) % 2,
            "phase_y": int(y) % 2,
        }
        self._spotrec_xy_series = []
        self._spotrec_phi_series = []
        self._spotrec_frames = []
        self._spotrec_actual_fps = None
        self._spotrec_tmp_path = None
        if self._spotrec_save_btn is not None:
            self._spotrec_save_btn.state(["disabled"])
        if self._spotrec_discard_btn is not None:
            self._spotrec_discard_btn.state(["disabled"])

        self._spotrec_running = True
        if self._spotrec_start_btn is not None:
            self._spotrec_start_btn.state(["disabled"])
        if self._spotrec_stop_btn is not None:
            self._spotrec_stop_btn.state(["!disabled"])
        self._spotrec_status_var.set("Recording...")
        self._spotrec_progress_var.set("")

    def _stop_spotrec(self) -> None:
        if not self._spotrec_running and self._spotrec_proc is None:
            if self._spotrec_start_btn is not None:
                self._spotrec_start_btn.state(["!disabled"])
            if self._spotrec_stop_btn is not None:
                self._spotrec_stop_btn.state(["disabled"])
            return

        self._spotrec_running = False
        if self._spotrec_stop_flag is not None:
            try:
                Path(self._spotrec_stop_flag).write_text("stop")
            except Exception:
                pass

        if self._spotrec_stop_btn is not None:
            self._spotrec_stop_btn.state(["disabled"])
        self._spotrec_status_var.set("Stopping...")

        proc = self._spotrec_proc
        self._spotrec_proc = None

        def _clear_preview():
            p = self._spotrec_preview_path
            self._spotrec_preview_path = None
            self._spotrec_preview_mtime = None
            self._spotrec_preview_last_frame = None
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

        def _wait_and_process():
            if proc is None:
                return
            try:
                out, err = proc.communicate()
            except Exception as e:
                self._ui_call(messagebox.showerror, "Spot examine", str(e))
                _clear_preview()
                return
            if proc.returncode != 0:
                msg = (err or out or "Recording failed.").strip()
                self._ui_call(messagebox.showerror, "Spot examine", msg)
                self._ui_call(self._spotrec_status_var.set, "Recording failed.")
                self._ui_call(self._spotrec_progress_var.set, "")
                self._ui_call(self._update_spotrec_label)
                if self._spotrec_start_btn is not None:
                    self._ui_call(self._spotrec_start_btn.state, ["!disabled"])
                _clear_preview()
                return

            payload = (out or "").strip().splitlines()
            if not payload:
                self._ui_call(messagebox.showerror, "Spot examine", "No output from recorder.")
                if self._spotrec_start_btn is not None:
                    self._ui_call(self._spotrec_start_btn.state, ["!disabled"])
                _clear_preview()
                return
            try:
                data = json.loads(payload[-1])
                path = Path(str(data.get("path", "")))
                actual_fps = data.get("actual_fps")
                actual_fps = float(actual_fps) if actual_fps is not None else None
                roi = data.get("roi")
            except Exception as e:
                self._ui_call(messagebox.showerror, "Spot examine", f"Bad output: {e}")
                if self._spotrec_start_btn is not None:
                    self._ui_call(self._spotrec_start_btn.state, ["!disabled"])
                _clear_preview()
                return
            if not path.exists():
                self._ui_call(messagebox.showerror, "Spot examine", "Recording file missing.")
                if self._spotrec_start_btn is not None:
                    self._ui_call(self._spotrec_start_btn.state, ["!disabled"])
                _clear_preview()
                return

            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as e:
                self._ui_call(messagebox.showerror, "Spot examine", f"Could not load: {e}")
                if self._spotrec_start_btn is not None:
                    self._ui_call(self._spotrec_start_btn.state, ["!disabled"])
                _clear_preview()
                return

            xy_series = []
            phi_series = []
            total = int(arr.shape[0]) if arr.ndim >= 3 else 0
            step = max(1, total // 20) if total else 1
            for i in range(total):
                frame = arr[i]
                x, y, phi = self._spotrec_compute_xy_phi(frame)
                xy_series.append((x, y))
                phi_series.append(phi)
                if (i + 1) % step == 0 or (i + 1) == total:
                    self._ui_call(self._spotrec_progress_var.set, f"Analyzing {i+1}/{total}")

            self._ui_call(self._spotrec_progress_var.set, "Analysis complete")
            self._ui_call(self._spotrec_status_var.set, f"Stopped. frames={total}")

            def _apply_results():
                self._spotrec_actual_fps = actual_fps
                if self._spotrec_actual_fps and self._spotrec_actual_fps > 0.0:
                    self._spotrec_fps_var.set(f"{float(self._spotrec_actual_fps):.3f}")
                # Update ROI meta from actual camera readback.
                if isinstance(roi, dict) and self._spotrec_roi_meta is not None:
                    try:
                        rx = roi.get("OffsetX", roi.get("x"))
                        ry = roi.get("OffsetY", roi.get("y"))
                        rw = roi.get("Width", roi.get("w"))
                        rh = roi.get("Height", roi.get("h"))
                        if rx is not None:
                            self._spotrec_roi_meta["x"] = int(round(float(rx)))
                        if ry is not None:
                            self._spotrec_roi_meta["y"] = int(round(float(ry)))
                        if rw is not None:
                            self._spotrec_roi_meta["w"] = int(round(float(rw)))
                        if rh is not None:
                            self._spotrec_roi_meta["h"] = int(round(float(rh)))
                        self._spotrec_roi_meta["phase_x"] = int(self._spotrec_roi_meta.get("x", 0)) % 2
                        self._spotrec_roi_meta["phase_y"] = int(self._spotrec_roi_meta.get("y", 0)) % 2
                    except Exception:
                        pass
                self._spotrec_xy_series = xy_series
                self._spotrec_phi_series = phi_series
                self._spotrec_tmp_path = path
                _clear_preview()
                if self._spotrec_save_btn is not None:
                    self._spotrec_save_btn.state(["!disabled"])
                if self._spotrec_discard_btn is not None:
                    self._spotrec_discard_btn.state(["!disabled"])
                self._spotrec_refresh_plots()
                self._update_spotrec_label()
                if self._spotrec_start_btn is not None:
                    self._spotrec_start_btn.state(["!disabled"])
                # Resume the live camera stream after spot recording completes.
                self._start_live_feed()
            self._ui_call(_apply_results)

        t = threading.Thread(target=_wait_and_process, daemon=True)
        t.start()

    def _spotrec_refresh_plots(self) -> None:
        fps = self._spotrec_actual_fps
        if not fps or fps <= 0.0:
            fps = self._parse_float(self._spotrec_fps_var.get()) or 1.0

        xy = list(self._spotrec_xy_series)
        phi = list(self._spotrec_phi_series)
        self._update_spotrec_brownian_label(phi, fps)

        if self._spotrec_xy_label is not None:
            img = self._make_xy_scatter_image(xy)
            img = img.resize((320, 320), resample=Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._spotrec_xy_label.configure(image=photo)
            self._spotrec_xy_ref = photo

        if self._spotrec_phi_label is not None:
            img = self._make_phi_plot_image(phi, fps)
            img = img.resize((320, 180), resample=Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._spotrec_phi_label.configure(image=photo)
            self._spotrec_phi_ref = photo

        if self._spotrec_fft_label is not None:
            img = self._make_fft_image(phi, fps)
            img = img.resize((320, 220), resample=Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._spotrec_fft_label.configure(image=photo)
            self._spotrec_fft_ref = photo

        m = self._directionality_metrics(xy, fps)
        if m is None:
            self._spotrec_dir_var.set("B: -")
            if self._spotrec_dir_label is not None:
                self._spotrec_dir_label.configure(image="")
            if self._spotrec_hand_label is not None:
                self._spotrec_hand_label.configure(image="")
            return

        b = m.get("B")
        if b is None:
            self._spotrec_dir_var.set("B: -")
        else:
            self._spotrec_dir_var.set(f"B: {float(b):.3f}")

        if self._spotrec_dir_label is not None:
            img = self._make_directionality_psd_image(m)
            img = img.resize((320, 220), resample=Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._spotrec_dir_label.configure(image=photo)
            self._spotrec_dir_ref = photo

        if self._spotrec_hand_label is not None:
            img = self._make_handedness_image(m)
            img = img.resize((320, 220), resample=Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._spotrec_hand_label.configure(image=photo)
            self._spotrec_hand_ref = photo

    def _spotrec_save(self) -> None:
        if self._spotrec_tmp_path is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save spot recording",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if not path:
            return
        dest = Path(path)
        try:
            if dest.exists():
                if not messagebox.askyesno("Overwrite", "File exists. Overwrite?"):
                    return
            shutil.copy2(self._spotrec_tmp_path, dest)
            self._spotrec_status_var.set(f"Saved copy: {dest.name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _spotrec_discard(self) -> None:
        if self._spotrec_tmp_path is None:
            return
        kept_name = Path(self._spotrec_tmp_path).name
        self._spotrec_tmp_path = None
        if self._spotrec_save_btn is not None:
            self._spotrec_save_btn.state(["disabled"])
        if self._spotrec_discard_btn is not None:
            self._spotrec_discard_btn.state(["disabled"])
        self._spotrec_status_var.set(f"Discarded from UI (kept file: {kept_name})")

    def _on_smap_click(self, event) -> None:
        """
        Click a spot ring on the S-map overview to jump to that spot.
        Uses nearest-center selection with a radius gate.
        """
        if self._smap_canvas is None or not self._spot_centers:
            return

        # Convert to canvas coordinates (accounts for scrolling).
        x = float(self._smap_canvas.canvasx(event.x))
        y = float(self._smap_canvas.canvasy(event.y))
        scale = float(self.S_MAP_DISPLAY_SCALE)
        if scale <= 0.0:
            return
        fx = x / scale
        fy = y / scale

        centers = np.asarray(self._spot_centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 2:
            return

        dx = centers[:, 0] - fx
        dy = centers[:, 1] - fy
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        # Require click reasonably close to the ring (in full-res pixels).
        gate = float(self.S_MAP_RING_R) * 1.4
        if float(d2[i]) > gate * gate:
            return

        self._set_selected_center_override(None, source="analysis")
        self._spot_idx = i
        self._update_spot_view()
        self._rebuild_smap_overlay()

    def _smap_preview_u8(self, s_map: np.ndarray) -> np.ndarray:
        """
        Display-oriented stretch for the overview S-map:
        percentile clip + mild gamma, to make faint dots visible without the full log stretch.
        """
        x = s_map.astype(np.float32, copy=False)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=np.uint8)

        # Treat exact zeros as background for visibility.
        nz = (x != 0.0) & finite
        vals = x[nz] if nz.any() else x[finite]
        if vals.size == 0:
            return np.zeros_like(x, dtype=np.uint8)

        # Narrower window to avoid over-stretching on-screen (keeps noise down).
        lo = float(np.percentile(vals, 15.0))
        hi = float(np.percentile(vals, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)

        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        gamma = float(self.S_MAP_DISPLAY_GAMMA)
        if gamma > 0.0 and gamma != 1.0:
            y = np.power(y, gamma)
        u8 = (y * 255.0 + 0.5).astype(np.uint8)
        u8[~finite] = 0
        u8[x == 0.0] = 0
        return u8

    def _set_smap_background(self) -> None:
        if self._smap_canvas is None or self._s_map is None:
            return
        u8 = self._smap_preview_u8(self._s_map)
        h, w = u8.shape
        scale = float(self.S_MAP_DISPLAY_SCALE)
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        img = Image.fromarray(u8).resize((out_w, out_h), resample=Image.BILINEAR).convert("RGB")
        self._smap_bg_ref = ImageTk.PhotoImage(img)
        if self._smap_canvas_img_id is None:
            self._smap_canvas_img_id = self._smap_canvas.create_image(
                0, 0, anchor="nw", image=self._smap_bg_ref
            )
        else:
            self._smap_canvas.itemconfigure(self._smap_canvas_img_id, image=self._smap_bg_ref)
        self._smap_canvas.configure(scrollregion=(0, 0, out_w, out_h))

    def _rebuild_smap_overlay(self) -> None:
        if self._smap_canvas is None:
            return
        if self._smap_overlay_after_id is not None:
            try:
                self.root.after_cancel(self._smap_overlay_after_id)
            except Exception:
                pass
            self._smap_overlay_after_id = None
        # Clear old overlay items (keep background).
        for item_id in self._smap_spot_ring_ids:
            self._smap_canvas.delete(item_id)
        for item_id in self._smap_spot_text_ids:
            self._smap_canvas.delete(item_id)
        self._smap_spot_ring_ids = []
        self._smap_spot_text_ids = []

        if not self._spot_centers:
            return

        scale = float(self.S_MAP_DISPLAY_SCALE)
        r = float(self.S_MAP_RING_R) * scale
        # Build overlay incrementally to avoid UI freezes on many candidates.
        # Pop from the end for O(1) per element (order doesn't matter for rings).
        self._smap_overlay_pending = list(enumerate(self._spot_centers))

        def _tick():
            if self._smap_canvas is None:
                self._smap_overlay_after_id = None
                return
            batch = 100
            for _ in range(batch):
                if not self._smap_overlay_pending:
                    self._smap_overlay_after_id = None
                    return
                i, (cx, cy) = self._smap_overlay_pending.pop()
                x = float(cx) * scale
                y = float(cy) * scale
                color = "#00cc44" if i == int(self._spot_idx) else "#ff3333"
                ring_id = self._smap_canvas.create_oval(
                    x - r, y - r, x + r, y + r, outline=color, width=2
                )
                self._smap_spot_ring_ids.append(ring_id)
            self._smap_overlay_after_id = self.root.after(1, _tick)

        self._smap_overlay_after_id = self.root.after(1, _tick)

    def _extract_window(self, img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError("Expected 2D image for window extraction.")
        h, w = img.shape
        half = size // 2
        x = int(round(cx))
        y = int(round(cy))
        x0 = x - half
        x1 = x + half + 1
        y0 = y - half
        y1 = y + half + 1

        window = np.zeros((size, size), dtype=img.dtype)

        src_x0 = max(0, x0)
        src_x1 = min(w, x1)
        src_y0 = max(0, y0)
        src_y1 = min(h, y1)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return window

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        window[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
        return window

    def _update_spot_bounds_intensity(self, raw_shape: tuple[int, int]) -> None:
        # Intensity-plane bounds for computing (I0,I45,I90,I135) statistics.
        h, w = raw_shape
        ih, iw = h // 2, w // 2
        win = max(1, int(round(self._spot_window_size / 2.0)))
        if win % 2 == 0:
            win += 1
        half = win // 2

        bounds: list[tuple[int, int, int, int]] = []
        for cx, cy in self._spot_centers_all:
            ix = int(round(cx / 2.0))
            iy = int(round(cy / 2.0))
            x0 = max(0, ix - half)
            x1 = min(iw, ix + half + 1)
            y0 = max(0, iy - half)
            y1 = min(ih, iy + half + 1)
            bounds.append((x0, x1, y0, y1))
        self._spot_bounds_int_all = bounds

    def _spot_xy_range_score(self, series: list[tuple[float, float]]) -> float:
        if not series:
            return float("-inf")
        arr = np.asarray(series, dtype=np.float32)
        if arr.size == 0:
            return float("-inf")
        x = arr[:, 0]
        y = arr[:, 1]
        range_x = float(np.max(x) - np.min(x))
        range_y = float(np.max(y) - np.min(y))
        return (range_x * range_x) + (range_y * range_y)

    def _spot_xy_max_axis_range(self, series: list[tuple[float, float]]) -> float:
        if not series:
            return float("-inf")
        arr = np.asarray(series, dtype=np.float32)
        if arr.size == 0:
            return float("-inf")
        x = arr[:, 0]
        y = arr[:, 1]
        range_x = float(np.max(x) - np.min(x))
        range_y = float(np.max(y) - np.min(y))
        return max(range_x, range_y)

    def _spot_center_key(self, center: tuple[float, float]) -> tuple[int, int]:
        cx, cy = center
        return (int(round(float(cx))), int(round(float(cy))))

    def _capture_auto_spot_series(
        self,
        center: tuple[float, float],
        n_frames: int,
        roi_raw: int,
        exp_ms: Optional[float],
    ) -> tuple[list[tuple[float, float]], list[float], Optional[float], int]:
        cx, cy = center
        roi_raw = max(1, int(roi_raw))
        if roi_raw % 2 == 0:
            roi_raw += 1

        # Fixed FOV for auto inspection.
        w_cam = int(roi_raw)
        h_cam = int(roi_raw)
        x = int(round(float(cx))) - (w_cam // 2)
        y = int(round(float(cy))) - (h_cam // 2)
        if (x % 2) != 0:
            x -= 1
        if (y % 2) != 0:
            y -= 1
        x = max(0, x)
        y = max(0, y)

        out_dir = self._recordings_subdir(self.RECORDINGS_SPOT_DIRNAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        token = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}"
        key = self._spot_center_key(center)
        out_path = out_dir / f"auto_spot_{key[0]}_{key[1]}_{token}.npy"
        n_frames = max(1, int(n_frames))

        script = Path(__file__).resolve().parent / "fetch_frames.py"
        args = [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--out-path",
            str(out_path),
            "--roi",
            str(x),
            str(y),
            str(w_cam),
            str(h_cam),
            "--json",
            "--n-frames",
            str(n_frames),
            "--stop-after",
            str(n_frames),
            "--fps",
            str(float(self.AUTO_INSPECT_FPS)),
        ]
        # Use a fixed auto-inspection frame rate.
        if exp_ms is not None and float(exp_ms) > 0.0:
            args.extend(["--exp-ms", str(float(exp_ms))])

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=300.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            raise RuntimeError((err or out or "Auto inspect capture timed out.").strip())

        if proc.returncode != 0:
            raise RuntimeError((err or out or "Auto inspect capture failed.").strip())

        payload = (out or "").strip().splitlines()
        if not payload:
            raise RuntimeError("Auto inspect recorder produced no output.")
        try:
            data = json.loads(payload[-1])
            path = Path(str(data.get("path", "")))
            actual_fps = data.get("actual_fps")
            actual_fps = float(actual_fps) if actual_fps is not None else None
            roi = data.get("roi")
        except Exception as e:
            raise RuntimeError(f"Auto inspect output parse failed: {e}")
        if not path.exists():
            raise RuntimeError("Auto inspect recording file missing.")

        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"Auto inspect could not load capture: {e}")

        total = int(arr.shape[0]) if arr.ndim >= 3 else 0
        if total <= 0:
            return ([], [], actual_fps, 0)

        roi_meta = {
            "x": int(x),
            "y": int(y),
            "w": int(w_cam),
            "h": int(h_cam),
            "cx": float(cx),
            "cy": float(cy),
            "win_raw": int(roi_raw),
            "phase_x": int(x) % 2,
            "phase_y": int(y) % 2,
        }
        if isinstance(roi, dict):
            try:
                rx = roi.get("OffsetX", roi.get("x"))
                ry = roi.get("OffsetY", roi.get("y"))
                rw = roi.get("Width", roi.get("w"))
                rh = roi.get("Height", roi.get("h"))
                if rx is not None:
                    roi_meta["x"] = int(round(float(rx)))
                if ry is not None:
                    roi_meta["y"] = int(round(float(ry)))
                if rw is not None:
                    roi_meta["w"] = int(round(float(rw)))
                if rh is not None:
                    roi_meta["h"] = int(round(float(rh)))
                roi_meta["phase_x"] = int(roi_meta.get("x", 0)) % 2
                roi_meta["phase_y"] = int(roi_meta.get("y", 0)) % 2
            except Exception:
                pass

        xy_series: list[tuple[float, float]] = []
        phi_series: list[float] = []
        win_raw = int(roi_meta.get("win_raw", roi_raw))
        win_raw = max(1, int(win_raw))
        win = max(1, int(round(win_raw / 2.0)))
        if win % 2 == 0:
            win += 1
        half = win // 2
        eps = 1e-6
        px = int(roi_meta.get("phase_x", int(roi_meta.get("x", 0)) % 2)) % 2
        py = int(roi_meta.get("phase_y", int(roi_meta.get("y", 0)) % 2)) % 2
        cx_rel = float(roi_meta.get("cx", cx)) - float(roi_meta.get("x", x))
        cy_rel = float(roi_meta.get("cy", cy)) - float(roi_meta.get("y", y))
        cx_i = int(round(cx_rel / 2.0))
        cy_i = int(round(cy_rel / 2.0))

        for i in range(total):
            g = arr[i]
            if g.ndim != 2:
                g = g[..., 0]
            I0 = g[py::2, px::2]
            I45 = g[py::2, (1 - px) :: 2]
            I135 = g[(1 - py) :: 2, px::2]
            I90 = g[(1 - py) :: 2, (1 - px) :: 2]

            ih, iw = I0.shape
            if ih <= 0 or iw <= 0:
                continue
            x0 = max(0, cx_i - half)
            x1 = min(iw, cx_i + half + 1)
            y0 = max(0, cy_i - half)
            y1 = min(ih, cy_i + half + 1)

            a0 = I0[y0:y1, x0:x1]
            a90 = I90[y0:y1, x0:x1]
            a45 = I45[y0:y1, x0:x1]
            a135 = I135[y0:y1, x0:x1]
            m0 = float(a0.mean()) if a0.size else 0.0
            m90 = float(a90.mean()) if a90.size else 0.0
            m45 = float(a45.mean()) if a45.size else 0.0
            m135 = float(a135.mean()) if a135.size else 0.0
            x_v = (m0 - m90) / (m0 + m90 + eps)
            y_v = (m45 - m135) / (m45 + m135 + eps)
            xy_series.append((float(x_v), float(y_v)))
            phi_series.append(float(0.5 * np.arctan2(y_v, x_v)))

        return (xy_series, phi_series, actual_fps, len(phi_series))

    def _auto_inspect_top_spots(self) -> int:
        with self._analysis_lock:
            spots = list(self._spot_centers[: int(self.AUTO_INSPECT_TOP_N)])
        if not spots:
            return 0

        exp_ms = 0.02
        n_frames = int(self.AUTO_INSPECT_FRAMES)
        roi_raw = int(self.AUTO_INSPECT_ROI_RAW)
        total = len(spots)
        updated = 0
        self._ui_call(
            self.bottom_var.set,
            f"Auto inspect: {total} spot(s), {n_frames} frame(s) each, ROI {roi_raw}x{roi_raw}.",
        )

        for i, center in enumerate(spots, start=1):
            self._ui_call(self.bottom_var.set, f"Auto inspect {i}/{total} in progress...")
            try:
                xy, phi, fps, n = self._capture_auto_spot_series(
                    center=center,
                    n_frames=n_frames,
                    roi_raw=roi_raw,
                    exp_ms=exp_ms,
                )
            except Exception as e:
                _append_timing_log(f"[SpotAnalysis] Auto inspect spot {i}/{total} failed: {e}")
                continue
            if not xy or not phi or n <= 0:
                continue
            key = self._spot_center_key(center)
            with self._analysis_lock:
                self._spot_inspect_overrides[key] = {
                    "xy": list(xy),
                    "phi": list(phi),
                    "fps": float(fps) if fps is not None and fps > 0.0 else None,
                }
                # Invalidate cached plots so UI redraw picks up improved series immediately.
                self._spot_view_cache = []
            updated += 1
            if fps is not None and fps > 0.0:
                msg = f"Auto inspect {i}/{total}: {n} frames at {float(fps):.1f} fps."
            else:
                msg = f"Auto inspect {i}/{total}: {n} frames."
            self._ui_call(self.bottom_var.set, msg)
            self._ui_call(self._update_spot_view)

        if updated > 0:
            self._ui_call(self.bottom_var.set, f"Auto inspect complete: updated {updated}/{total} spots.")
        else:
            self._ui_call(self.bottom_var.set, "Auto inspect complete: no spots updated.")
        return updated

    def _start_auto_inspect_async(self) -> bool:
        with self._auto_inspect_start_lock:
            if self._auto_inspect_running:
                self._ui_call(self.bottom_var.set, "Auto inspect already running.")
                return False
            with self._analysis_lock:
                has_spots = bool(self._spot_centers)
            if not has_spots:
                self._ui_call(self.bottom_var.set, "Auto inspect: no spots available.")
                return False
            self._auto_inspect_running = True

        def _worker():
            t_auto = time.perf_counter()
            try:
                self._auto_inspect_top_spots()
            finally:
                _append_timing_log(f"[SpotAnalysis] Auto inspect: {time.perf_counter() - t_auto:.3f}s")
                with self._auto_inspect_start_lock:
                    self._auto_inspect_running = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def _sort_spots_by_xy_range(self) -> None:
        if not self._spot_centers or not self._spot_xy_series:
            return
        scores = [self._spot_xy_range_score(series) for series in self._spot_xy_series]
        if all(score == float("-inf") for score in scores):
            return
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        self._spot_centers = [self._spot_centers[i] for i in indices]
        self._spot_xy_series = [self._spot_xy_series[i] for i in indices]
        if self._spot_phi_series:
            self._spot_phi_series = [self._spot_phi_series[i] for i in indices]
        self._spot_idx = 0
        # Reset caches since ordering changed.
        self._spot_window_cache = [None for _ in self._spot_centers]
        self._spot_window_cache_size = int(self._spot_window_size)
        self._spot_view_cache = [
            {
                "spot_u8": None,
                "spot_win": None,
                "phi_len": None,
                "xy_len": None,
                "fft": None,
                "phi": None,
                "xy": None,
                "dir_len": None,
                "dir_psd": None,
                "dir_hand": None,
                "dir_B": None,
            }
            for _ in self._spot_centers
        ]

    def _append_xy_from_frame(self, gray: np.ndarray) -> None:
        # Compute per-spot X/Y using intensity subframes.
        if (not self._spot_bounds_int_all) or (
            len(self._spot_bounds_int_all) != len(self._spot_centers_all)
        ):
            self._update_spot_bounds_intensity(gray.shape)

        I0 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I90 = gray[1::2, 1::2]

        eps = 1e-6
        for i, (x0, x1, y0, y1) in enumerate(self._spot_bounds_int_all):
            a0 = I0[y0:y1, x0:x1]
            a90 = I90[y0:y1, x0:x1]
            a45 = I45[y0:y1, x0:x1]
            a135 = I135[y0:y1, x0:x1]

            m0 = float(a0.mean()) if a0.size else 0.0
            m90 = float(a90.mean()) if a90.size else 0.0
            m45 = float(a45.mean()) if a45.size else 0.0
            m135 = float(a135.mean()) if a135.size else 0.0

            x = (m0 - m90) / (m0 + m90 + eps)
            y = (m45 - m135) / (m45 + m135 + eps)
            self._spot_xy_series_all[i].append((float(x), float(y)))
            # Phi uses normalized Stokes-like quantities:
            # q = (I0 - I90) / (I0 + I90), u = (I45 - I135) / (I45 + I135)
            self._spot_phi_series_all[i].append(float(0.5 * np.arctan2(y, x)))

    def _append_xy_frame(self, gray: np.ndarray, frame_idx: int) -> None:
        # Compute per-frame XY/phi for all candidates, but avoid holding the analysis lock
        # across the expensive per-spot mean computations (this can freeze the UI).
        with self._analysis_lock:
            if not self._spot_centers_all:
                return
            if self._xy_frames_processed >= frame_idx:
                return
            if (not self._spot_bounds_int_all) or (
                len(self._spot_bounds_int_all) != len(self._spot_centers_all)
            ):
                self._update_spot_bounds_intensity(gray.shape)
            bounds = list(self._spot_bounds_int_all)
            xy_series_all = self._spot_xy_series_all
            phi_series_all = self._spot_phi_series_all

        I0 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I90 = gray[1::2, 1::2]

        eps = 1e-6
        vals: list[tuple[float, float, float]] = []
        for (x0, x1, y0, y1) in bounds:
            a0 = I0[y0:y1, x0:x1]
            a90 = I90[y0:y1, x0:x1]
            a45 = I45[y0:y1, x0:x1]
            a135 = I135[y0:y1, x0:x1]

            m0 = float(a0.mean()) if a0.size else 0.0
            m90 = float(a90.mean()) if a90.size else 0.0
            m45 = float(a45.mean()) if a45.size else 0.0
            m135 = float(a135.mean()) if a135.size else 0.0

            x = (m0 - m90) / (m0 + m90 + eps)
            y = (m45 - m135) / (m45 + m135 + eps)
            phi = float(0.5 * np.arctan2(y, x))
            vals.append((float(x), float(y), phi))

        with self._analysis_lock:
            # If analysis was reset while we were computing, drop this frame's update.
            if self._spot_xy_series_all is not xy_series_all:
                return
            if self._xy_frames_processed >= frame_idx:
                return
            n = min(len(vals), len(xy_series_all), len(phi_series_all))
            for i in range(n):
                x, y, phi = vals[i]
                xy_series_all[i].append((x, y))
                phi_series_all[i].append(phi)
            self._xy_frames_processed = frame_idx
            self._phi_frames_processed = frame_idx

    def _init_spot_analysis(
        self,
        s_map_full: np.ndarray,
        s_map_int: np.ndarray,
        centers_full: list[tuple[float, float]],
    ) -> None:
        # Sort spots by S value (descending) so "Spot 1" is the strongest candidate.
        if s_map_int is not None and centers_full:
            scored: list[tuple[float, tuple[float, float]]] = []
            h_int, w_int = s_map_int.shape
            h_full, w_full = s_map_full.shape
            sx = (float(w_int - 1) / float(max(1, w_full - 1))) if w_int > 1 else 0.0
            sy = (float(h_int - 1) / float(max(1, h_full - 1))) if h_int > 1 else 0.0
            for cx, cy in centers_full:
                ix = int(round(float(cx) * sx))
                iy = int(round(float(cy) * sy))
                if 0 <= ix < w_int and 0 <= iy < h_int:
                    s_val = float(s_map_int[iy, ix])
                else:
                    s_val = float("-inf")
                scored.append((s_val, (cx, cy)))
            scored.sort(key=lambda t: t[0], reverse=True)
            centers_full = [c for _s, c in scored]

        self._s_map = s_map_full
        self._s_map_int = s_map_int
        self._spot_centers_all = centers_full
        self._spot_centers = list(centers_full)
        self._spot_inspect_overrides = {}
        self._spot_idx = 0
        self._spot_window_cache = [None for _ in centers_full]
        self._spot_window_cache_size = int(self._spot_window_size)
        self._spot_view_cache = [
            {
                "spot_u8": None,
                "spot_win": None,
                "phi_len": None,
                "xy_len": None,
                "fft": None,
                "phi": None,
                "xy": None,
                "dir_len": None,
                "dir_psd": None,
                "dir_hand": None,
                "dir_B": None,
            }
            for _ in centers_full
        ]
        with self._analysis_lock:
            self._spot_phi_series_all = [[] for _ in centers_full]
            self._spot_phi_series = list(self._spot_phi_series_all)
            self._phi_frames_processed = 0
            self._spot_xy_series_all = [[] for _ in centers_full]
            self._spot_xy_series = list(self._spot_xy_series_all)
            self._xy_frames_processed = 0

        # Seed XY+phi using already-available frames.
        # - AVI: from cached decoded frames.
        # - NPY: replay the first S-map frames from the on-disk array (no caching).
        with self._gray_lock:
            gray_frames = list(self._gray_frames)

        seed_kind = None
        seed_count = 0
        if centers_full and gray_frames:
            seed_kind = "list"
            seed_count = len(gray_frames)
        elif centers_full and (self.source_kind == "npy") and self.npy_frames is not None and self.npy_has_frames_dim:
            seed_kind = "npy"
            seed_count = int(min(int(self._st2_frames), int(self.frame_count)))

        if centers_full and seed_kind and seed_count > 0:
            raw_shape = gray_frames[0].shape if gray_frames else self._source_shape
            if raw_shape is None:
                return
            with self._analysis_lock:
                self._spot_xy_series_all = [[] for _ in centers_full]
                self._spot_xy_series = list(self._spot_xy_series_all)
                self._xy_frames_processed = 0
                self._spot_phi_series_all = [[] for _ in centers_full]
                self._spot_phi_series = list(self._spot_phi_series_all)
                self._phi_frames_processed = 0
                self._update_spot_bounds_intensity(tuple(raw_shape))

                used = 0
                if seed_kind == "list":
                    for t in range(seed_count):
                        self._append_xy_from_frame(gray_frames[t])
                        used += 1
                else:
                    # NPY replay (first seed_count frames)
                    for t in range(seed_count):
                        gray = _to_gray_u8(self.npy_frames[t])
                        if gray is None:
                            break
                        self._append_xy_from_frame(gray)
                        used += 1

                self._xy_frames_processed = used
                self._phi_frames_processed = used

        self._sort_spots_by_xy_range()
        # Refresh the overview image + overlay on the UI thread.
        self._ui_call(self._set_smap_background)
        self._ui_call(self._rebuild_smap_overlay)


    def _make_fft_image(self, phi_series: list[float], fps: float) -> Image.Image:
        width = 300
        height = self._spot_window_size * self._spot_scale
        margin_left = 36
        margin_right = 10
        margin_top = 10
        margin_bottom = 28
        plot_w = max(1, width - margin_left - margin_right)
        plot_h = max(1, height - margin_top - margin_bottom)
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + plot_w - 1
        y1 = margin_top + plot_h - 1
        draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0))
        draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0))

        use_fps = float(fps) if fps and fps > 0.0 else 1.0
        unit_label = "Hz" if fps and fps > 0.0 else "cycles/frame"

        if len(phi_series) < 2:
            draw.text((margin_left + 4, margin_top + 4), "Waiting for frames", fill=(0, 0, 0))
            draw.text((margin_left + 4, height - margin_bottom + 6), f"freq ({unit_label})", fill=(0, 0, 0))
            draw.text((4, margin_top), "amp", fill=(0, 0, 0))
            return img

        arr = np.array(phi_series, dtype=np.float32)
        arr = arr - float(arr.mean())
        mag = np.abs(np.fft.rfft(arr))
        if mag.size == 0:
            return img
        mag[0] = 0.0
        max_mag = float(np.max(mag))
        if max_mag <= 0.0:
            return img
        mag = mag / max_mag

        freqs = np.fft.rfftfreq(arr.size, d=1.0 / use_fps)
        f_max = float(freqs[-1]) if freqs.size else 0.0

        last = None
        for x in range(plot_w):
            if mag.size == 1:
                idx = 0
            else:
                idx = int(round(x * (mag.size - 1) / (plot_w - 1)))
            y = y1 - int(mag[idx] * (plot_h - 1))
            pt = (x0 + x, y)
            if last is not None:
                draw.line([last, pt], fill=(0, 0, 0))
            last = pt

        ticks = [0.0, f_max * 0.5, f_max]
        for f in ticks:
            if f_max > 0.0:
                tx = x0 + int(round((f / f_max) * (plot_w - 1)))
            else:
                tx = x0
            draw.line([(tx, y1), (tx, y1 + 4)], fill=(0, 0, 0))
            draw.text((tx - 6, y1 + 6), f"{f:.2f}", fill=(0, 0, 0))

        draw.text((margin_left + 4, height - margin_bottom + 6), f"freq ({unit_label})", fill=(0, 0, 0))
        draw.text((4, margin_top), "amp", fill=(0, 0, 0))
        return img

    def _make_phi_plot_image(self, phi_series: list[float], fps: float) -> Image.Image:
        width = 300
        height = 140
        margin_left = 36
        margin_right = 10
        margin_top = 10
        margin_bottom = 28
        plot_w = max(1, width - margin_left - margin_right)
        plot_h = max(1, height - margin_top - margin_bottom)
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + plot_w - 1
        y1 = margin_top + plot_h - 1
        draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0))
        draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0))

        use_fps = float(fps) if fps and fps > 0.0 else 1.0
        unit_label = "s" if fps and fps > 0.0 else "frames"

        if len(phi_series) < 2:
            draw.text((margin_left + 4, margin_top + 4), "Waiting for frames", fill=(0, 0, 0))
            draw.text((margin_left + 4, height - margin_bottom + 6), f"time ({unit_label})", fill=(0, 0, 0))
            draw.text((4, margin_top), "phi (rad)", fill=(0, 0, 0))
            return img

        arr = np.array(phi_series, dtype=np.float32)
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v == min_v:
            max_v = min_v + 1.0

        last = None
        n = arr.size
        for x in range(plot_w):
            if n == 1:
                idx = 0
            else:
                idx = int(round(x * (n - 1) / (plot_w - 1)))
            val = arr[idx]
            y = y1 - int(((val - min_v) / (max_v - min_v)) * (plot_h - 1))
            pt = (x0 + x, y)
            if last is not None:
                draw.line([last, pt], fill=(0, 0, 0))
            last = pt

        t_max = (n - 1) / use_fps
        ticks = [0.0, t_max * 0.5, t_max]
        for t in ticks:
            if t_max > 0.0:
                tx = x0 + int(round((t / t_max) * (plot_w - 1)))
            else:
                tx = x0
            draw.line([(tx, y1), (tx, y1 + 4)], fill=(0, 0, 0))
            draw.text((tx - 6, y1 + 6), f"{t:.2f}", fill=(0, 0, 0))

        for val in (min_v, max_v):
            if max_v > min_v:
                ty = y1 - int(((val - min_v) / (max_v - min_v)) * (plot_h - 1))
            else:
                ty = y1
            draw.line([(x0 - 4, ty), (x0, ty)], fill=(0, 0, 0))
            draw.text((2, ty - 6), f"{val:.2f}", fill=(0, 0, 0))

        draw.text((margin_left + 4, height - margin_bottom + 6), f"time ({unit_label})", fill=(0, 0, 0))
        draw.text((4, margin_top), "phi (rad)", fill=(0, 0, 0))
        return img

    def _make_xy_scatter_image(
        self,
        xy_series: list[tuple[float, float]],
        ring_score: Optional[float] = None,
        ring_thr: Optional[float] = None,
    ) -> Image.Image:
        width = 300
        height = 300
        margin_left = 36
        margin_right = 10
        margin_top = 20
        margin_bottom = 28
        plot_w = max(1, width - margin_left - margin_right)
        plot_h = max(1, height - margin_top - margin_bottom)
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + plot_w - 1
        y1 = margin_top + plot_h - 1
        draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0))
        draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0))

        title = "X=(I0-I90)/(I0+I90)  Y=(I45-I135)/(I45+I135)"
        if ring_score is not None and ring_thr is not None:
            title = f"{title}  ring={ring_score:.2f} (min {ring_thr:.2f})"
        draw.text((margin_left, 2), title, fill=(0, 0, 0))

        # Axis range is naturally [-1, 1] for normalized differences.
        def px(v: float) -> int:
            vv = max(-1.0, min(1.0, float(v)))
            return x0 + int(round(((vv + 1.0) / 2.0) * (plot_w - 1)))

        def py(v: float) -> int:
            vv = max(-1.0, min(1.0, float(v)))
            return y1 - int(round(((vv + 1.0) / 2.0) * (plot_h - 1)))

        # Draw axes at 0.
        zx = px(0.0)
        zy = py(0.0)
        draw.line([(zx, y0), (zx, y1)], fill=(210, 210, 210))
        draw.line([(x0, zy), (x1, zy)], fill=(210, 210, 210))

        for (xv, yv) in xy_series:
            draw.point((px(xv), py(yv)), fill=(0, 0, 0))

        # Tick labels
        draw.text((x0 - 10, y1 + 6), "-1", fill=(0, 0, 0))
        draw.text((zx - 6, y1 + 6), "0", fill=(0, 0, 0))
        draw.text((x1 - 12, y1 + 6), "1", fill=(0, 0, 0))

        draw.text((2, y1 - 6), "-1", fill=(0, 0, 0))
        draw.text((2, zy - 6), "0", fill=(0, 0, 0))
        draw.text((2, y0 - 6), "1", fill=(0, 0, 0))

        draw.text((margin_left + plot_w // 2 - 6, height - margin_bottom + 6), "X", fill=(0, 0, 0))
        draw.text((4, margin_top + plot_h // 2 - 6), "Y", fill=(0, 0, 0))

        return img

    def _mpl_fig_to_image(self, fig: "Figure") -> Image.Image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            raw = canvas.tostring_rgb()
            buf = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        else:
            raw = canvas.buffer_rgba()
            buf = np.asarray(raw, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return Image.fromarray(buf)

    def _spotrec_brownian_metrics(self, phi_series: list[float], fs: float) -> Optional[dict]:
        phi = np.asarray(phi_series, dtype=np.float64)
        if phi.ndim != 1:
            return None
        phi = phi[np.isfinite(phi)]
        if phi.size < 20:
            return None
        phi = np.unwrap(phi)

        nphi = int(phi.size)
        max_lag = int(max(5, min(nphi - 1, int(round(0.9 * (nphi - 1))))))
        if max_lag < 5:
            return None
        if max_lag <= 2500:
            lags = np.arange(1, max_lag + 1, dtype=np.int32)
        else:
            m = 2200
            u = np.linspace(0.0, 1.0, m)
            frac = 1.0 - np.power(1.0 - u, 3.0)
            core = 1 + np.round(frac * float(max_lag - 1)).astype(np.int32)
            head = np.arange(1, min(140, max_lag) + 1, dtype=np.int32)
            tail = np.arange(max(1, max_lag - 320), max_lag + 1, dtype=np.int32)
            lags = np.unique(np.concatenate([head, core, tail]).astype(np.int32))

        fs_use = max(1e-9, float(fs))
        tau = lags.astype(np.float64) / fs_use
        msd = np.empty((lags.size,), dtype=np.float64)
        for i, k in enumerate(lags):
            delta = phi[k:] - phi[:-k]
            msd[i] = float(np.mean(delta * delta)) if delta.size else np.nan

        valid = np.isfinite(msd) & np.isfinite(tau) & (msd > 0.0) & (tau > 0.0)
        alpha = float("nan")
        r2_log = float("nan")
        if int(np.count_nonzero(valid)) >= 5:
            lt = np.log(tau[valid])
            lm = np.log(msd[valid])
            p = np.polyfit(lt, lm, 1)
            alpha = float(p[0])
            pred = np.polyval(p, lt)
            ss_res = float(np.sum((lm - pred) ** 2))
            ss_tot = float(np.sum((lm - np.mean(lm)) ** 2))
            r2_log = (1.0 - (ss_res / ss_tot)) if ss_tot > 1e-18 else 1.0

        alpha_tol = 0.25
        r2_min = 0.95
        is_brownian = bool(
            np.isfinite(alpha)
            and np.isfinite(r2_log)
            and (abs(alpha - 1.0) <= alpha_tol)
            and (r2_log >= r2_min)
        )
        return {
            "tau": tau,
            "msd": msd,
            "max_lag": int(max_lag),
            "lag_count": int(lags.size),
            "alpha": alpha,
            "r2_log": r2_log,
            "alpha_tol": float(alpha_tol),
            "r2_min": float(r2_min),
            "is_brownian": is_brownian,
        }

    def _update_spotrec_brownian_label(self, phi_series: list[float], fs: float) -> None:
        m = self._spotrec_brownian_metrics(phi_series, fs)
        if m is None:
            self._spotrec_brownian_var.set("Brownian: -")
            return
        alpha = float(m.get("alpha", float("nan")))
        r2 = float(m.get("r2_log", float("nan")))
        tol = float(m.get("alpha_tol", 0.25))
        r2min = float(m.get("r2_min", 0.95))
        n_lags = int(m.get("lag_count", 0))
        tau = np.asarray(m.get("tau", []), dtype=np.float64)
        max_lag_s = float(np.max(tau)) if tau.size else 0.0
        if np.isfinite(alpha) and np.isfinite(r2):
            verdict = "Likely Brownian" if bool(m.get("is_brownian", False)) else "Not Brownian"
            self._spotrec_brownian_var.set(
                f"Brownian: {verdict} | alpha={alpha:.3f}, R2={r2:.3f} "
                f"(|alpha-1|<={tol:.2f}, R2>={r2min:.2f}) | lags={n_lags}, max lag={max_lag_s:.3g}s"
            )
            return
        self._spotrec_brownian_var.set(
            f"Brownian: need >=5 valid lag points | lags={n_lags}, max lag={max_lag_s:.3g}s"
        )

    def _directionality_metrics(
        self, xy_series: list[tuple[float, float]], fps: float
    ) -> Optional[dict]:
        """
        Compute two-sided PSD of Z=X+iY and directionality index B.
        Integration bounds are symmetric:
          P- = sum_{f<0} PSD(f)
          P+ = sum_{f>0} PSD(f)
        i.e. (-inf..0) and (0..inf) within Welch's available band.
        Returns None if the series is too short or PSD cannot be computed.
        """
        if len(xy_series) < int(self.MIN_DIR_FRAMES):
            return None

        arr = np.asarray(xy_series, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None

        x = arr[:, 0]
        y = arr[:, 1]
        z = x.astype(np.complex64) + 1j * y.astype(np.complex64)

        fs = float(fps) if fps and fps > 0.0 else 1.0
        nperseg = int(min(256, len(z)))
        if nperseg < 8:
            return None
        noverlap = nperseg // 2

        freqs, psd = _safe_welch(z, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)
        if freqs.size == 0 or psd.size == 0:
            return None

        # Sort by frequency for plotting/band selection.
        order = np.argsort(freqs)
        freqs = freqs[order]
        psd = psd[order]

        # Rotation frequency estimate: peak on positive side excluding DC region.
        dc_ex = float(getattr(self, "DIR_EXCLUDE_DC_HZ", 0.0))
        mpos = (freqs > max(0.0, dc_ex)) & np.isfinite(psd)
        if not np.any(mpos):
            mpos = (freqs > 0.0) & np.isfinite(psd)
        f0 = None
        if np.any(mpos):
            idxs = np.where(mpos)[0]
            i_peak = int(idxs[np.argmax(psd[idxs])])
            f0 = float(freqs[i_peak])

        p_plus = float(np.sum(psd[(freqs > 0.0) & np.isfinite(psd)]))
        p_minus = float(np.sum(psd[(freqs < 0.0) & np.isfinite(psd)]))

        eps = 1e-18
        b = (p_plus - p_minus) / (p_plus + p_minus + eps)

        # Optional handedness spectrum via cross spectral density.
        try:
            freqs_xy, pxy = _safe_csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
            freqs_xy = np.asarray(freqs_xy, dtype=np.float64)
            pxy = np.asarray(pxy, dtype=np.complex128)
            order2 = np.argsort(freqs_xy)
            freqs_xy = freqs_xy[order2]
            hand = np.imag(pxy[order2])
        except Exception:
            freqs_xy = np.asarray([], dtype=np.float64)
            hand = np.asarray([], dtype=np.float64)

        return {
            "fs": fs,
            "nperseg": nperseg,
            "freqs": freqs,
            "psd": psd,
            "f0": float(f0) if f0 is not None else None,
            "freq_limit_pos": float(np.max(freqs[freqs > 0.0])) if np.any(freqs > 0.0) else 0.0,
            "freq_limit_neg": float(np.max(np.abs(freqs[freqs < 0.0]))) if np.any(freqs < 0.0) else 0.0,
            "p_plus": p_plus,
            "p_minus": p_minus,
            "B": float(b),
            "freqs_xy": freqs_xy,
            "hand": hand,
        }

    def _directionality_B_only(self, xy_series: list[tuple[float, float]], fps: float) -> Optional[float]:
        """
        Fast directionality score B using only Welch(Z) + two-sided integration.
        Returns None if insufficient data or PSD cannot be computed.
        """
        if len(xy_series) < int(self.MIN_DIR_FRAMES):
            return None

        arr = np.asarray(xy_series, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        x = arr[:, 0]
        y = arr[:, 1]
        z = x.astype(np.complex64) + 1j * y.astype(np.complex64)

        fs = float(fps) if fps and fps > 0.0 else 1.0
        nperseg = int(min(256, len(z)))
        if nperseg < 8:
            return None
        noverlap = nperseg // 2

        freqs, psd = _safe_welch(z, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)
        order = np.argsort(freqs)
        freqs = freqs[order]
        psd = psd[order]

        p_plus = float(np.sum(psd[(freqs > 0.0) & np.isfinite(psd)]))
        p_minus = float(np.sum(psd[(freqs < 0.0) & np.isfinite(psd)]))
        eps = 1e-18
        b = (p_plus - p_minus) / (p_plus + p_minus + eps)
        return float(b)

    def _apply_directionality_filter(self, force: bool = False) -> None:
        """
        Optional final filter: keep only spots whose directionality score B exceeds threshold.
        Operates on the current (post-DoG + post-hollowness) list.
        """
        enabled = bool(getattr(self, "_dir_filter_enabled", False))
        bmin = float(self.DIR_FILTER_B_MIN)

        with self._analysis_lock:
            base_centers, base_xy, base_phi = self._dir_filter_base if self._dir_filter_base else ([], [], [])
            # Prefer the saved "post-hollowness" list; fall back to current list if missing.
            centers = list(base_centers) if base_centers else list(self._spot_centers)
            xy_series = list(base_xy) if base_xy else (list(self._spot_xy_series) if self._spot_xy_series else [])
            phi_series = list(base_phi) if base_phi else (list(self._spot_phi_series) if self._spot_phi_series else [])

        if (not enabled) or (not centers) or (not xy_series):
            return

        keep = []
        for i, series in enumerate(xy_series):
            b = self._directionality_B_only(series, self.source_fps)
            if b is None:
                continue
            if abs(float(b)) >= bmin:
                keep.append(i)

        if not keep:
            msg = f"Dir filter kept 0/{len(centers)} (|B|>{bmin:.2f})."
            self._ui_call(self.bottom_var.set, msg)
            if not force:
                return
            with self._analysis_lock:
                self._spot_centers = []
                self._spot_xy_series = []
                self._spot_phi_series = []
                self._spot_idx = 0
                self._spot_window_cache = []
                self._spot_window_cache_size = None
                self._spot_view_cache = []
            self._ui_call(self._rebuild_smap_overlay)
            self._ui_call(self._update_spot_view)
            return

        filt_centers = [centers[i] for i in keep]
        filt_xy = [xy_series[i] for i in keep]
        filt_phi = [phi_series[i] for i in keep] if phi_series else []

        with self._analysis_lock:
            self._spot_centers = filt_centers
            self._spot_xy_series = filt_xy
            self._spot_phi_series = filt_phi
            self._spot_idx = 0
            self._spot_window_cache = [None for _ in self._spot_centers]
            self._spot_window_cache_size = int(self._spot_window_size)
            self._spot_view_cache = []

        msg = f"Dir filter kept {len(keep)}/{len(centers)} (|B|>{bmin:.2f})."
        self._ui_call(self.bottom_var.set, msg)
        self._ui_call(self._rebuild_smap_overlay)
        self._ui_call(self._update_spot_view)


    def _make_directionality_psd_image(self, m: dict) -> Image.Image:
        if Figure is None or FigureCanvas is None:
            return Image.new("RGB", (300, 220), color=(255, 255, 255))
        fig = Figure(figsize=(3.2, 2.3), dpi=100)
        ax = fig.add_subplot(111)
        freqs = np.asarray(m.get("freqs", []), dtype=np.float64)
        psd = np.asarray(m.get("psd", []), dtype=np.float64)
        f0_raw = m.get("f0", None)
        f0 = float(f0_raw) if f0_raw is not None else None
        b = float(m["B"])
        if freqs.size == 0 or psd.size == 0:
            ax.set_title("Two-sided PSD of Z=X+iY  B=n/a  f0=n/a", fontsize=9)
            ax.set_xlabel("freq (Hz)")
            ax.set_ylabel("PSD")
            ax.tick_params(labelsize=8)
            fig.tight_layout(pad=0.6)
            return self._mpl_fig_to_image(fig)
        limit_pos = float(m.get("freq_limit_pos", freqs[-1]))
        limit_neg = float(m.get("freq_limit_neg", abs(freqs[0])))
        # Decimate for speed if needed.
        if freqs.size > 3000:
            step = int(np.ceil(freqs.size / 3000))
            freqs_p = freqs[::step]
            psd_p = psd[::step]
        else:
            freqs_p = freqs
            psd_p = psd
        ax.plot(freqs_p, psd_p, lw=1.0)
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
        span_pos = min(limit_pos, float(freqs[-1]))
        span_neg = min(limit_neg, float(abs(freqs[0])))
        if np.isfinite(span_pos) and span_pos > 0.0:
            try:
                ax.axvspan(0.0, span_pos, color="tab:green", alpha=0.15)
            except Exception:
                pass
        if np.isfinite(span_neg) and span_neg > 0.0:
            try:
                ax.axvspan(-span_neg, 0.0, color="tab:green", alpha=0.15)
            except Exception:
                pass
        if f0 is None:
            f0_label = "n/a"
        else:
            f0_label = f"{abs(f0):.2f}Hz"
        ax.set_title(f"Two-sided PSD of Z=X+iY  B={b:.2f}  f0={f0_label}", fontsize=9)
        ax.set_xlabel("freq (Hz)")
        ax.set_ylabel("PSD")
        ax.tick_params(labelsize=8)
        fig.tight_layout(pad=0.6)
        return self._mpl_fig_to_image(fig)

    def _make_handedness_image(self, m: dict) -> Image.Image:
        if Figure is None or FigureCanvas is None:
            return Image.new("RGB", (300, 220), color=(255, 255, 255))
        fig = Figure(figsize=(3.2, 2.3), dpi=100)
        ax = fig.add_subplot(111)
        freqs = m["freqs_xy"]
        hand = m["hand"]
        if freqs is None or hand is None or len(freqs) == 0 or len(hand) == 0:
            ax.set_title("Handedness spectrum: Im{CSD(X,Y)}", fontsize=9)
            ax.set_xlabel("freq (Hz)")
            ax.set_ylabel("Im(Sxy)")
            ax.tick_params(labelsize=8)
            fig.tight_layout(pad=0.6)
            return self._mpl_fig_to_image(fig)
        if freqs.size > 3000:
            step = int(np.ceil(freqs.size / 3000))
            freqs_p = freqs[::step]
            hand_p = hand[::step]
        else:
            freqs_p = freqs
            hand_p = hand
        ax.plot(freqs_p, hand_p, lw=1.0)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.set_title("Handedness spectrum: Im{CSD(X,Y)}", fontsize=9)
        ax.set_xlabel("freq (Hz)")
        ax.set_ylabel("Im(Sxy)")
        ax.tick_params(labelsize=8)
        fig.tight_layout(pad=0.6)
        return self._mpl_fig_to_image(fig)

    def _recompute_xy_series(self, gray_frames=None, centers=None) -> None:
        if gray_frames is None:
            if self.source_kind == "avi":
                with self._gray_lock:
                    gray_frames = list(self._gray_frames)
            else:
                gray_frames = []
        with self._analysis_lock:
            if centers is not None:
                self._spot_centers_all = centers
            self._spot_xy_series_all = [[] for _ in self._spot_centers_all]
            self._spot_phi_series_all = [[] for _ in self._spot_centers_all]
            # Start with unfiltered view = all candidates.
            self._spot_centers = list(self._spot_centers_all)
            self._spot_xy_series = list(self._spot_xy_series_all)
            self._spot_phi_series = list(self._spot_phi_series_all)
            used = 0
            if gray_frames:
                self._update_spot_bounds_intensity(gray_frames[0].shape)
                for g in gray_frames:
                    self._append_xy_from_frame(g)
                    used += 1
            elif self.source_kind == "npy" and self.npy_frames is not None:
                raw_shape = self._source_shape
                if raw_shape is not None:
                    self._update_spot_bounds_intensity(raw_shape)
                if self.npy_has_frames_dim:
                    n = int(self._playback_frame_count())
                    for i in range(n):
                        g = _to_gray_u8(self.npy_frames[i])
                        if g is None:
                            break
                        self._append_xy_from_frame(g)
                        used += 1
                else:
                    g = _to_gray_u8(self.npy_frames)
                    if g is not None:
                        self._append_xy_from_frame(g)
                        used = 1
            self._xy_frames_processed = used
            self._phi_frames_processed = used

    def _apply_spot_params(self) -> None:
        try:
            k_std = float(self._dog_k_var.get())
            win = int(self._spot_win_var.get())
            ring_score_min = float(self._ring_score_min_var.get())
        except ValueError:
            messagebox.showerror(
                "Spot analysis",
                "DoG k / ring score must be numbers and window must be an integer.",
            )
            return

        if k_std < 0.0:
            messagebox.showerror("Spot analysis", "DoG k must be >= 0.")
            return
        if win < 3:
            messagebox.showerror("Spot analysis", "Phi window must be >= 3.")
            return
        if win % 2 == 0:
            messagebox.showerror("Spot analysis", "Phi window must be an odd integer (e.g. 19).")
            return
        if not (0.0 <= ring_score_min <= 1.0):
            messagebox.showerror("Spot analysis", "Min hollowness score must be between 0 and 1.")
            return

        k_std = float(k_std)
        ring_score_min = float(ring_score_min)
        recompute_needed = (
            k_std != float(self._dog_k_std) or win != int(self._spot_window_size)
        )
        self._dog_k_std = k_std
        self._spot_window_size = win
        self._ring_score_min = ring_score_min
        self._dog_k_var.set(f"{self._dog_k_std:.2f}")
        self._spot_win_var.set(str(win))
        self._ring_score_min_var.set(f"{ring_score_min:.2f}")
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []

        if self._s_map is None:
            messagebox.showinfo("Spot analysis", "S_map is not ready yet.")
            return

        if not recompute_needed and self._spot_centers:
            # Only ring score changed; re-apply ring filter to current candidates.
            self._apply_ring_filter(force=True)
            self._update_spot_view()
            self._refresh_stationary_candidates(True)
            return

        centers = self._find_centers_on_s_map(self._s_map)
        self._set_selected_center_override(None, source="analysis")
        self._spot_idx = 0
        self._spot_inspect_overrides = {}
        self._recompute_xy_series(centers=centers)
        self._apply_ring_filter(force=True)
        self._show_st2_popup(self._s_map)
        self._update_spot_view()
        self._refresh_stationary_candidates(False)

    def _update_spot_view(self) -> None:
        if not self._spot_centers or self._s_map is None:
            self._spot_status_var.set("Spot 0 / 0")
            self.spot_prev_btn.configure(state=tk.DISABLED)
            self.spot_next_btn.configure(state=tk.DISABLED)
            self._spot_img_label.configure(image="")
            self._fft_img_label.configure(image="")
            self._phi_img_label.configure(image="")
            self._xy_img_label.configure(image="")
            if hasattr(self, "_dir_var"):
                self._dir_var.set("B: -")
            if hasattr(self, "_dir_psd_label"):
                self._dir_psd_label.configure(image="")
            if hasattr(self, "_dir_hand_label"):
                self._dir_hand_label.configure(image="")
            self._spot_img_ref = None
            self._fft_img_ref = None
            self._phi_img_ref = None
            self._xy_img_ref = None
            self._dir_psd_ref = None
            self._dir_hand_ref = None
            self._update_spotrec_label()
            return

        n = len(self._spot_centers)
        self.spot_prev_btn.configure(state=tk.NORMAL)
        self.spot_next_btn.configure(state=tk.NORMAL)
        self._set_spot_playback_enabled(True)
        self._spot_idx = max(0, min(self._spot_idx, n - 1))
        cx, cy = self._spot_centers[self._spot_idx]
        self._update_spotrec_label()

        cache_ok = (
            self._spot_window_cache_size == int(self._spot_window_size)
            and len(self._spot_window_cache) == n
        )
        if not cache_ok:
            self._spot_window_cache = [None for _ in range(n)]
            self._spot_window_cache_size = int(self._spot_window_size)
            cache_ok = True
        u8 = None
        cache_entry = None
        if self._spot_view_cache and len(self._spot_view_cache) == n:
            cache_entry = self._spot_view_cache[self._spot_idx]
        if cache_entry and cache_entry.get("spot_win") == int(self._spot_window_size):
            u8 = cache_entry.get("spot_u8")
        if u8 is None and cache_ok:
            u8 = self._spot_window_cache[self._spot_idx]
        if u8 is None:
            window = self._extract_window(self._s_map, cx, cy, self._spot_window_size)
            u8 = detect_spinners.to_u8_preview(window, lo_pct=0.0, hi_pct=100.0)
            if cache_ok:
                self._spot_window_cache[self._spot_idx] = u8
            if cache_entry is not None:
                cache_entry["spot_u8"] = u8
                cache_entry["spot_win"] = int(self._spot_window_size)
        size_px = self._spot_window_size * self._spot_scale
        spot_img = Image.fromarray(u8).resize(
            (size_px, size_px), resample=Image.NEAREST
        )
        spot_img = spot_img.convert("RGB")

        spot_img_tk = ImageTk.PhotoImage(spot_img)
        self._spot_img_label.configure(image=spot_img_tk)
        self._spot_img_ref = spot_img_tk

        spot_key = self._spot_center_key((cx, cy))
        plot_fps = float(self.source_fps)
        with self._analysis_lock:
            phi_series = list(self._spot_phi_series[self._spot_idx]) if self._spot_phi_series else []
            xy_series = list(self._spot_xy_series[self._spot_idx]) if self._spot_xy_series else []
            over = self._spot_inspect_overrides.get(spot_key)
            if isinstance(over, dict):
                over_phi = over.get("phi")
                over_xy = over.get("xy")
                over_fps = over.get("fps")
                if isinstance(over_phi, list):
                    phi_series = list(over_phi)
                if isinstance(over_xy, list):
                    xy_series = list(over_xy)
                if over_fps is not None:
                    try:
                        of = float(over_fps)
                        if of > 0.0:
                            plot_fps = of
                    except Exception:
                        pass
        cache_entry = None
        if self._spot_view_cache and len(self._spot_view_cache) == n:
            cache_entry = self._spot_view_cache[self._spot_idx]
        fps_key = round(float(plot_fps), 6) if plot_fps and plot_fps > 0.0 else 0.0
        phi_len = len(phi_series)
        fft_img = None
        if (
            cache_entry is not None
            and cache_entry.get("phi_len") == phi_len
            and cache_entry.get("phi_fps") == fps_key
        ):
            fft_img = cache_entry.get("fft")
        if fft_img is None:
            fft_img = self._make_fft_image(phi_series, plot_fps)
            if cache_entry is not None:
                cache_entry["fft"] = fft_img
                cache_entry["phi_len"] = phi_len
                cache_entry["phi_fps"] = fps_key
        fft_img_tk = ImageTk.PhotoImage(fft_img)
        self._fft_img_label.configure(image=fft_img_tk)
        self._fft_img_ref = fft_img_tk

        phi_img = None
        if (
            cache_entry is not None
            and cache_entry.get("phi_len") == phi_len
            and cache_entry.get("phi_fps") == fps_key
        ):
            phi_img = cache_entry.get("phi")
        if phi_img is None:
            phi_img = self._make_phi_plot_image(phi_series, plot_fps)
            if cache_entry is not None:
                cache_entry["phi"] = phi_img
                cache_entry["phi_len"] = phi_len
                cache_entry["phi_fps"] = fps_key
        phi_img_tk = ImageTk.PhotoImage(phi_img)
        self._phi_img_label.configure(image=phi_img_tk)
        self._phi_img_ref = phi_img_tk

        ring_score = self._ring_likeness_score(xy_series) if xy_series else 0.0
        xy_len = len(xy_series)
        xy_img = None
        if cache_entry is not None and cache_entry.get("xy_len") == xy_len:
            xy_img = cache_entry.get("xy")
        if xy_img is None:
            xy_img = self._make_xy_scatter_image(
                xy_series, ring_score=ring_score, ring_thr=self._ring_score_min
            )
            if cache_entry is not None:
                cache_entry["xy"] = xy_img
                cache_entry["xy_len"] = xy_len
        xy_img_tk = ImageTk.PhotoImage(xy_img)
        self._xy_img_label.configure(image=xy_img_tk)
        self._xy_img_ref = xy_img_tk

        # Directionality analysis (unidirectional rotation vs back-and-forth)
        try:
            if Figure is None or FigureCanvas is None:
                self._dir_var.set("B: - (install matplotlib)")
                if hasattr(self, "_dir_psd_label"):
                    self._dir_psd_label.configure(image="")
                if hasattr(self, "_dir_hand_label"):
                    self._dir_hand_label.configure(image="")
                self._dir_psd_ref = None
                self._dir_hand_ref = None
            else:
                dir_img = None
                hand_img = None
                b_val = None
                if (
                    cache_entry is not None
                    and cache_entry.get("dir_len") == xy_len
                    and cache_entry.get("dir_fps") == fps_key
                ):
                    dir_img = cache_entry.get("dir_psd")
                    hand_img = cache_entry.get("dir_hand")
                    b_val = cache_entry.get("dir_B")
                if dir_img is None or hand_img is None or b_val is None:
                    m = self._directionality_metrics(xy_series, plot_fps)
                    if m is None:
                        self._dir_var.set("B: - (waiting for frames)")
                        if hasattr(self, "_dir_psd_label"):
                            self._dir_psd_label.configure(image="")
                        if hasattr(self, "_dir_hand_label"):
                            self._dir_hand_label.configure(image="")
                        self._dir_psd_ref = None
                        self._dir_hand_ref = None
                    else:
                        b_val = float(m["B"])
                        dir_img = self._make_directionality_psd_image(m)
                        hand_img = self._make_handedness_image(m)
                        if cache_entry is not None:
                            cache_entry["dir_len"] = xy_len
                            cache_entry["dir_fps"] = fps_key
                            cache_entry["dir_psd"] = dir_img
                            cache_entry["dir_hand"] = hand_img
                            cache_entry["dir_B"] = b_val
                if dir_img is not None and hand_img is not None and b_val is not None:
                    self._dir_var.set(f"B: {b_val:+.3f}   (+1 => +f, -1 => -f)")
                    dir_tk = ImageTk.PhotoImage(dir_img)
                    hand_tk = ImageTk.PhotoImage(hand_img)
                    self._dir_psd_label.configure(image=dir_tk)
                    self._dir_hand_label.configure(image=hand_tk)
                    self._dir_psd_ref = dir_tk
                    self._dir_hand_ref = hand_tk
        except Exception as exc:
            err = f"{exc.__class__.__name__}: {exc}"
            if getattr(self, "_dir_error_last", None) != err:
                self._dir_error_last = err
                if hasattr(self, "bottom_var"):
                    self.bottom_var.set(f"Dir plot error: {err}")
                traceback.print_exc()
            self._dir_var.set("B: - (dir plot error)")
            if hasattr(self, "_dir_psd_label"):
                self._dir_psd_label.configure(image="")
            if hasattr(self, "_dir_hand_label"):
                self._dir_hand_label.configure(image="")
            self._dir_psd_ref = None
            self._dir_hand_ref = None

        self._spot_status_var.set(f"Spot {self._spot_idx + 1} / {n}")

    def _prev_spot(self) -> None:
        if not self._spot_centers:
            return
        self._set_selected_center_override(None, source="analysis")
        self._spot_idx = (self._spot_idx - 1) % len(self._spot_centers)
        self._update_spot_view()
        self._rebuild_smap_overlay()

    def _next_spot(self) -> None:
        if not self._spot_centers:
            return
        self._set_selected_center_override(None, source="analysis")
        self._spot_idx = (self._spot_idx + 1) % len(self._spot_centers)
        self._update_spot_view()
        self._rebuild_smap_overlay()

    def _stop_spot_playback(self) -> None:
        self._play_running = False
        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except Exception:
                pass
        self._play_after_id = None

    def _close_spot_playback(self) -> None:
        self._stop_spot_playback()
        if self._play_popup is not None:
            try:
                self._play_popup.destroy()
            except Exception:
                pass
        self._play_popup = None
        self._play_raw_label = None
        self._play_s_label = None
        self._play_raw_ref = None
        self._play_s_ref = None
        # Keep playback caches to avoid GC spikes that slow spot flipping.
        # They will be reused if the popout is opened again.

    def _playback_frame_count(self) -> int:
        # AVI: play only what has been decoded so far.
        if self.source_kind == "avi":
            with self._gray_lock:
                return int(len(self._gray_frames))
        # NPY: do not cache frames; read from the array on-demand.
        if self.source_kind == "npy":
            if self.npy_frames is None:
                return 0
            if self.npy_has_frames_dim:
                if self.decode_done:
                    return int(self.frame_count)
                return int(max(0, min(int(self.proc_done), int(self.frame_count))))
            return 1
        return 0

    def _get_playback_gray_frame(self, i: int) -> Optional[np.ndarray]:
        if self.source_kind == "avi":
            with self._gray_lock:
                if 0 <= i < len(self._gray_frames):
                    return self._gray_frames[i]
            return None
        if self.source_kind == "npy":
            if self.npy_frames is None:
                return None
            if self.npy_has_frames_dim:
                n = int(self.frame_count)
                if n <= 0:
                    return None
                frame = self.npy_frames[int(i) % n]
            else:
                frame = self.npy_frames
            gray = _to_gray_u8(frame)
            if gray is None:
                return None
            if self._source_shape is not None and tuple(gray.shape) != tuple(self._source_shape):
                return None
            return self._apply_flat_field(gray, base_dir=Path(self.video_path).parent if self.video_path else None)
        return None

    def _spot_playback_windows(
        self, gray: np.ndarray, cx: float, cy: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (raw_window_u8, s_window_u8) for the current spot and frame.

        - raw_window: extracted directly from the raw mosaic frame (full-res).
        - s_window: per-frame anisotropy energy in "S-map space":
              S_inst = X_sm^2 + Y_sm^2 where X=I0-I90, Y=I45-I135,
          smoothed with the same box filter used for the S-map range tracking.
        """
        win = int(self._spot_window_size)
        raw_win = self._extract_window(gray, cx, cy, win).astype(np.uint8, copy=False)

        # Intensity-plane computations (half-res), then expand back to full-res.
        I0 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I90 = gray[1::2, 1::2]

        ix = int(round(cx / 2.0))
        iy = int(round(cy / 2.0))
        win_int = max(1, int(round(win / 2.0)))
        if win_int % 2 == 0:
            win_int += 1
        half = win_int // 2
        h_int, w_int = I0.shape
        x0 = max(0, ix - half)
        x1 = min(w_int, ix + half + 1)
        y0 = max(0, iy - half)
        y1 = min(h_int, iy + half + 1)

        a0 = I0[y0:y1, x0:x1].astype(np.float32, copy=False)
        a90 = I90[y0:y1, x0:x1].astype(np.float32, copy=False)
        a45 = I45[y0:y1, x0:x1].astype(np.float32, copy=False)
        a135 = I135[y0:y1, x0:x1].astype(np.float32, copy=False)

        X = a0 - a90
        Y = a45 - a135

        if self.S_MAP_SMOOTH_K > 1:
            X = cv2.boxFilter(
                X,
                ddepth=-1,
                ksize=(self.S_MAP_SMOOTH_K, self.S_MAP_SMOOTH_K),
                normalize=True,
                borderType=cv2.BORDER_REPLICATE,
            )
            Y = cv2.boxFilter(
                Y,
                ddepth=-1,
                ksize=(self.S_MAP_SMOOTH_K, self.S_MAP_SMOOTH_K),
                normalize=True,
                borderType=cv2.BORDER_REPLICATE,
            )

        s_int = (X * X) + (Y * Y)
        s_full = np.repeat(np.repeat(s_int, 2, axis=0), 2, axis=1)

        # Center-crop/pad to match the raw window size.
        sh, sw = s_full.shape
        if sh < win or sw < win:
            # Pad with zeros if we hit the borders in intensity space.
            pad_y = max(0, win - sh)
            pad_x = max(0, win - sw)
            s_full = np.pad(
                s_full,
                ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
                mode="constant",
                constant_values=0.0,
            )
            sh, sw = s_full.shape
        sy0 = (sh - win) // 2
        sx0 = (sw - win) // 2
        s_win = s_full[sy0 : sy0 + win, sx0 : sx0 + win]
        s_u8 = detect_spinners.to_u8_preview(
            s_win.astype(np.float32, copy=False), lo_pct=0.0, hi_pct=100.0
        )

        return (raw_win, s_u8)

    def _build_spot_playback_cache(self) -> None:
        if not self._spot_centers:
            self._play_raw_u8 = []
            self._play_s_u8 = []
            return
        with self._gray_lock:
            gray_frames = list(self._gray_frames)
        if not gray_frames:
            self._play_raw_u8 = []
            self._play_s_u8 = []
            return

        cx, cy = self._spot_centers[self._spot_idx]
        raw_list: list[np.ndarray] = []
        s_list: list[np.ndarray] = []
        for g in gray_frames:
            raw_u8, s_u8 = self._spot_playback_windows(g, cx, cy)
            raw_list.append(raw_u8)
            s_list.append(s_u8)

        self._play_raw_u8 = raw_list
        self._play_s_u8 = s_list
        self._play_frame_i = 0

    def _render_spot_playback_frame(self) -> None:
        if self._play_popup is None or self._play_raw_label is None or self._play_s_label is None:
            return
        total = self._playback_frame_count()
        if total <= 0:
            return

        i = int(self._play_frame_i) % int(total)
        gray = self._get_playback_gray_frame(i)
        if gray is None or not self._spot_centers:
            return
        cx, cy = self._spot_centers[self._spot_idx]
        raw_u8, s_u8 = self._spot_playback_windows(gray, cx, cy)

        size_px = int(self._spot_window_size) * int(self._spot_scale)
        raw_img = Image.fromarray(raw_u8).resize((size_px, size_px), resample=Image.NEAREST).convert("RGB")
        s_img = Image.fromarray(s_u8).resize((size_px, size_px), resample=Image.NEAREST).convert("RGB")

        raw_tk = ImageTk.PhotoImage(raw_img)
        s_tk = ImageTk.PhotoImage(s_img)
        self._play_raw_label.configure(image=raw_tk)
        self._play_s_label.configure(image=s_tk)
        self._play_raw_ref = raw_tk
        self._play_s_ref = s_tk
        self._play_status_var.set(f"Frame {i + 1}/{total}")
        if raw_u8.size:
            self._play_raw_max_var.set(f"Max raw: {int(raw_u8.max())}")
        else:
            self._play_raw_max_var.set("Max raw: -")

    def _spot_playback_tick(self) -> None:
        if not self._play_running:
            return
        if self._play_popup is None:
            self._stop_spot_playback()
            return
        total = self._playback_frame_count()
        if total <= 0:
            self._stop_spot_playback()
            return

        self._render_spot_playback_frame()
        self._play_frame_i = (int(self._play_frame_i) + 1) % int(total)
        delay_ms = int(round(1000.0 / max(1.0, float(self.PLAYBACK_FPS))))
        self._play_after_id = self.root.after(delay_ms, self._spot_playback_tick)

    def _toggle_spot_playback(self) -> None:
        if self._play_popup is None:
            return
        if self._play_running:
            self._stop_spot_playback()
            self._play_status_var.set("Paused")
            return
        if self._playback_frame_count() <= 0:
            self._play_status_var.set("No frames available yet")
            return
        self._play_running = True
        self._spot_playback_tick()

    def _on_dir_filter_toggle(self) -> None:
        enabled = bool(self._dir_filter_enabled_var.get())
        self._dir_filter_enabled = enabled
        if not enabled:
            # Restore the base list (post-DoG + post-hollowness) if available.
            centers, xy, phi = self._dir_filter_base if self._dir_filter_base else ([], [], [])
            with self._analysis_lock:
                self._spot_centers = list(centers)
                self._spot_xy_series = list(xy) if xy else []
                self._spot_phi_series = list(phi) if phi else []
                self._spot_idx = 0
                self._spot_window_cache = [None for _ in self._spot_centers]
                self._spot_window_cache_size = int(self._spot_window_size)
                self._spot_view_cache = []
            self._rebuild_smap_overlay()
            self._update_spot_view()
            self.bottom_var.set("Dir filter disabled")
            return
        # Apply filtering on demand.
        self._apply_directionality_filter(force=True)

    def _on_abs_range_filter_toggle(self) -> None:
        self._abs_range_filter_enabled = bool(self._abs_range_filter_enabled_var.get())
        if not self._spot_centers_all:
            return
        self._apply_ring_filter(force=True)
        self._update_spot_view()

    def _on_auto_inspect_toggle(self) -> None:
        self._auto_inspect_enabled = bool(self._auto_inspect_enabled_var.get())
        if not self._auto_inspect_enabled:
            return
        # If initial analysis already finished, start inspection immediately
        # without rerunning the main analysis pipeline.
        if bool(getattr(self, "_analysis_finished", False)):
            self._start_auto_inspect_async()

    def _refresh_spot_playback_if_open(self) -> None:
        if self._play_popup is None:
            return
        # Do not rebuild the playback cache here; it is expensive and can stall
        # spot-to-spot flipping. Leave the cache to be built on-demand when the
        # user hits Play in the popout.
        self._stop_spot_playback()
        self._play_raw_u8 = []
        self._play_s_u8 = []
        self._play_frame_i = 0
        if self._play_raw_label is not None:
            self._play_raw_label.configure(image="")
        if self._play_s_label is not None:
            self._play_s_label.configure(image="")
        self._play_raw_ref = None
        self._play_s_ref = None
        self._play_status_var.set("Spot changed - press Play to load")
        self._play_raw_max_var.set("")

    def _open_spot_playback(self) -> None:
        if not self._spot_centers:
            messagebox.showinfo("Spot playback", "No detected spots yet.")
            return

        if self._play_popup is not None:
            try:
                self._play_popup.lift()
                self._play_popup.focus_force()
            except Exception:
                pass
            self._refresh_spot_playback_if_open()
            return

        if self.source_kind == "avi":
            with self._gray_lock:
                have_frames = bool(self._gray_frames)
            if not have_frames:
                messagebox.showinfo("Spot playback", "No decoded frames available yet.")
                return
        elif self.source_kind == "npy":
            if self.npy_frames is None:
                messagebox.showinfo("Spot playback", "No NPY frames loaded.")
                return
        else:
            messagebox.showinfo("Spot playback", "No source loaded.")
            return

        pop = tk.Toplevel(self.root)
        pop.title("Spot playback (Raw + S-space)")
        pop.protocol("WM_DELETE_WINDOW", self._close_spot_playback)
        self._play_popup = pop

        top = ttk.Frame(pop, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Play/Pause", command=self._toggle_spot_playback).pack(side=tk.LEFT)
        ttk.Button(top, text="Close", command=self._close_spot_playback).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(top, textvariable=self._play_status_var).pack(side=tk.LEFT, padx=(10, 0))

        body = ttk.Frame(pop, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(left, text="Raw (mosaic)").pack(side=tk.TOP, anchor="w")
        ttk.Label(right, text="S-space (unnormalised)").pack(side=tk.TOP, anchor="w")

        self._play_raw_label = ttk.Label(left)
        self._play_raw_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(left, textvariable=self._play_raw_max_var).pack(side=tk.TOP, anchor="w", pady=(4, 0))
        self._play_s_label = ttk.Label(right)
        self._play_s_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._render_spot_playback_frame()
        self._play_running = True
        self._spot_playback_tick()

    def _show_finished(self, show: bool):
        if not show:
            self._analysis_finished = False
            if self._auto_inspect_chk is not None:
                self._auto_inspect_chk.state(["disabled"])
            self.bottom_var.set("")
            return
        self._analysis_finished = True
        if self._auto_inspect_chk is not None:
            self._auto_inspect_chk.state(["!disabled"])
        if bool(getattr(self, "_auto_inspect_enabled", False)):
            self._start_auto_inspect_async()
        cur = self.bottom_var.get()
        if cur:
            # Preserve any warning/status message the analysis already set.
            return
        self.bottom_var.set("Processing finished")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select AVI or NPY file",
            filetypes=[
                ("AVI or NumPy", "*.avi *.npy"),
                ("AVI files", "*.avi"),
                ("NumPy files", "*.npy"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self._close_video()
        if not self._load_source(path):
            return

    def _load_source(self, path: str) -> bool:
        suffix = Path(path).suffix.lower()
        if suffix == ".npy":
            return self._load_npy_source(path)
        return self._load_avi_source(path)

    def _load_avi_source(self, path: str) -> bool:
        cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open: {path}")
            return False

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            messagebox.showerror("Error", "Could not read first frame.")
            return False

        gray0 = _to_gray_u8(frame0)
        if gray0 is None:
            cap.release()
            messagebox.showerror("Error", "Could not convert first frame to grayscale.")
            return False
        gray0 = self._apply_flat_field(gray0, base_dir=Path(path).parent)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap = cap
        self.npy_frames = None
        self.npy_has_frames_dim = False
        self.source_kind = "avi"
        self.video_path = path

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.source_fps = fps if fps > 1.0 else 30.0

        self._start_after_load(gray0)
        return True

    def _load_npy_source(self, path: str) -> bool:
        # Detect Git LFS pointer files early so we can give a clear error.
        try:
            with open(path, "rb") as f:
                head = f.read(200)
            if b"version https://git-lfs.github.com/spec/v1" in head:
                messagebox.showerror(
                    "Error",
                    "This .npy is a Git LFS pointer, not the real data. Run `git lfs pull` to download it.",
                )
                return False
        except Exception:
            pass

        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load NPY: {e}")
            return False

        if arr.dtype == object:
            messagebox.showerror("Error", "Unsupported NPY: object arrays are not supported.")
            return False

        if arr.ndim < 2 or arr.ndim > 4:
            messagebox.showerror(
                "Error",
                "Unsupported NPY shape. Expected (H,W), (H,W,C), (N,H,W), or (N,H,W,C).",
            )
            return False

        if arr.ndim == 2:
            frame0 = arr
            frame_count = 1
            has_frames_dim = False
        elif arr.ndim == 3:
            # Disambiguate (H,W,C) single-frame from (N,H,W) frame stacks.
            # Heuristic: if the last axis looks like channels, treat as a single frame.
            if int(arr.shape[-1]) in (1, 3, 4) and int(arr.shape[0]) > 4 and int(arr.shape[1]) > 4:
                frame0 = arr
                frame_count = 1
                has_frames_dim = False
            else:
                if arr.shape[0] < 1:
                    messagebox.showerror("Error", "NPY file has no frames.")
                    return False
                frame0 = arr[0]
                frame_count = int(arr.shape[0])
                has_frames_dim = True
        else:
            if arr.shape[0] < 1:
                messagebox.showerror("Error", "NPY file has no frames.")
                return False
            frame0 = arr[0]
            frame_count = int(arr.shape[0])
            has_frames_dim = True

        gray0 = _to_gray_u8(frame0)
        if gray0 is None:
            messagebox.showerror("Error", "Could not convert first frame to grayscale.")
            return False
        gray0 = self._apply_flat_field(gray0, base_dir=Path(path).parent)

        self.cap = None
        self.npy_frames = arr
        self.npy_has_frames_dim = has_frames_dim
        self.source_kind = "npy"
        self.video_path = path
        self.frame_count = frame_count
        self.source_fps = 30.0

        self._start_after_load(gray0)
        return True

    def _start_after_load(self, gray0: np.ndarray) -> None:
        self._show_finished(False)
        self._analysis_finished = False

        if gray0 is None or gray0.ndim != 2:
            messagebox.showerror("Error", "Source frames must be 2D grayscale.")
            return
        if (gray0.shape[0] % 2) != 0 or (gray0.shape[1] % 2) != 0:
            messagebox.showerror(
                "Error",
                f"Frame shape must be even (polar mosaic). Got {gray0.shape}.",
            )
            return

        self._source_shape = tuple(gray0.shape)
        self._cache_gray_frames = bool(self.source_kind == "avi")
        self.current_idx = 0
        self.last_frame_gray = gray0
        self._overlay_base_frame = gray0.copy()

        # Reset state
        self.proc_done = 0
        self.decode_done = False
        self._clear_queue()
        self.stop_event.clear()

        # Reset spinner popup buffers/state
        self._q_buf = []
        self._u_buf = []
        self._st2_frames = max(2, int(detect_spinners.S_MAP_FRAMES))
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None
        with self._gray_lock:
            self._gray_frames = []
        self._s_map = None
        self._s_map_int = None
        self._spot_centers_all = []
        self._spot_centers = []
        self._spot_phi_series_all = []
        self._spot_phi_series = []
        self._spot_xy_series_all = []
        self._spot_xy_series = []
        self._spot_bounds_int_all = []
        self._spot_inspect_overrides = {}
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._xy_frames_processed = 0
        self._phi_frames_processed = 0
        self._spot_idx = 0
        self._set_selected_center_override(None, source="analysis")
        self._stationary_candidates = []
        self._stationary_idx = 0
        self._spotrec_preview_frame_i = 0
        self._reset_live_tracking(keep_shift=False)
        self._update_spot_view()
        self._update_stationary_view()

        # Start recon thread (consumes queue, processes EVERY frame)
        H, W = gray0.shape
        self.recon_thread = threading.Thread(target=self._recon_worker, args=((H, W),), daemon=True)
        self.recon_thread.start()

        # Start decode thread (produces queue, BLOCKING puts, no drops)
        decode_target = self._decode_worker
        if self.source_kind == "npy":
            decode_target = self._decode_worker_npy
        self.decode_thread = threading.Thread(target=decode_target, daemon=True)
        self.decode_thread.start()

        name = self.video_path.split("/")[-1]
        self.status_var.set(
            f"Loaded: {name}  src≈{self.source_fps:.2f}fps  Q/U: 0%"
        )

        self._status_tick()

    def _status_tick(self):
        if not self.video_path:
            return

        if self.frame_count > 0:
            pct = 100.0 * (self.proc_done / float(self.frame_count))
            pct = max(10.0, min(100.0, pct))
        else:
            pct = 0.0

        name = self.video_path.split("/")[-1]
        self.status_var.set(
            f"Loaded: {name}  src≈{self.source_fps:.2f}fps  Q/U: {pct:.1f}%"
        )
        self.root.after(200, self._status_tick)

    def _decode_worker(self):
        idx = 0
        try:
            while not self.stop_event.is_set() and self.cap is not None:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break

                gray = _to_gray_u8(frame)
                if gray is None:
                    break
                gray = self._apply_flat_field(gray, base_dir=Path(self.video_path).parent if self.video_path else None)

                idx += 1
                self.current_idx = idx
                self.last_frame_gray = gray
                if self._cache_gray_frames:
                    with self._gray_lock:
                        self._gray_frames.append(gray)

                # BLOCK until recon consumes enough space -> guarantees no dropped frames
                while not self.stop_event.is_set():
                    try:
                        self.frame_q.put(gray, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            self.decode_done = True

    def _decode_worker_npy(self):
        idx = 0
        try:
            if self.npy_frames is None:
                return

            expected = self._source_shape
            if self.npy_has_frames_dim:
                total = int(self.frame_count)
                for i in range(total):
                    if self.stop_event.is_set():
                        break
                    frame = self.npy_frames[i]
                    gray = _to_gray_u8(frame)
                    if gray is None:
                        break
                    gray = self._apply_flat_field(
                        gray, base_dir=Path(self.video_path).parent if self.video_path else None
                    )
                    if expected is not None and tuple(gray.shape) != tuple(expected):
                        msg = f"NPY frame shape changed from {expected} to {gray.shape}."
                        self._ui_call(messagebox.showerror, "Error", msg)
                        break
                    idx += 1
                    self.current_idx = idx
                    self.last_frame_gray = gray
                    if self._cache_gray_frames:
                        with self._gray_lock:
                            self._gray_frames.append(gray)
                    while not self.stop_event.is_set():
                        try:
                            self.frame_q.put(gray, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            else:
                if not self.stop_event.is_set():
                    gray = _to_gray_u8(self.npy_frames)
                    if gray is not None:
                        gray = self._apply_flat_field(
                            gray, base_dir=Path(self.video_path).parent if self.video_path else None
                        )
                        if expected is not None and tuple(gray.shape) != tuple(expected):
                            msg = f"NPY frame shape changed from {expected} to {gray.shape}."
                            self._ui_call(messagebox.showerror, "Error", msg)
                            return
                        idx = 1
                        self.current_idx = idx
                        self.last_frame_gray = gray
                        if self._cache_gray_frames:
                            with self._gray_lock:
                                self._gray_frames.append(gray)
                        while not self.stop_event.is_set():
                            try:
                                self.frame_q.put(gray, timeout=0.1)
                                break
                            except queue.Full:
                                continue
        finally:
            self.decode_done = True

    def _recon_worker(self, shape: tuple[int, int]):
        # For the initial S-map, track per-pixel *unnormalised* anisotropy ranges
        # over the first N frames in full-resolution intersection space.
        qu_recon = make_qu_reconstructor(shape, out_dtype=np.float32)
        t_recon_start = time.perf_counter()
        min_x_raw = max_x_raw = None
        min_y_raw = max_y_raw = None
        min_x_sm = max_x_sm = None
        min_y_sm = max_y_sm = None
        x_sm = y_sm = None
        smap_frames_seen = 0
        processed = 0

        try:
            while not self.stop_event.is_set():
                # exit condition: decoding finished and queue drained
                if self.decode_done and self.frame_q.empty():
                    break

                try:
                    gray = self.frame_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Buffer first N frames for initial S-map:
                # S = range(X)^2 + range(Y)^2 over time (per pixel),
                # where X/Y are Q/U on the intersection grid.
                if (not self._st_popup_done) and (smap_frames_seen < self._st2_frames):
                    X, Y = qu_recon(gray)
                    # Spatially average X and Y before range tracking (suppresses isolated noisy pixels).
                    if x_sm is None:
                        x_sm = np.empty_like(X)
                        y_sm = np.empty_like(Y)
                    cv2.boxFilter(
                        X,
                        ddepth=-1,
                        ksize=(self.S_MAP_SMOOTH_K, self.S_MAP_SMOOTH_K),
                        dst=x_sm,
                        normalize=True,
                        borderType=cv2.BORDER_REPLICATE,
                    )
                    cv2.boxFilter(
                        Y,
                        ddepth=-1,
                        ksize=(self.S_MAP_SMOOTH_K, self.S_MAP_SMOOTH_K),
                        dst=y_sm,
                        normalize=True,
                        borderType=cv2.BORDER_REPLICATE,
                    )

                    if min_x_raw is None:
                        min_x_raw = X.copy()
                        max_x_raw = X.copy()
                        min_y_raw = Y.copy()
                        max_y_raw = Y.copy()
                    else:
                        np.minimum(min_x_raw, X, out=min_x_raw)
                        np.maximum(max_x_raw, X, out=max_x_raw)
                        np.minimum(min_y_raw, Y, out=min_y_raw)
                        np.maximum(max_y_raw, Y, out=max_y_raw)

                    if min_x_sm is None:
                        min_x_sm = x_sm.copy()
                        max_x_sm = x_sm.copy()
                        min_y_sm = y_sm.copy()
                        max_y_sm = y_sm.copy()
                    else:
                        np.minimum(min_x_sm, x_sm, out=min_x_sm)
                        np.maximum(max_x_sm, x_sm, out=max_x_sm)
                        np.minimum(min_y_sm, y_sm, out=min_y_sm)
                        np.maximum(max_y_sm, y_sm, out=max_y_sm)

                    smap_frames_seen += 1

                    if smap_frames_seen == self._st2_frames:
                        try:
                            t_stage = time.perf_counter()
                            s_full_raw, _s_int_raw = self._anisotropy_range_s_map(
                                min_x_raw, max_x_raw, min_y_raw, max_y_raw, gray.shape
                            )
                            s_full, s_int = self._anisotropy_range_s_map(
                                min_x_sm, max_x_sm, min_y_sm, max_y_sm, gray.shape
                            )
                            _append_timing_log(f"[SpotAnalysis] S-map compute: {time.perf_counter() - t_stage:.3f}s")
                            # Diagnostics disabled: skip histogram + S-map variant saves.
                            t_centers = time.perf_counter()
                            centers = self._find_centers_on_s_map(s_full)
                            _append_timing_log(f"[SpotAnalysis] Find centers: {time.perf_counter() - t_centers:.3f}s")
                        except Exception as e:
                            self._ui_call(messagebox.showerror, "Spinner detect error", str(e))
                        else:
                            h, w = gray.shape
                            m = int(self.EDGE_EXCLUDE_PX)
                            centers = [
                                (cx, cy)
                                for (cx, cy) in centers
                                if (m <= cx <= (w - 1 - m)) and (m <= cy <= (h - 1 - m))
                            ]
                            t_init = time.perf_counter()
                            self._init_spot_analysis(s_full, s_int, centers)
                            _append_timing_log(f"[SpotAnalysis] Init analysis: {time.perf_counter() - t_init:.3f}s")
                            self._st_popup_done = True
                            self._ui_call(self._show_st2_popup, s_full)
                            self._ui_call(self._update_spot_view)

                frame_idx = processed + 1
                self._append_xy_frame(gray, frame_idx)

                processed += 1
                self.proc_done = processed
        except Exception as e:
            # If recon dies, decoder may be blocked on a full queue; stop it and surface the error.
            self.stop_event.set()
            self._ui_call(messagebox.showerror, "Analysis error", str(e))
            return

        if (not self._st_popup_done) and (smap_frames_seen >= 2) and (min_x_raw is not None):
            # Short clips: build S-map from whatever frames we got.
            try:
                t_stage = time.perf_counter()
                s_full_raw, _s_int_raw = self._anisotropy_range_s_map(
                    min_x_raw, max_x_raw, min_y_raw, max_y_raw, self.last_frame_gray.shape
                )
                s_full, s_int = self._anisotropy_range_s_map(
                    min_x_sm, max_x_sm, min_y_sm, max_y_sm, self.last_frame_gray.shape
                )
                _append_timing_log(
                    f"[SpotAnalysis] (short) S-map compute: {time.perf_counter() - t_stage:.3f}s"
                )
                # Diagnostics disabled: skip S-map variant saves.
                t_centers = time.perf_counter()
                centers = self._find_centers_on_s_map(s_full)
                _append_timing_log(
                    f"[SpotAnalysis] (short) Find centers: {time.perf_counter() - t_centers:.3f}s"
                )
            except Exception:
                centers = []
                s_full = self._s_map
                s_int = self._s_map_int
            h, w = self.last_frame_gray.shape
            m = int(self.EDGE_EXCLUDE_PX)
            centers = [
                (cx, cy)
                for (cx, cy) in centers
                if (m <= cx <= (w - 1 - m)) and (m <= cy <= (h - 1 - m))
            ]
            t_init = time.perf_counter()
            self._init_spot_analysis(s_full, s_int, centers)
            _append_timing_log(f"[SpotAnalysis] (short) Init analysis: {time.perf_counter() - t_init:.3f}s")
            self._st_popup_done = True
            self._ui_call(self._show_st2_popup, s_full)
            self._ui_call(self._update_spot_view)

        if self._spot_centers:
            t_ring = time.perf_counter()
            # Force one initial apply so default-checked filters are active
            # without requiring a manual untick/tick cycle.
            self._apply_ring_filter(force=True)
            _append_timing_log(f"[SpotAnalysis] Ring filter: {time.perf_counter() - t_ring:.3f}s")
            # Overwrite S_map_spots.png with filtered "rotators" view at the end.
            if self._s_map is not None:
                self._ui_call(self._show_st2_popup, self._s_map)
            self._ui_call(self._update_spot_view)
        self._ui_call(self._refresh_stationary_candidates, True)
        self._ui_call(self._show_finished, True)
        _append_timing_log(f"[SpotAnalysis] Total analysis time: {time.perf_counter() - t_recon_start:.3f}s")

    def _clear_queue(self):
        try:
            while True:
                self.frame_q.get_nowait()
        except queue.Empty:
            pass

    def _stop_workers(self):
        self.stop_event.set()
        self.decode_done = True

        if self.decode_thread and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=1.0)
        self.decode_thread = None

        if self.recon_thread and self.recon_thread.is_alive():
            self.recon_thread.join(timeout=1.0)
        self.recon_thread = None

        self._clear_queue()
        self.stop_event.clear()

    def _close_video(self):
        # New-source safety: drop all cached images/buffers so repeated opens don't slow down.
        self._clear_all_caches()
        self._show_finished(False)
        self._stop_spotrec_preview_loop()

        self._close_spot_playback()

        self._stop_workers()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.npy_frames = None
        self.npy_has_frames_dim = False
        self.source_kind = None

        self.video_path = None
        self.frame_count = 0
        self.source_fps = 30.0
        self.current_idx = 0
        self.last_frame_gray = None
        self._live_last_frame = None
        self.proc_done = 0
        self.decode_done = False
        self._reset_live_tracking(keep_shift=False)

        self.status_var.set("No video loaded")
        self.bottom_var.set("")

        self._q_buf = []
        self._u_buf = []
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None
        self._s_map = None
        self._spot_centers_all = []
        self._spot_centers = []
        self._spot_phi_series_all = []
        self._spot_phi_series = []
        self._spot_xy_series_all = []
        self._spot_xy_series = []
        self._spot_bounds_int_all = []
        self._spot_inspect_overrides = {}
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._xy_frames_processed = 0
        self._phi_frames_processed = 0
        self._spot_idx = 0
        self._set_selected_center_override(None, source="analysis")
        self._stationary_candidates = []
        self._stationary_idx = 0
        with self._gray_lock:
            self._gray_frames = []
        self._update_spot_view()
        self._update_stationary_view()
        if self._smap_canvas is not None:
            self._smap_canvas.delete("all")
            self._smap_bg_ref = None
            self._smap_canvas_img_id = None
            self._smap_spot_ring_ids = []
            self._smap_spot_text_ids = []
            if self._smap_overlay_after_id is not None:
                try:
                    self.root.after_cancel(self._smap_overlay_after_id)
                except Exception:
                    pass
            self._smap_overlay_after_id = None
            self._smap_overlay_pending = []

    def on_close(self):
        self._stop_live_feed()
        self._stop_spotrec()
        self._stop_spotrec_preview_loop()
        self._close_video()
        self.root.destroy()

        
if __name__ == "__main__":
    root = tk.Tk()
    app = BasicVideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
