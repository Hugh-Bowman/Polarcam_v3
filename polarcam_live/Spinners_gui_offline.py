# pol_basic_player_min_throttled_display_with_qu_single_decode_process_all.py
import time
from pathlib import Path
import threading
import queue
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
    import matplotlib

    matplotlib.use("Agg")  # offscreen rendering for Tkinter image labels
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
except Exception:  # pragma: no cover
    Figure = None
    FigureCanvas = None

import Detection_alg_offline as detect_spinners
from pol_reconstruction import make_xy_reconstructor


#
# Camera capture defaults (edit these)
#
# NOTE: `exp_ms` is exposure time in milliseconds.
CAMERA_CAPTURE_DEFAULTS = {
    "fps": 40.0,
    "exp_ms": 0.05,  # 0.05 ms = 50 µs
    "analog_gain": 0.0,   # "no gain" (adjust if your camera uses a different baseline)
    "digital_gain": 0.0,  # "no gain"
    "full_fov": True,
    # If the camera outputs 12-bit packed into uint16, use shift=4 for a raw-ish 8-bit.
    # Set to None to fall back to `_to_gray_u8`'s heuristic scaling.
    "u16_to_u8_shift": 4,
}


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


def _camera_frame_to_u8(frame: object, *, u16_to_u8_shift: Optional[int]) -> Optional[np.ndarray]:
    """
    Convert an incoming camera frame to a contiguous (H,W) uint8 mosaic frame.
    """
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3:
        try:
            arr = cv2.extractChannel(arr, 0)
        except Exception:
            arr = arr[..., 0]

    if arr.ndim != 2:
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    if u16_to_u8_shift is not None and np.issubdtype(arr.dtype, np.integer):
        shift = int(max(0, u16_to_u8_shift))
        a = arr.astype(np.uint16, copy=False)
        return np.ascontiguousarray((a >> shift).astype(np.uint8, copy=False))

    return _to_gray_u8(arr)


class BasicVideoPlayer:
    DEFAULT_RING_SCORE_MIN = 0.0
    MIN_RING_FRAMES = 20
    S_MAP_SMOOTH_K = int(detect_spinners.S_MAP_SMOOTH_K)
    EDGE_EXCLUDE_PX = int(detect_spinners.EDGE_EXCLUDE_PX)
    PLAYBACK_FPS = 20.0
    # Overview S-map display scale. 0.5 is "half-res".
    S_MAP_DISPLAY_SCALE = 0.6
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
        try:
            u8 = detect_spinners.to_u8_preview(s_map, lo_pct=0.0, hi_pct=100.0)
        except Exception as e:
            messagebox.showerror("Spinner detect error", str(e))
            return

        img_rgb = Image.fromarray(u8).convert("RGB")

        try:
            out_path = Path.cwd() / "S_map_spots.png"
            img_rgb.save(out_path)
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_map image: {e}")

        if self._overlay_base_frame is not None:
            try:
                raw_rgb = Image.fromarray(self._overlay_base_frame).convert("RGB")
                raw_out_path = Path.cwd() / "frame1_spots.png"
                raw_rgb.save(raw_out_path)
                try:
                    u8_log = self._log_stretch_u8(self._overlay_base_frame)
                    Image.fromarray(u8_log).convert("RGB").save(
                        Path.cwd() / "frame1_spots_log_stretch.png"
                    )
                except Exception as e:
                    messagebox.showwarning(
                        "Save image warning", f"Could not save raw frame log stretch: {e}"
                    )
            except Exception as e:
                messagebox.showwarning(
                    "Save image warning", f"Could not save raw frame image: {e}"
                )

        self._st_popup_img_ref = None
        self._st_popup_label = None

    def _save_raw_s_map_image(self, s_map: np.ndarray) -> None:
        """
        Save the raw S_map (before DoG filtering) for comparison.
        """
        try:
            u8 = detect_spinners.to_u8_preview(s_map, lo_pct=0.0, hi_pct=100.0)
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not preview S_map: {e}")
            return
        try:
            out_path = Path.cwd() / "S_map_raw.png"
            Image.fromarray(u8).convert("RGB").save(out_path)
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save raw S_map image: {e}")

    def _log_stretch_u8(self, img: np.ndarray) -> np.ndarray:
        """
        Log intensity stretch to uint8 for better background visibility.
        """
        x = img.astype(np.float32, copy=False)
        x = np.maximum(x, 0.0)
        vmax = float(np.max(x)) if x.size else 0.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            return np.zeros_like(x, dtype=np.uint8)
        y = np.log1p(x) / np.log1p(vmax)
        return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def _save_s_map_variants(self, s_no_smooth: np.ndarray, s_smoothed: np.ndarray) -> None:
        """
        Save:
          - S_no_smooth: range metric without spatial smoothing
          - S_smoothed: range metric after 5x5 smoothing of X/Y
          - S_DoG: DoG-filtered version of S_smoothed (positive values)
        """
        try:
            u8_raw = detect_spinners.to_u8_preview(s_no_smooth, lo_pct=0.0, hi_pct=100.0)
            Image.fromarray(u8_raw).convert("RGB").save(Path.cwd() / "S_no_smooth.png")
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_no_smooth: {e}")
        else:
            try:
                u8_log = self._log_stretch_u8(s_no_smooth)
                Image.fromarray(u8_log).convert("RGB").save(
                    Path.cwd() / "S_no_smooth_log_stretch.png"
                )
            except Exception as e:
                messagebox.showwarning("Save image warning", f"Could not save S_no_smooth log stretch: {e}")

        try:
            u8_sm = detect_spinners.to_u8_preview(s_smoothed, lo_pct=0.0, hi_pct=100.0)
            Image.fromarray(u8_sm).convert("RGB").save(Path.cwd() / "S_smoothed.png")
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_smoothed: {e}")
        else:
            try:
                u8_log = self._log_stretch_u8(s_smoothed)
                Image.fromarray(u8_log).convert("RGB").save(
                    Path.cwd() / "S_smoothed_log_stretch.png"
                )
            except Exception as e:
                messagebox.showwarning("Save image warning", f"Could not save S_smoothed log stretch: {e}")

        if cv2 is None:
            return
        try:
            g1 = cv2.GaussianBlur(
                s_smoothed.astype(np.float32, copy=False),
                ksize=(0, 0),
                sigmaX=float(detect_spinners.DOG_SIGMA_SMALL),
                borderType=cv2.BORDER_REPLICATE,
            )
            g2 = cv2.GaussianBlur(
                s_smoothed.astype(np.float32, copy=False),
                ksize=(0, 0),
                sigmaX=float(detect_spinners.DOG_SIGMA_LARGE),
                borderType=cv2.BORDER_REPLICATE,
            )
            dog = g1 - g2
            dog = np.maximum(dog, 0.0)
            u8_dog = detect_spinners.to_u8_preview(dog, lo_pct=0.0, hi_pct=100.0)
            Image.fromarray(u8_dog).convert("RGB").save(Path.cwd() / "S_DoG.png")
            try:
                u8_log = self._log_stretch_u8(dog)
                Image.fromarray(u8_log).convert("RGB").save(
                    Path.cwd() / "S_DoG_log_stretch.png"
                )
            except Exception as e:
                messagebox.showwarning("Save image warning", f"Could not save S_DoG log stretch: {e}")
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_DoG: {e}")

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
        s_int = (rx * rx) + (ry * ry)  # (H/2, W/2)

        # Expand to raw pixel grid so spot coordinates remain in full-res space.
        h, w = raw_shape
        s_full = np.repeat(np.repeat(s_int, 2, axis=0), 2, axis=1)
        return (s_full[:h, :w].astype(np.float32, copy=False), s_int.astype(np.float32, copy=False))

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

        prev_n = len(centers_all)
        if not keep:
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
            prof = np.load(p, allow_pickle=False)
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

        # Clear all per-spot rendered images (PhotoImage refs keep memory alive in Tk).
        self._spot_img_ref = None
        self._fft_img_ref = None
        self._phi_img_ref = None
        self._xy_img_ref = None

        # Clear spot caches.
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []

        # Clear decoded frame cache (AVI only; NPY isn't cached anyway).
        with self._gray_lock:
            self._gray_frames = []

        # Clear directionality filter base snapshot.
        self._dir_filter_base = ([], [], [])
        self._dir_filter_enabled = bool(self._dir_filter_enabled_var.get())
        # Clear flat-field cache (profile depends on input frame shape).
        self._flat_profile = None
        self._flat_inv = None
        self._flat_profile_path = None
        self._flat_warned_mismatch = False
        self._flat_field_enabled = bool(self._flat_field_enabled_var.get())

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AVI/NPY Spot Analysis (DoG spot detection)")
        self._ui_thread_id = threading.get_ident()
        self._ui_queue: "queue.Queue[tuple[object, tuple, dict]]" = queue.Queue()
        self._ui_pump_after_id = None
        self._dir_filter_enabled_var = tk.BooleanVar(value=False)
        self._dir_filter_enabled = False
        self._dir_filter_base = ([], [], [])
        # Flat-field / illumination correction
        self._flat_field_enabled_var = tk.BooleanVar(value=False)
        self._flat_field_enabled = False
        self._flat_profile = None  # float32 (H,W)
        self._flat_inv = None      # float32 (H,W) : scale/profile, zeros where profile==0
        self._flat_profile_path = None
        self._flat_warned_mismatch = False

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
        # S-map overview canvas (background + overlay items)
        self._smap_canvas = None
        self._smap_bg_ref = None
        self._smap_canvas_img_id = None
        self._smap_spot_ring_ids = []
        self._smap_spot_text_ids = []
        self._smap_overlay_after_id = None
        self._smap_overlay_pending = []
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
        self._camera_capture_thread = None
        self._camera_capture_busy = False

        # EOF indicator for decoder
        self.decode_done = False

        # Processing progress
        self.proc_done = 0  # frames processed by recon thread

        # Queue: bounded to avoid RAM blow-up, BLOCKING put => no frame drops
        self.frame_q = queue.Queue(maxsize=16)

        # Camera capture UI state
        self._cam_n_frames_var = tk.StringVar(value="200")
        self._cam_save_enabled_var = tk.BooleanVar(value=True)
        self._cam_save_format_var = tk.StringVar(value="npz")  # "npz" | "raw8"
        self._cam_fps_var = tk.StringVar(value=str(float(CAMERA_CAPTURE_DEFAULTS["fps"])))
        self._cam_exp_ms_var = tk.StringVar(value=str(float(CAMERA_CAPTURE_DEFAULTS["exp_ms"])))
        self._cam_analog_gain_var = tk.StringVar(value=str(float(CAMERA_CAPTURE_DEFAULTS["analog_gain"])))
        self._cam_digital_gain_var = tk.StringVar(value=str(float(CAMERA_CAPTURE_DEFAULTS["digital_gain"])))
        self._cam_full_fov_var = tk.BooleanVar(value=bool(CAMERA_CAPTURE_DEFAULTS["full_fov"]))
        shift_default = CAMERA_CAPTURE_DEFAULTS.get("u16_to_u8_shift", None)
        self._cam_u16_shift_var = tk.StringVar(value="" if shift_default is None else str(int(shift_default)))
        self._cam_get_btn = None
        self._cam_n_entry = None
        self._cam_save_check = None
        self._cam_fmt_combo = None
        self._cam_fps_entry = None
        self._cam_exp_entry = None
        self._cam_ag_entry = None
        self._cam_dg_entry = None
        self._cam_shift_entry = None
        self._cam_fullfov_check = None

        self._build_ui()
        self._start_ui_pump()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Select AVI/NPY", command=self.open_video).pack(side=tk.LEFT)

        # Camera capture controls
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        cam = ttk.Frame(top)
        cam.pack(side=tk.LEFT)

        ttk.Label(cam, text="Frames").grid(row=0, column=0, sticky="w")
        self._cam_n_entry = ttk.Entry(cam, textvariable=self._cam_n_frames_var, width=6)
        self._cam_n_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(cam, text="FPS").grid(row=0, column=2, sticky="w")
        self._cam_fps_entry = ttk.Entry(cam, textvariable=self._cam_fps_var, width=6)
        self._cam_fps_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(cam, text="Exp (ms)").grid(row=0, column=4, sticky="w")
        self._cam_exp_entry = ttk.Entry(cam, textvariable=self._cam_exp_ms_var, width=6)
        self._cam_exp_entry.grid(row=0, column=5, sticky="w", padx=(6, 12))

        self._cam_fullfov_check = ttk.Checkbutton(cam, text="Full FOV", variable=self._cam_full_fov_var)
        self._cam_fullfov_check.grid(row=0, column=6, sticky="w")

        ttk.Label(cam, text="A gain").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._cam_ag_entry = ttk.Entry(cam, textvariable=self._cam_analog_gain_var, width=6)
        self._cam_ag_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(4, 0))

        ttk.Label(cam, text="D gain").grid(row=1, column=2, sticky="w", pady=(4, 0))
        self._cam_dg_entry = ttk.Entry(cam, textvariable=self._cam_digital_gain_var, width=6)
        self._cam_dg_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(4, 0))

        ttk.Label(cam, text="U16>>").grid(row=1, column=4, sticky="w", pady=(4, 0))
        self._cam_shift_entry = ttk.Entry(cam, textvariable=self._cam_u16_shift_var, width=6)
        self._cam_shift_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(4, 0))

        self._cam_save_check = ttk.Checkbutton(cam, text="Save", variable=self._cam_save_enabled_var)
        self._cam_save_check.grid(row=1, column=6, sticky="w", pady=(4, 0))

        self._cam_fmt_combo = ttk.Combobox(
            cam, textvariable=self._cam_save_format_var, values=("npz", "raw8"), width=6, state="readonly"
        )
        self._cam_fmt_combo.grid(row=1, column=7, sticky="w", padx=(6, 0), pady=(4, 0))

        self._cam_get_btn = ttk.Button(cam, text="Get frames", command=self.get_frames_from_camera)
        self._cam_get_btn.grid(row=0, column=7, sticky="w", padx=(6, 0))

        ttk.Checkbutton(
            top,
            text=f"Flat-field ({self.FLAT_FIELD_FILENAME})",
            variable=self._flat_field_enabled_var,
            command=self._on_flat_field_toggle,
        ).pack(side=tk.LEFT, padx=(10, 0))
        self.status_var = tk.StringVar(value="No video loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(self.root, padding=8)
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

        # Left: main S_map overview (S_smoothed, scaled by 1/2) + overlay circles/labels.
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
        ttk.Button(nav, text="Play spot", command=self._open_spot_playback).pack(side=tk.RIGHT)

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

        self._spot_update_btn = ttk.Button(controls, text="Update analysis", command=self._apply_spot_params)
        self._spot_update_btn.pack(side=tk.TOP, anchor="w", pady=(8, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(right, text="S-map window").pack(side=tk.TOP, anchor="w")
        self._spot_img_label = ttk.Label(right)
        self._spot_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="Phi FFT").pack(side=tk.TOP, anchor="w")
        self._fft_img_label = ttk.Label(right)
        self._fft_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="Phi(t)").pack(side=tk.TOP, anchor="w")
        self._phi_img_label = ttk.Label(right)
        self._phi_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 10))

        ttk.Label(right, text="X/Y scatter").pack(side=tk.TOP, anchor="w")
        self._xy_img_label = ttk.Label(right)
        self._xy_img_label.pack(side=tk.TOP, anchor="w", pady=(2, 0))

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

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for w in (
            self._cam_get_btn,
            self._cam_n_entry,
            self._cam_fps_entry,
            self._cam_exp_entry,
            self._cam_ag_entry,
            self._cam_dg_entry,
            self._cam_shift_entry,
            self._cam_save_check,
            self._cam_fullfov_check,
        ):
            if w is None:
                continue
            try:
                w.configure(state=state)
            except Exception:
                pass

        if self._cam_fmt_combo is not None:
            try:
                self._cam_fmt_combo.configure(state="readonly" if enabled else "disabled")
            except Exception:
                pass

    def get_frames_from_camera(self) -> None:
        """
        Capture an N-frame burst from the IDS camera and load it into the GUI for processing.
        """
        if bool(getattr(self, "_camera_capture_busy", False)):
            return

        def _parse_float(name: str, s: str) -> float:
            try:
                return float(str(s).strip())
            except Exception:
                raise ValueError(f"{name} must be a number.")

        try:
            n_frames = int(str(self._cam_n_frames_var.get()).strip())
        except Exception:
            messagebox.showerror("Camera", "Cam frames must be an integer.")
            return
        if n_frames < 1:
            messagebox.showerror("Camera", "Cam frames must be >= 1.")
            return

        try:
            fps = _parse_float("FPS", self._cam_fps_var.get())
            exp_ms = _parse_float("Exposure (ms)", self._cam_exp_ms_var.get())
            analog_gain = _parse_float("Analog gain", self._cam_analog_gain_var.get())
            digital_gain = _parse_float("Digital gain", self._cam_digital_gain_var.get())
        except Exception as e:
            messagebox.showerror("Camera", str(e))
            return
        if fps <= 0.0:
            messagebox.showerror("Camera", "FPS must be > 0.")
            return
        if exp_ms <= 0.0:
            messagebox.showerror("Camera", "Exposure (ms) must be > 0.")
            return

        full_fov = bool(self._cam_full_fov_var.get())
        shift_s = str(self._cam_u16_shift_var.get()).strip()
        try:
            u16_to_u8_shift = None if shift_s == "" else int(shift_s)
        except Exception:
            messagebox.showerror("Camera", "U16>> shift must be an integer (or blank).")
            return

        save_enabled = bool(self._cam_save_enabled_var.get())
        save_fmt = str(self._cam_save_format_var.get() or "npz").strip().lower()
        if save_fmt not in ("npz", "raw8"):
            save_fmt = "npz"

        save_path = None
        if save_enabled:
            ts = time.strftime("%Y%m%d-%H%M%S")
            default_name = f"camera_capture_{ts}.{save_fmt if save_fmt != 'raw8' else 'raw'}"
            filetypes = [("NPZ", "*.npz")] if save_fmt == "npz" else [("RAW (uint8)", "*.raw")]
            p = filedialog.asksaveasfilename(
                title="Save captured frames",
                defaultextension=".npz" if save_fmt == "npz" else ".raw",
                initialfile=default_name,
                filetypes=filetypes + [("All files", "*.*")],
            )
            if p:
                save_path = p

        self._camera_capture_busy = True
        self._set_camera_controls_enabled(False)
        self.bottom_var.set(f"Capturing {n_frames} frame(s) from camera…")

        def _worker():
            ctrl = None
            try:
                # Lazy import so the GUI still works on machines without camera deps.
                from Controlling.controller import Controller  # type: ignore
                from PySide6.QtWidgets import QApplication  # type: ignore

                # Ensure a Qt application exists before constructing any QObject-based facades.
                app = QApplication.instance()
                if app is None:
                    app = QApplication([])

                ctrl = Controller()
                ctrl.open()
                if full_fov:
                    ctrl.full_sensor()
                ctrl.set_timing(fps=fps, exp_ms=exp_ms)
                ctrl.set_gains(analog=analog_gain, digital=digital_gain)
                ctrl.start()

                frames: list[np.ndarray] = []
                done = False

                cam = getattr(ctrl, "cam", None)
                frame_sig = getattr(cam, "frame", None)
                if frame_sig is None or not hasattr(frame_sig, "connect"):
                    raise RuntimeError("Camera backend does not expose a Qt `frame` signal.")

                def _on_frame(arr_obj: object) -> None:
                    nonlocal done
                    if done:
                        return
                    u8 = _camera_frame_to_u8(arr_obj, u16_to_u8_shift=u16_to_u8_shift)
                    if u8 is None:
                        return
                    frames.append(u8)
                    if len(frames) >= n_frames:
                        done = True

                frame_sig.connect(_on_frame)
                try:
                    last_ui = 0.0
                    t0 = time.time()
                    fps_eff = float(fps) if fps and fps > 0.0 else 1.0
                    timeout_s = max(5.0, (float(n_frames) / fps_eff) * 3.0)
                    while (not done) and (not self.stop_event.is_set()):
                        app.processEvents()
                        time.sleep(0.002)
                        now = time.time()
                        if (now - t0) > timeout_s:
                            raise TimeoutError(
                                f"Timed out waiting for {n_frames} frame(s) "
                                f"(got {len(frames)} after {timeout_s:.1f}s)."
                            )
                        if (now - last_ui) > 0.1:
                            last_ui = now
                            self._ui_call(
                                self.bottom_var.set,
                                f"Capturing… {len(frames)}/{n_frames}",
                            )
                finally:
                    try:
                        frame_sig.disconnect(_on_frame)
                    except Exception:
                        pass

                if not frames:
                    raise RuntimeError("No frames captured.")

                stack = np.stack(frames, axis=0).astype(np.uint8, copy=False)

                # Stop camera as soon as frames are ready for processing.
                try:
                    ctrl.stop()
                except Exception:
                    pass
                try:
                    ctrl.close()
                except Exception:
                    pass

                # Optional save
                if save_path:
                    if save_fmt == "npz":
                        np.savez_compressed(
                            save_path,
                            frames=stack,
                            fps=np.float32(fps),
                            exp_ms=np.float32(exp_ms),
                            analog_gain=np.float32(analog_gain) if analog_gain is not None else np.float32(np.nan),
                            digital_gain=np.float32(digital_gain) if digital_gain is not None else np.float32(np.nan),
                            u16_to_u8_shift=np.int32(u16_to_u8_shift) if u16_to_u8_shift is not None else np.int32(-1),
                        )
                    else:
                        raw_path = Path(save_path)
                        raw_path.write_bytes(stack.tobytes(order="C"))
                        meta_path = raw_path.with_suffix(raw_path.suffix + ".json")
                        meta_obj = {
                            "dtype": "uint8",
                            "shape": [int(x) for x in stack.shape],
                            "order": "C",
                            "fps": float(fps),
                            "exp_ms": float(exp_ms),
                            "analog_gain": analog_gain,
                            "digital_gain": digital_gain,
                            "u16_to_u8_shift": u16_to_u8_shift,
                        }
                        meta_path.write_text(__import__("json").dumps(meta_obj, indent=2))

                label_path = save_path or "Camera capture"
                self._ui_call(self._load_camera_stack, stack, label_path, fps)
            except Exception as e:
                self._ui_call(messagebox.showerror, "Camera capture error", str(e))
            finally:
                try:
                    if ctrl is not None:
                        ctrl.shutdown()  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._ui_call(self._on_camera_capture_done)

        self._camera_capture_thread = threading.Thread(target=_worker, daemon=True)
        self._camera_capture_thread.start()

    def _on_camera_capture_done(self) -> None:
        self._camera_capture_busy = False
        self._set_camera_controls_enabled(True)

    def _load_camera_stack(self, stack_u8: np.ndarray, label_path: str, fps: float) -> None:
        """
        Load a captured (N,H,W) uint8 stack into the existing NPY pipeline.
        Runs on the Tk/UI thread.
        """
        if stack_u8 is None or stack_u8.ndim != 3:
            messagebox.showerror("Camera", "Captured stack must be (N,H,W).")
            return

        self._close_video()

        self.cap = None
        self.npy_frames = np.ascontiguousarray(stack_u8)
        self.npy_has_frames_dim = True
        self.source_kind = "npy"
        self.video_path = str(label_path)
        self.frame_count = int(stack_u8.shape[0])
        self.source_fps = float(fps) if fps and fps > 0.0 else 30.0

        gray0 = stack_u8[0]
        self._start_after_load(gray0)

    def _on_smap_click(self, event) -> None:
        """
        Click a numbered ring on the S-map overview to jump to that spot.
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

        self._spot_idx = i
        self._update_spot_view()

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
        # Pop from the end for O(1) per element (order doesn't matter for labels).
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
                ring_id = self._smap_canvas.create_oval(
                    x - r, y - r, x + r, y + r, outline="#ff3333", width=2
                )
                text_id = self._smap_canvas.create_text(
                    x + r + 6,
                    y - r - 2,
                    text=str(i + 1),
                    fill="#ff3333",
                    anchor="nw",
                    font=("TkDefaultFont", 12, "bold"),
                )
                self._smap_spot_ring_ids.append(ring_id)
                self._smap_spot_text_ids.append(text_id)
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

        I90 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I0 = gray[1::2, 1::2]

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

        I90 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I0 = gray[1::2, 1::2]

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
            for cx, cy in centers_full:
                ix = int(round(cx / 2.0))
                iy = int(round(cy / 2.0))
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
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        return Image.fromarray(buf)

    def _directionality_metrics(
        self, xy_series: list[tuple[float, float]], fps: float
    ) -> Optional[dict]:
        """
        Compute two-sided PSD of Z=X+iY and directionality index B.
        Integration bounds are symmetric:
          P- = sum_{f<0} PSD(f)
          P+ = sum_{f>0} PSD(f)
        i.e. (-inf..0) and (0..inf) within Welch's available band.
        Returns None if scipy isn't available or series too short.
        """
        if _welch is None or _csd is None:
            return None
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

        freqs, psd = _welch(
            z,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=False,
            scaling="density",
        )
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)

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
        freqs_xy, pxy = _csd(
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
        freqs_xy = np.asarray(freqs_xy, dtype=np.float64)
        pxy = np.asarray(pxy, dtype=np.complex128)
        order2 = np.argsort(freqs_xy)
        freqs_xy = freqs_xy[order2]
        hand = np.imag(pxy[order2])

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
        Returns None if insufficient data or scipy missing.
        """
        if _welch is None:
            return None
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

        freqs, psd = _welch(
            z,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=False,
            scaling="density",
        )
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
        freqs = m["freqs"]
        psd = m["psd"]
        f0 = float(m["f0"])
        b = float(m["B"])
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
        ax.axvspan(0.0, min(limit_pos, freqs[-1]), color="tab:green", alpha=0.15)
        ax.axvspan(-min(limit_neg, abs(freqs[0])), 0.0, color="tab:green", alpha=0.15)
        ax.set_title(f"Two-sided PSD of Z=X+iY  B={b:.2f}  f0={abs(f0):.2f}Hz", fontsize=9)
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
            return

        centers = self._find_centers_on_s_map(self._s_map)
        self._spot_idx = 0
        self._recompute_xy_series(centers=centers)
        self._apply_ring_filter(force=True)
        self._show_st2_popup(self._s_map)
        self._update_spot_view()

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
            return

        n = len(self._spot_centers)
        self.spot_prev_btn.configure(state=tk.NORMAL)
        self.spot_next_btn.configure(state=tk.NORMAL)
        self._spot_idx = max(0, min(self._spot_idx, n - 1))
        cx, cy = self._spot_centers[self._spot_idx]

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

        with self._analysis_lock:
            phi_series = (
                list(self._spot_phi_series[self._spot_idx]) if self._spot_phi_series else []
            )
        cache_entry = None
        if self._spot_view_cache and len(self._spot_view_cache) == n:
            cache_entry = self._spot_view_cache[self._spot_idx]
        phi_len = len(phi_series)
        fft_img = None
        if cache_entry is not None and cache_entry.get("phi_len") == phi_len:
            fft_img = cache_entry.get("fft")
        if fft_img is None:
            fft_img = self._make_fft_image(phi_series, self.source_fps)
            if cache_entry is not None:
                cache_entry["fft"] = fft_img
                cache_entry["phi_len"] = phi_len
        fft_img_tk = ImageTk.PhotoImage(fft_img)
        self._fft_img_label.configure(image=fft_img_tk)
        self._fft_img_ref = fft_img_tk

        phi_img = None
        if cache_entry is not None and cache_entry.get("phi_len") == phi_len:
            phi_img = cache_entry.get("phi")
        if phi_img is None:
            phi_img = self._make_phi_plot_image(phi_series, self.source_fps)
            if cache_entry is not None:
                cache_entry["phi"] = phi_img
                cache_entry["phi_len"] = phi_len
        phi_img_tk = ImageTk.PhotoImage(phi_img)
        self._phi_img_label.configure(image=phi_img_tk)
        self._phi_img_ref = phi_img_tk

        with self._analysis_lock:
            xy_series = (
                list(self._spot_xy_series[self._spot_idx]) if self._spot_xy_series else []
            )
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
        if _welch is None or _csd is None:
            self._dir_var.set("B: - (install scipy)")
            if hasattr(self, "_dir_psd_label"):
                self._dir_psd_label.configure(image="")
            if hasattr(self, "_dir_hand_label"):
                self._dir_hand_label.configure(image="")
            self._dir_psd_ref = None
            self._dir_hand_ref = None
        elif Figure is None or FigureCanvas is None:
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
            if cache_entry is not None and cache_entry.get("dir_len") == xy_len:
                dir_img = cache_entry.get("dir_psd")
                hand_img = cache_entry.get("dir_hand")
                b_val = cache_entry.get("dir_B")
            if dir_img is None or hand_img is None or b_val is None:
                m = self._directionality_metrics(xy_series, self.source_fps)
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

        self._spot_status_var.set(f"Spot {self._spot_idx + 1} / {n}")

    def _prev_spot(self) -> None:
        if not self._spot_centers:
            return
        self._spot_idx = (self._spot_idx - 1) % len(self._spot_centers)
        self._update_spot_view()

    def _next_spot(self) -> None:
        if not self._spot_centers:
            return
        self._spot_idx = (self._spot_idx + 1) % len(self._spot_centers)
        self._update_spot_view()

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
        I90 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I0 = gray[1::2, 1::2]

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
            self.bottom_var.set("")
            return
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
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
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
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._xy_frames_processed = 0
        self._phi_frames_processed = 0
        self._spot_idx = 0
        self._update_spot_view()

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
        # over the first N frames (raw differences I0-I90 and I45-I135).
        xy_recon = make_xy_reconstructor(shape, eps=1e-6, out_dtype=np.float32, normalize=False)
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
                # where X=(I0-I90), Y=(I45-I135).
                if (not self._st_popup_done) and (smap_frames_seen < self._st2_frames):
                    X, Y = xy_recon(gray)
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
                            s_full_raw, _s_int_raw = self._anisotropy_range_s_map(
                                min_x_raw, max_x_raw, min_y_raw, max_y_raw, gray.shape
                            )
                            s_full, s_int = self._anisotropy_range_s_map(
                                min_x_sm, max_x_sm, min_y_sm, max_y_sm, gray.shape
                            )
                            try:
                                detect_spinners.save_s_histogram(
                                    s_int,
                                    out_path=Path.cwd() / "S_map_hist.png",
                                    title="S_map pixel distribution (range(unnormalised anisotropy) metric)",
                                )
                            except Exception as e:
                                self._ui_call(
                                    messagebox.showwarning,
                                    "Histogram warning",
                                    f"Could not save S_map histogram: {e}",
                                )
                            self._save_s_map_variants(s_full_raw, s_full)
                            centers = self._find_centers_on_s_map(s_full)
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
                            self._init_spot_analysis(s_full, s_int, centers)
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
                s_full_raw, _s_int_raw = self._anisotropy_range_s_map(
                    min_x_raw, max_x_raw, min_y_raw, max_y_raw, self.last_frame_gray.shape
                )
                s_full, s_int = self._anisotropy_range_s_map(
                    min_x_sm, max_x_sm, min_y_sm, max_y_sm, self.last_frame_gray.shape
                )
                self._save_s_map_variants(s_full_raw, s_full)
                centers = self._find_centers_on_s_map(s_full)
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
            self._init_spot_analysis(s_full, s_int, centers)
            self._st_popup_done = True
            self._ui_call(self._show_st2_popup, s_full)
            self._ui_call(self._update_spot_view)

        if self._spot_centers:
            self._apply_ring_filter()
            # Overwrite S_map_spots.png with filtered "rotators" view at the end.
            if self._s_map is not None:
                self._ui_call(self._show_st2_popup, self._s_map)
            self._ui_call(self._update_spot_view)
        self._ui_call(self._show_finished, True)

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
        self.proc_done = 0
        self.decode_done = False

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
        self._spot_window_cache = []
        self._spot_window_cache_size = None
        self._spot_view_cache = []
        self._xy_frames_processed = 0
        self._phi_frames_processed = 0
        self._spot_idx = 0
        with self._gray_lock:
            self._gray_frames = []
        self._update_spot_view()
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
        self._close_video()
        self.root.destroy()

        
if __name__ == "__main__":
    root = tk.Tk()
    app = BasicVideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
