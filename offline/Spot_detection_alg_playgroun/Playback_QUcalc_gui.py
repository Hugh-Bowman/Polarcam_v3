# pol_basic_player_min_throttled_display_with_qu_single_decode_process_all.py
import time
import json
import re
from pathlib import Path
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

import detect_spinners
from pol_reconstruction import make_xy_reconstructor


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """
    OpenCV decodes your AVI into BGR with replicated grayscale.
    extractChannel gives a contiguous (H,W) uint8 array.
    """
    if frame is None:
        return None
    if frame.ndim == 3:
        return cv2.extractChannel(frame, 0)  # contiguous
    return frame.astype(np.uint8, copy=False)


class BasicVideoPlayer:
    # Default to no filtering by phi variance (units: rad^2).
    DEFAULT_PHI_VAR_MIN = 0.0
    DEFAULT_RING_SCORE_MIN = 0.0
    MIN_RING_FRAMES = 20
    S_MAP_SMOOTH_K = 5
    EDGE_EXCLUDE_PX = 10

    def _find_centers_on_s_int(self, s_int: np.ndarray) -> list[tuple[float, float]]:
        """
        Spot finding on the intensity-plane S-map, with edge exclusion implemented by cropping
        before thresholding / connected-components.
        """
        if s_int is None:
            return []

        edge_int = int(np.ceil(float(self.EDGE_EXCLUDE_PX) / 2.0))
        h, w = s_int.shape
        if edge_int > 0 and (2 * edge_int) < min(h, w):
            work = s_int[edge_int : h - edge_int, edge_int : w - edge_int]
            offset = edge_int
        else:
            work = s_int
            offset = 0

        min_area_int = max(1, int(round(self._spot_min_area / 4.0)))
        max_area_int = (
            None if self._spot_max_area is None else max(1, int(round(self._spot_max_area / 4.0)))
        )
        centers = detect_spinners.find_spot_centers_smap(
            work,
            percentile=self._spot_percentile,
            min_area=min_area_int,
            max_area=max_area_int,
            connect_radius=self._spot_connect_radius,
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

        img_rgb = Image.fromarray(u8, mode="L").convert("RGB")

        try:
            out_path = Path.cwd() / "S_map_spots.png"
            img_rgb.save(out_path)
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_map image: {e}")

        if self._overlay_base_frame is not None:
            try:
                raw_rgb = Image.fromarray(self._overlay_base_frame, mode="L").convert("RGB")
                raw_out_path = Path.cwd() / "frame1_spots.png"
                raw_rgb.save(raw_out_path)
            except Exception as e:
                messagebox.showwarning(
                    "Save image warning", f"Could not save raw frame image: {e}"
                )

        self._st_popup_img_ref = None
        self._st_popup_label = None

    def _anisotropy_range_s_map(
        self,
        min_x: np.ndarray,
        max_x: np.ndarray,
        min_y: np.ndarray,
        max_y: np.ndarray,
        raw_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        # S: squared sum of the *normalized* anisotropy ranges (no intensity weighting).
        rx = (max_x.astype(np.float32) - min_x.astype(np.float32))
        ry = (max_y.astype(np.float32) - min_y.astype(np.float32))
        s_int = (rx * rx) + (ry * ry)  # (H/2, W/2)

        # Expand to raw pixel grid so spot coordinates remain in full-res space.
        h, w = raw_shape
        s_full = np.repeat(np.repeat(s_int, 2, axis=0), 2, axis=1)
        return (s_full[:h, :w].astype(np.float32, copy=False), s_int.astype(np.float32, copy=False))

    def _phi_var_ok(self, phi_series: list[float]) -> bool:
        if len(phi_series) < 2:
            return False
        arr = np.asarray(phi_series, dtype=np.float32)
        # unwrap so we don't get fooled by [-pi, pi] discontinuities
        arr = np.unwrap(arr)
        return float(np.var(arr)) >= float(self._phi_var_min)

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

    def _apply_phi_var_filter(self, force: bool = False) -> None:
        # Filter detected spots to those with sufficient phi variance.
        with self._analysis_lock:
            if not self._spot_centers or not self._spot_phi_series:
                return
            keep = [i for i, s in enumerate(self._spot_phi_series) if self._phi_var_ok(s)]
            if not keep:
                msg = f"Phi var filter kept 0/{len(self._spot_centers)} (min={self._phi_var_min:.3f} rad^2)."
                self.root.after(0, lambda m=msg: self.bottom_var.set(m))
                if not force:
                    # Avoid surprising 'everything disappeared' during auto-refresh.
                    return
                self._spot_centers = []
                self._spot_phi_series = []
                self._spot_xy_series = []
                self._spot_idx = 0
                return
            if len(keep) == len(self._spot_centers):
                msg = f"Phi var filter kept {len(keep)}/{len(self._spot_centers)} (min={self._phi_var_min:.3f} rad^2)."
                self.root.after(0, lambda m=msg: self.bottom_var.set(m))
                return
            prev_n = len(self._spot_centers)
            self._spot_centers = [self._spot_centers[i] for i in keep]
            self._spot_phi_series = [self._spot_phi_series[i] for i in keep]
            if self._spot_xy_series:
                self._spot_xy_series = [self._spot_xy_series[i] for i in keep]
            if self.last_frame_gray is not None:
                self._update_spot_bounds_intensity(self.last_frame_gray.shape)
            self._spot_idx = 0
            self._sort_spots_by_xy_range()
            msg = f"Phi var filter kept {len(keep)}/{prev_n} (min={self._phi_var_min:.3f} rad^2)."
            self.root.after(0, lambda m=msg: self.bottom_var.set(m))

    def _apply_ring_filter(self, force: bool = False) -> None:
        # Final filter: keep only spots whose XY scatter is annulus-like.
        with self._analysis_lock:
            if not self._spot_centers or not self._spot_xy_series:
                return

            keep = []
            for i, series in enumerate(self._spot_xy_series):
                if self._ring_likeness_score(series) >= float(self._ring_score_min):
                    keep.append(i)

            if not keep:
                msg = f"Ring filter kept 0/{len(self._spot_centers)} (min={self._ring_score_min:.2f})."
                self.root.after(0, lambda m=msg: self.bottom_var.set(m))
                if not force:
                    return
                self._spot_centers = []
                self._spot_phi_series = []
                self._spot_xy_series = []
                self._spot_idx = 0
                return
            if len(keep) == len(self._spot_centers):
                msg = f"Ring filter kept {len(keep)}/{len(self._spot_centers)} (min={self._ring_score_min:.2f})."
                self.root.after(0, lambda m=msg: self.bottom_var.set(m))
                return

            prev_n = len(self._spot_centers)
            self._spot_centers = [self._spot_centers[i] for i in keep]
            self._spot_phi_series = [self._spot_phi_series[i] for i in keep]
            self._spot_xy_series = [self._spot_xy_series[i] for i in keep]
            if self.last_frame_gray is not None:
                self._update_spot_bounds_intensity(self.last_frame_gray.shape)
            self._spot_idx = 0
            self._sort_spots_by_xy_range()
            msg = f"Ring filter kept {len(keep)}/{prev_n} (min={self._ring_score_min:.2f})."
            self.root.after(0, lambda m=msg: self.bottom_var.set(m))

    def _find_truth_json_path(self) -> Optional[Path]:
        outputs_dir = Path(__file__).resolve().parent / "outputs"
        if not outputs_dir.exists():
            return None

        if self.video_path:
            stem = Path(self.video_path).stem
            match = re.search(r"(?:movie|truth)_(\d+)", stem)
            if match:
                candidate = outputs_dir / f"truth_{match.group(1)}.json"
                if candidate.exists():
                    return candidate

        truth_files = list(outputs_dir.glob("truth_*.json"))
        if not truth_files:
            return None
        return max(truth_files, key=lambda p: p.stat().st_mtime)

    def _load_truth_rotators(self, s_map_shape: tuple[int, int]) -> list[dict]:
        path = self._find_truth_json_path()
        if path is None:
            return []
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            return []

        h, w = s_map_shape
        rotators: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("category") != "gold_rotating":
                continue
            center = item.get("center_xy")
            if not center or len(center) < 2:
                continue
            try:
                cx = float(center[0]) * 2.0
                cy = float(center[1]) * 2.0
            except Exception:
                continue
            cx = max(0.0, min(float(w - 1), cx))
            cy = max(0.0, min(float(h - 1), cy))
            rotators.append({"id": item.get("id"), "cx": cx, "cy": cy})
        return rotators

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AVI/NPY Spot Analysis (Q/U + rotators)")

        # Single capture (decoded once)
        self.cap = None
        self.video_path = None
        self.source_kind = None
        self.npy_frames = None
        self.npy_has_frames_dim = False

        # Initial S-map metric (first N frames)
        # legacy buffers (kept for now)
        self._q_buf = []
        self._u_buf = []
        self._st2_frames = max(2, int(detect_spinners.ST2_MEDIAN_FRAMES))
        self._st_popup_done = False
        self._st_popup_img_ref = None  # keep PhotoImage alive
        self._st_popup_label = None
        self._overlay_base_frame = None
        self._s_map = None
        self._s_map_int = None
        self._spot_centers = []
        self._spot_idx = 0
        self._spot_phi_series = []
        self._phi_frames_processed = 0
        self._spot_xy_series = []
        self._xy_frames_processed = 0
        self._spot_bounds_int = []
        self._spot_window_size = 19
        self._spot_scale = 10
        self._spot_percentile = 99.00
        self._spot_min_area = 10
        self._spot_max_area = 1000
        self._spot_connect_radius = 2
        self._phi_var_min = float(self.DEFAULT_PHI_VAR_MIN)
        self._ring_score_min = float(self.DEFAULT_RING_SCORE_MIN)
        self._normalize_s_var = tk.BooleanVar(value=True)
        self._normalize_s_enabled = bool(self._normalize_s_var.get())
        self._spot_img_ref = None
        self._fft_img_ref = None
        self._phi_img_ref = None
        self._xy_img_ref = None
        self._gray_frames = []
        self._gray_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._truth_rotators = []
        self._text_font = self._load_font(size=14)

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

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Select AVI/NPY", command=self.open_video).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="No video loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Spot Analysis")

        nav = ttk.Frame(analysis_tab, padding=(0, 0, 0, 6))
        nav.pack(side=tk.TOP, fill=tk.X)
        self.spot_prev_btn = ttk.Button(nav, text="<", command=self._prev_spot)
        self.spot_prev_btn.pack(side=tk.LEFT)
        self.spot_next_btn = ttk.Button(nav, text=">", command=self._next_spot)
        self.spot_next_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._spot_status_var = tk.StringVar(value="Spot 0 of 0")
        ttk.Label(nav, textvariable=self._spot_status_var).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(nav, text="Percentile").pack(side=tk.LEFT, padx=(12, 0))
        self._spot_pct_var = tk.StringVar(value=f"{self._spot_percentile:.2f}")
        ttk.Entry(nav, textvariable=self._spot_pct_var, width=6).pack(side=tk.LEFT)
        ttk.Label(nav, text="Min area").pack(side=tk.LEFT, padx=(8, 0))
        self._spot_min_var = tk.StringVar(value=str(self._spot_min_area))
        ttk.Entry(nav, textvariable=self._spot_min_var, width=5).pack(side=tk.LEFT)
        ttk.Label(nav, text="Max area").pack(side=tk.LEFT, padx=(8, 0))
        self._spot_max_var = tk.StringVar(value=str(self._spot_max_area))
        ttk.Entry(nav, textvariable=self._spot_max_var, width=5).pack(side=tk.LEFT)

        ttk.Label(nav, text="Phi window").pack(side=tk.LEFT, padx=(8, 0))
        self._spot_win_var = tk.StringVar(value=str(self._spot_window_size))
        ttk.Entry(nav, textvariable=self._spot_win_var, width=5).pack(side=tk.LEFT)

        ttk.Label(nav, text="Min phi var (rad^2)").pack(side=tk.LEFT, padx=(8, 0))
        self._phi_var_min_var = tk.StringVar(value=f"{self._phi_var_min:.3f}")
        ttk.Entry(nav, textvariable=self._phi_var_min_var, width=6).pack(side=tk.LEFT)

        ttk.Label(nav, text="Min ring score").pack(side=tk.LEFT, padx=(8, 0))
        self._ring_score_min_var = tk.StringVar(value=f"{self._ring_score_min:.2f}")
        ttk.Entry(nav, textvariable=self._ring_score_min_var, width=5).pack(side=tk.LEFT)

        ttk.Checkbutton(
            nav,
            text="Normalize S metric",
            variable=self._normalize_s_var,
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(nav, text="px").pack(side=tk.LEFT, padx=(2, 0))

        self._spot_update_btn = ttk.Button(nav, text="Update analysis", command=self._apply_spot_params)
        self._spot_update_btn.pack(side=tk.LEFT, padx=(8, 0))

        spot_body = ttk.Frame(analysis_tab)
        spot_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        spot_body.columnconfigure(0, weight=1)
        spot_body.columnconfigure(1, weight=1)
        spot_body.rowconfigure(0, weight=1)
        spot_body.rowconfigure(1, weight=0)

        self._spot_img_label = ttk.Label(spot_body)
        self._spot_img_label.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self._fft_img_label = ttk.Label(spot_body)
        self._fft_img_label.grid(row=0, column=1, sticky="nsew")

        self._phi_img_label = ttk.Label(spot_body)
        self._phi_img_label.grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))

        self._xy_img_label = ttk.Label(spot_body)
        self._xy_img_label.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        self.bottom_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.bottom_var).pack(side=tk.BOTTOM, anchor="w")

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
        for cx, cy in self._spot_centers:
            ix = int(round(cx / 2.0))
            iy = int(round(cy / 2.0))
            x0 = max(0, ix - half)
            x1 = min(iw, ix + half + 1)
            y0 = max(0, iy - half)
            y1 = min(ih, iy + half + 1)
            bounds.append((x0, x1, y0, y1))
        self._spot_bounds_int = bounds

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
        if self._spot_bounds_int:
            self._spot_bounds_int = [self._spot_bounds_int[i] for i in indices]
        self._spot_idx = 0

    def _append_xy_from_frame(self, gray: np.ndarray) -> None:
        # Compute per-spot X/Y using intensity subframes.
        if (not self._spot_bounds_int) or (len(self._spot_bounds_int) != len(self._spot_centers)):
            self._update_spot_bounds_intensity(gray.shape)

        I90 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I0 = gray[1::2, 1::2]

        eps = 1e-6
        for i, (x0, x1, y0, y1) in enumerate(self._spot_bounds_int):
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
            self._spot_xy_series[i].append((float(x), float(y)))
            # Phi uses normalized Stokes-like quantities:
            # q = (I0 - I90) / (I0 + I90), u = (I45 - I135) / (I45 + I135)
            self._spot_phi_series[i].append(float(0.5 * np.arctan2(y, x)))

    def _append_xy_frame(self, gray: np.ndarray, frame_idx: int) -> None:
        if not self._spot_centers:
            return
        with self._analysis_lock:
            if not self._spot_centers:
                return
            if self._xy_frames_processed >= frame_idx:
                return
            self._append_xy_from_frame(gray)
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
        self._spot_centers = centers_full
        self._spot_idx = 0
        with self._analysis_lock:
            self._spot_phi_series = [[] for _ in centers_full]
            self._phi_frames_processed = 0
            self._spot_xy_series = [[] for _ in centers_full]
            self._xy_frames_processed = 0

        # Seed XY+phi using stored raw frames (if available).
        with self._gray_lock:
            gray_frames = list(self._gray_frames)
        if centers_full and gray_frames:
            upto = len(gray_frames)
            with self._analysis_lock:
                self._spot_xy_series = [[] for _ in centers_full]
                self._xy_frames_processed = 0
                self._spot_phi_series = [[] for _ in centers_full]
                self._phi_frames_processed = 0
                self._update_spot_bounds_intensity(gray_frames[0].shape)
                for t in range(upto):
                    self._append_xy_from_frame(gray_frames[t])
                self._xy_frames_processed = upto
                self._phi_frames_processed = upto

        self._sort_spots_by_xy_range()


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

    def _recompute_xy_series(self, gray_frames=None, centers=None) -> None:
        if gray_frames is None:
            with self._gray_lock:
                gray_frames = list(self._gray_frames)
        with self._analysis_lock:
            if centers is not None:
                self._spot_centers = centers
            self._spot_xy_series = [[] for _ in self._spot_centers]
            self._spot_phi_series = [[] for _ in self._spot_centers]
            if gray_frames:
                self._update_spot_bounds_intensity(gray_frames[0].shape)
            for g in gray_frames:
                self._append_xy_from_frame(g)
            self._xy_frames_processed = len(gray_frames)
            self._phi_frames_processed = len(gray_frames)

    def _apply_spot_params(self) -> None:
        try:
            percentile = float(self._spot_pct_var.get())
            min_area = int(self._spot_min_var.get())
            max_area = int(self._spot_max_var.get())
            win = int(self._spot_win_var.get())
            phi_var_min = float(self._phi_var_min_var.get())
            ring_score_min = float(self._ring_score_min_var.get())
        except ValueError:
            messagebox.showerror(
                "Spot analysis",
                "Percentile/phi var must be numbers and min/max area/window must be integers.",
            )
            return

        if not (0.0 < percentile < 100.0):
            messagebox.showerror("Spot analysis", "Percentile must be between 0 and 100.")
            return
        if min_area < 1:
            messagebox.showerror("Spot analysis", "Min area must be >= 1.")
            return
        if max_area < min_area:
            messagebox.showerror("Spot analysis", "Max area must be >= min area.")
            return
        if win < 3:
            messagebox.showerror("Spot analysis", "Phi window must be >= 3.")
            return
        if win % 2 == 0:
            messagebox.showerror("Spot analysis", "Phi window must be an odd integer (e.g. 19).")
            return
        if phi_var_min < 0.0:
            messagebox.showerror("Spot analysis", "Min phi var must be >= 0.")
            return
        if not (0.0 <= ring_score_min <= 1.0):
            messagebox.showerror("Spot analysis", "Min ring score must be between 0 and 1.")
            return

        self._spot_percentile = percentile
        self._spot_min_area = min_area
        self._spot_max_area = max_area
        self._spot_window_size = win
        self._phi_var_min = phi_var_min
        self._ring_score_min = ring_score_min
        self._spot_pct_var.set(f"{percentile:.2f}")
        self._spot_min_var.set(str(min_area))
        self._spot_max_var.set(str(max_area))
        self._spot_win_var.set(str(win))
        self._phi_var_min_var.set(f"{phi_var_min:.3f}")
        self._ring_score_min_var.set(f"{ring_score_min:.2f}")

        if self._s_map is None or self._s_map_int is None:
            messagebox.showinfo("Spot analysis", "S_map is not ready yet.")
            return

        centers_int = self._find_centers_on_s_int(self._s_map_int)
        centers = [(cx * 2.0, cy * 2.0) for (cx, cy) in centers_int]
        self._spot_idx = 0
        self._recompute_xy_series(centers=centers)
        self._apply_phi_var_filter(force=True)
        self._apply_ring_filter(force=True)
        self._show_st2_popup(self._s_map)
        self._update_spot_view()

    def _update_spot_view(self) -> None:
        if not self._spot_centers or self._s_map is None:
            self._spot_status_var.set("Spot 0 of 0")
            self.spot_prev_btn.configure(state=tk.DISABLED)
            self.spot_next_btn.configure(state=tk.DISABLED)
            self._spot_img_label.configure(image="")
            self._fft_img_label.configure(image="")
            self._phi_img_label.configure(image="")
            self._xy_img_label.configure(image="")
            self._spot_img_ref = None
            self._fft_img_ref = None
            self._phi_img_ref = None
            self._xy_img_ref = None
            return

        n = len(self._spot_centers)
        self.spot_prev_btn.configure(state=tk.NORMAL)
        self.spot_next_btn.configure(state=tk.NORMAL)
        self._spot_idx = max(0, min(self._spot_idx, n - 1))
        cx, cy = self._spot_centers[self._spot_idx]

        window = self._extract_window(self._s_map, cx, cy, self._spot_window_size)
        u8 = detect_spinners.to_u8_preview(window, lo_pct=0.0, hi_pct=100.0)
        size_px = self._spot_window_size * self._spot_scale
        spot_img = Image.fromarray(u8, mode="L").resize(
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
        fft_img = self._make_fft_image(phi_series, self.source_fps)
        fft_img_tk = ImageTk.PhotoImage(fft_img)
        self._fft_img_label.configure(image=fft_img_tk)
        self._fft_img_ref = fft_img_tk

        phi_img = self._make_phi_plot_image(phi_series, self.source_fps)
        phi_img_tk = ImageTk.PhotoImage(phi_img)
        self._phi_img_label.configure(image=phi_img_tk)
        self._phi_img_ref = phi_img_tk

        with self._analysis_lock:
            xy_series = (
                list(self._spot_xy_series[self._spot_idx]) if self._spot_xy_series else []
            )
        ring_score = self._ring_likeness_score(xy_series) if xy_series else 0.0
        xy_img = self._make_xy_scatter_image(xy_series, ring_score=ring_score, ring_thr=self._ring_score_min)
        xy_img_tk = ImageTk.PhotoImage(xy_img)
        self._xy_img_label.configure(image=xy_img_tk)
        self._xy_img_ref = xy_img_tk

        self._spot_status_var.set(f"Spot {self._spot_idx + 1} of {n}")

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
                "Unsupported NPY shape. Expected (H,W), (N,H,W), or (N,H,W,C).",
            )
            return False

        if arr.ndim == 2:
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

        gray0 = _to_gray_u8(frame0)
        if gray0 is None:
            messagebox.showerror("Error", "Could not convert first frame to grayscale.")
            return False

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
        self._normalize_s_enabled = bool(self._normalize_s_var.get())

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
        self._st2_frames = max(2, int(detect_spinners.ST2_MEDIAN_FRAMES))
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None
        self._truth_rotators = []
        with self._gray_lock:
            self._gray_frames = []
        self._s_map = None
        self._s_map_int = None
        self._spot_centers = []
        self._spot_phi_series = []
        self._spot_xy_series = []
        self._spot_bounds_int = []
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

                idx += 1
                self.current_idx = idx
                self.last_frame_gray = gray

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

            if self.npy_has_frames_dim:
                total = int(self.frame_count)
                for i in range(total):
                    if self.stop_event.is_set():
                        break
                    frame = self.npy_frames[i]
                    gray = _to_gray_u8(frame)
                    if gray is None:
                        break
                    idx += 1
                    self.current_idx = idx
                    self.last_frame_gray = gray
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
                        idx = 1
                        self.current_idx = idx
                        self.last_frame_gray = gray
                        while not self.stop_event.is_set():
                            try:
                                self.frame_q.put(gray, timeout=0.1)
                                break
                            except queue.Full:
                                continue
        finally:
            self.decode_done = True

    def _recon_worker(self, shape: tuple[int, int]):
        # For the initial S-map, track per-pixel normalized anisotropy ranges over the first N frames.
        normalize_s = bool(self._normalize_s_enabled)
        xy_recon = make_xy_reconstructor(
            shape, eps=1e-6, out_dtype=np.float32, normalize=normalize_s
        )
        min_x = max_x = None
        min_y = max_y = None
        x_sm = y_sm = None
        smap_frames_seen = 0
        processed = 0

        while not self.stop_event.is_set():
            # exit condition: decoding finished and queue drained
            if self.decode_done and self.frame_q.empty():
                break

            try:
                gray = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._gray_lock:
                self._gray_frames.append(gray)
            # Buffer first N frames for initial S-map:
            # S = range(X)^2 + range(Y)^2 over time (per pixel),
            # where X=(I0-I90)/(I0+I90+eps), Y=(I45-I135)/(I45+I135+eps).
            if (not self._st_popup_done) and (smap_frames_seen < self._st2_frames):
                X, Y = xy_recon(gray)
                # Spatially average X and Y over a 4x4 region before range tracking.
                # This suppresses isolated noisy pixels.
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

                if min_x is None:
                    min_x = x_sm.copy()
                    max_x = x_sm.copy()
                    min_y = y_sm.copy()
                    max_y = y_sm.copy()
                else:
                    np.minimum(min_x, x_sm, out=min_x)
                    np.maximum(max_x, x_sm, out=max_x)
                    np.minimum(min_y, y_sm, out=min_y)
                    np.maximum(max_y, y_sm, out=max_y)

                smap_frames_seen += 1

                if smap_frames_seen == self._st2_frames:
                    try:
                        s_full, s_int = self._anisotropy_range_s_map(
                            min_x, max_x, min_y, max_y, gray.shape
                        )
                        try:
                            detect_spinners.save_s_histogram(
                                s_int,
                                out_path=Path.cwd() / "S_map_hist.png",
                                title="S_map pixel distribution (range(norm anisotropy) metric)",
                            )
                        except Exception as e:
                            self.root.after(
                                0,
                                lambda: messagebox.showwarning(
                                    "Histogram warning", f"Could not save S_map histogram: {e}"
                                ),
                            )
                        centers_int = self._find_centers_on_s_int(s_int)
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Spinner detect error", str(e)))
                    else:
                        centers = [(cx * 2.0, cy * 2.0) for (cx, cy) in centers_int]
                        h, w = gray.shape
                        m = int(self.EDGE_EXCLUDE_PX)
                        centers = [
                            (cx, cy)
                            for (cx, cy) in centers
                            if (m <= cx <= (w - 1 - m)) and (m <= cy <= (h - 1 - m))
                        ]
                        self._init_spot_analysis(s_full, s_int, centers)
                        self._apply_phi_var_filter()
                        self._st_popup_done = True
                        self.root.after(0, lambda arr=s_full: self._show_st2_popup(arr))
                        self.root.after(0, self._update_spot_view)

            frame_idx = processed + 1
            self._append_xy_frame(gray, frame_idx)

            processed += 1
            self.proc_done = processed

        if (not self._st_popup_done) and (smap_frames_seen >= 2) and (min_x is not None):
            # Short clips: build S-map from whatever frames we got.
            try:
                s_full, s_int = self._anisotropy_range_s_map(
                    min_x, max_x, min_y, max_y, self.last_frame_gray.shape
                )
                centers_int = self._find_centers_on_s_int(s_int)
            except Exception:
                centers_int = []
                s_full = self._s_map
                s_int = self._s_map_int
            centers = [(cx * 2.0, cy * 2.0) for (cx, cy) in centers_int]
            h, w = self.last_frame_gray.shape
            m = int(self.EDGE_EXCLUDE_PX)
            centers = [
                (cx, cy)
                for (cx, cy) in centers
                if (m <= cx <= (w - 1 - m)) and (m <= cy <= (h - 1 - m))
            ]
            self._init_spot_analysis(s_full, s_int, centers)
            self._st_popup_done = True
            self.root.after(0, lambda arr=s_full: self._show_st2_popup(arr))
            self.root.after(0, self._update_spot_view)

        if self._spot_centers:
            self._apply_phi_var_filter()
            self._apply_ring_filter()
            # Overwrite S_map_spots.png with filtered "rotators" view at the end.
            if self._s_map is not None:
                self.root.after(0, lambda arr=self._s_map: self._show_st2_popup(arr))
            self.root.after(0, self._update_spot_view)
        self.root.after(0, lambda: self._show_finished(True))

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
        self._show_finished(False)

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
        self._truth_rotators = []
        self._s_map = None
        self._spot_centers = []
        self._spot_phi_series = []
        self._spot_xy_series = []
        self._spot_bounds_int = []
        self._xy_frames_processed = 0
        self._phi_frames_processed = 0
        self._spot_idx = 0
        with self._gray_lock:
            self._gray_frames = []
        self._update_spot_view()

    def on_close(self):
        self._close_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BasicVideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
