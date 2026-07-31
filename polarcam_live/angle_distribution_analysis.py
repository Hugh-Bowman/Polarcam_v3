from __future__ import annotations

import math
import threading
import tkinter as tk
import json
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from PIL import Image
except Exception:
    Image = None

import Detection_alg_offline as detect_spinners
from pol_reconstruction import make_qu_reconstructor

try:
    from scipy.signal import welch as _welch  # type: ignore
except Exception:
    _welch = None

try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:
    _scipy_stats = None

def _to_gray_u8(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.ndim == 3:
        try:
            frame = cv2.extractChannel(frame, 0)
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


def _welch_fallback(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    if x.ndim != 1 or nperseg <= 0 or x.size < nperseg:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    fs = float(fs) if fs and fs > 0.0 else 1.0
    step = max(1, int(nperseg) - max(0, int(noverlap)))
    if step <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    win = np.hanning(nperseg)
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
        seg = (seg - np.mean(seg)) * win
        fft = np.fft.fft(seg, n=nperseg)
        p = (np.abs(fft) ** 2) / scale
        acc = p if acc is None else (acc + p)

    psd = acc / float(nseg)
    freqs = np.fft.fftfreq(nperseg, d=1.0 / fs)
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


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
    return _welch_fallback(x=x, fs=fs, nperseg=nperseg, noverlap=noverlap)


class AngleDistributionApp:
    DEFAULT_RING_SCORE_MIN = 0.0
    ABS_RANGE_MIN = 0.50
    DIR_FILTER_B_MIN = 0.4
    AVG_BIN_DEG = 6.0
    SPHERE_FIT_BINS_Z = 18
    SPHERE_FIT_BINS_PHI = 36
    THETA_MODELS = {
        "hole+fresnel": {
            "a": 0.1865937176,
            "b": 0.5576753053,
            "c": 0.4215514426,
            "r_max": 0.9170101839,
        },
        "nohole": {
            "a": 0.1895779531,
            "b": 0.6256149990,
            "c": 0.4867530374,
            "r_max": 0.9250130600,
        },
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Angle Distribution Analysis")
        self.root.geometry("1300x900")

        self._dog_k_std = float(detect_spinners.DOG_K_STD)
        self._spot_window_size = 19
        self._ring_score_min = float(self.DEFAULT_RING_SCORE_MIN)
        self._abs_range_filter_enabled = True
        self._dir_filter_enabled = False

        self.source_fps = 30.0
        self.source_path: Optional[Path] = None
        self.source_paths: list[Path] = []
        self._spot_fps: list[float] = []
        self.npy_frames: Optional[np.ndarray] = None
        self.npy_has_frames_dim = False
        self.frame_count = 0
        self._source_shape: Optional[tuple[int, int]] = None

        self._s_map: Optional[np.ndarray] = None
        self._s_map_int: Optional[np.ndarray] = None
        self._spot_centers_all: list[tuple[float, float]] = []
        self._spot_xy_series_all: list[list[tuple[float, float]]] = []
        self._spot_phi_series_all: list[list[float]] = []
        self._spot_bounds_int_all: list[tuple[int, int, int, int]] = []

        self._spot_centers: list[tuple[float, float]] = []
        self._spot_xy_series: list[list[tuple[float, float]]] = []
        self._spot_phi_series: list[list[float]] = []
        self._spot_idx = 0

        self._fit_center: Optional[tuple[float, float]] = None
        self._fit_radius: Optional[float] = None
        self._fit_shifted_xy: Optional[np.ndarray] = None
        self._fit_shifted_phi: Optional[np.ndarray] = None
        self._fit_fft_freqs: Optional[np.ndarray] = None
        self._fit_fft_psd: Optional[np.ndarray] = None
        self._fft_peak_records: list[dict] = []
        self._fft_marker_freq_hz: Optional[float] = None
        self._drag_fft_marker = False
        self._analysis_mode = "widefield"
        self._spot_names: list[str] = []
        self._spot_file_summaries: list[str] = []
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1: Optional[float] = None
        self._phi_sel_tmax = 0.0
        self._phi_sel_line0 = None
        self._phi_sel_line1 = None
        self._phi_sel_span = None
        self._phi_sel_tmax_sphere = 0.0
        self._phi_sel_line0_sphere = None
        self._phi_sel_line1_sphere = None
        self._phi_sel_span_sphere = None
        self._drag_phi_handle: Optional[str] = None
        self._drag_phi_source: Optional[str] = None
        self._fps_mode_var = tk.StringVar(value="Auto")
        self._fps_manual_var = tk.StringVar(value="1600")
        self._bin_deg_var = tk.StringVar(value="9")
        self._theta_model_var = tk.StringVar(value="hole+fresnel")
        self._sphere_trail_var = tk.StringVar(value="50")
        self._sphere_speed_var = tk.StringVar(value="1.0")
        self._fit_circle_sphere_var = tk.BooleanVar(value=False)
        self._unwrap_phi_continuity_var = tk.BooleanVar(value=True)
        self._sphere_dual_axis_var = tk.BooleanVar(value=False)
        self._phi_residual_avg_var = tk.BooleanVar(value=False)
        self._phi_residual_window_var = tk.StringVar(value="81")
        self._phi_grid_23_var = tk.BooleanVar(value=False)
        self._speed_avg_window_var = tk.StringVar(value="81")
        self._fft_max_hz_var = tk.StringVar(value="100")
        self._fft_peak_info_var = tk.StringVar(value="No Fourier peaks added yet.")

        self._avg_selected_spots: list[bool] = []
        self._include_current_var = tk.BooleanVar(value=False)
        self._include_speed_theta_var = tk.BooleanVar(value=False)
        self._syncing_include_var = False
        self._syncing_speed_theta_var = False
        self._speed_theta_points: list[dict] = []
        self._speed_theta_selected_spots: list[bool] = []
        self._speed_theta_info_var = tk.StringVar(
            value="No points yet. Tick rods to include; metric is P95(phi)-P5(phi) post circle-fit."
        )
        self._brownian_phi_source_var = tk.StringVar(value="Shifted+fitted phi")
        self._brownian_result_var = tk.StringVar(
            value="Press test button for log-log MSD straight-line Brownian check."
        )
        self._brownian_lag_min_var = tk.StringVar(value="")
        self._brownian_lag_max_var = tk.StringVar(value="")
        self._brownian_show_refs_var = tk.BooleanVar(value=True)
        self._brownian_last: Optional[dict] = None
        self._dwell_avg_points_var = tk.StringVar(value="51")
        self._dwell_jump_deg_var = tk.StringVar(value="25")
        self._dwell_min_points_var = tk.StringVar(value="10")
        self._dwell_bin_width_s_var = tk.StringVar(value="0.05")
        self._dwell_info_var = tk.StringVar(value="Analyze current spot first, then compute plateau dwell times.")
        self._raw_play_speed_var = tk.StringVar(value="100")
        self._raw_play_trail_var = tk.StringVar(value="200")
        self._raw_play_info_var = tk.StringVar(value="Raw XY playback and gallery export.")
        self._raw_play_running = False
        self._raw_play_after: Optional[str] = None
        self._raw_play_idx = 0
        self._raw_xy_path_artist = None
        self._raw_xy_trail_artist = None
        self._raw_xy_head_artist = None
        self._raw_2phi_marker = None
        self._raw_r_marker = None
        self._raw_theta_marker = None

        self._sphere_fit: Optional[dict] = None
        self._sphere_anim_running = False
        self._sphere_anim_after: Optional[str] = None
        self._sphere_anim_idx = 0
        self._sphere_fit_i0 = 0
        self._sphere_fit_i1 = 0
        self._sphere_trail_artist = None
        self._sphere_head_artist = None
        self._sphere_plane_trail_artist = None
        self._sphere_plane_head_artist = None
        self._ax_sphere_heatmap = None
        self._ax_sphere_theta = None
        self._sphere_info_var = tk.StringVar(value="Press analysis button to reconstruct unit-sphere trajectory.")
        self._arc_result: Optional[dict] = None
        self._arc_info_var = tk.StringVar(value="Press Analyse Arc to fit the selected unit-sphere window.")
        self._arc_analyse_btn = None
        self._ax_arc_3d = None
        self._ax_arc_angle = None
        self._ax_arc_hist = None
        self._arc_fig = None
        self._arc_canvas = None
        self._heatmap_info_var = tk.StringVar(value="Spherical-coordinate density for the current selected window.")
        self._ax_spherical_heatmap = None
        self._spherical_heatmap_fig = None
        self._spherical_heatmap_canvas = None
        self._unwrap_phi_chk = None
        self._fit_circle_sphere_chk = None
        self._include_chk = None
        self._include_speed_theta_chk = None
        self._avg_refresh_btn = None
        self._include_upto_btn = None
        self._st_auto_half_btn = None
        self._st_add_half_btn = None
        self._st_add_full_btn = None
        self._st_clear_btn = None
        self._bm_source_box = None
        self._bm_test_btn = None
        self._bm_lag_min_entry = None
        self._bm_lag_max_entry = None
        self._bm_apply_lag_btn = None
        self._bm_show_refs_chk = None
        self._fft_peak_clear_btn = None
        self._avg_ax = None
        self._avg_fig = None
        self._avg_canvas = None
        self._st_ax = None
        self._st_fig = None
        self._st_canvas = None
        self._dwell_ax_phi = None
        self._dwell_ax_hist = None
        self._dwell_fig = None
        self._dwell_canvas = None
        self._raw_ax_xy = None
        self._raw_ax_2phi = None
        self._raw_ax_r = None
        self._raw_ax_theta = None
        self._raw_fig = None
        self._raw_canvas = None
        self._bm_ax_msd = None
        self._bm_ax_inc = None
        self._bm_fig = None
        self._bm_canvas = None
        self._fft_peak_ax_hist = None
        self._fft_peak_ax_scatter = None
        self._fft_peak_fig = None
        self._fft_peak_canvas = None

        self._busy = False

        self._build_ui()
        self._render_all()

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        try:
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        try:
            return int(text)
        except Exception:
            return None

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        top2 = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        top2.pack(side=tk.TOP, fill=tk.X)

        self._load_btn = ttk.Button(top, text="Load .npy", command=self._on_load_npy)
        self._load_btn.pack(side=tk.LEFT)
        self._load_many_btn = ttk.Button(top, text="Load many inspection .npy", command=self._on_load_many_inspection_npy)
        self._load_many_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._run_btn = ttk.Button(top, text="Re-run analysis", command=self._rerun_analysis)
        self._run_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(top, text="DoG k").pack(side=tk.LEFT, padx=(16, 4))
        self._dog_k_var = tk.StringVar(value=f"{self._dog_k_std:.2f}")
        ttk.Entry(top, textvariable=self._dog_k_var, width=7).pack(side=tk.LEFT)

        ttk.Label(top, text="Phi window").pack(side=tk.LEFT, padx=(12, 4))
        self._spot_win_var = tk.StringVar(value=str(self._spot_window_size))
        ttk.Entry(top, textvariable=self._spot_win_var, width=6).pack(side=tk.LEFT)

        ttk.Label(top, text="Min hollowness").pack(side=tk.LEFT, padx=(12, 4))
        self._ring_score_min_var = tk.StringVar(value=f"{self._ring_score_min:.2f}")
        ttk.Entry(top, textvariable=self._ring_score_min_var, width=6).pack(side=tk.LEFT)

        ttk.Label(top, text="FPS").pack(side=tk.LEFT, padx=(12, 4))
        self._fps_mode_box = ttk.Combobox(
            top,
            textvariable=self._fps_mode_var,
            values=["Auto", "1600", "77", "Manual"],
            width=7,
            state="readonly",
        )
        self._fps_mode_box.pack(side=tk.LEFT)
        self._fps_manual_entry = ttk.Entry(top, textvariable=self._fps_manual_var, width=7)
        self._fps_manual_entry.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(top, text="Bin (deg)").pack(side=tk.LEFT, padx=(12, 4))
        self._bin_deg_entry = ttk.Entry(top, textvariable=self._bin_deg_var, width=6)
        self._bin_deg_entry.pack(side=tk.LEFT)

        ttk.Label(top, text="Theta model").pack(side=tk.LEFT, padx=(12, 4))
        self._theta_model_box = ttk.Combobox(
            top,
            textvariable=self._theta_model_var,
            values=["hole+fresnel", "nohole"],
            width=14,
            state="readonly",
        )
        self._theta_model_box.pack(side=tk.LEFT)
        self._theta_model_box.bind("<<ComboboxSelected>>", self._on_theta_model_changed)

        ttk.Checkbutton(
            top2,
            text="Subtract time average",
            variable=self._phi_residual_avg_var,
            command=self._on_phi_residual_settings_changed,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top2, text="Window pts").pack(side=tk.LEFT, padx=(8, 4))
        self._phi_residual_window_entry = ttk.Entry(top2, textvariable=self._phi_residual_window_var, width=7)
        self._phi_residual_window_entry.pack(side=tk.LEFT)
        self._phi_residual_window_var.trace_add("write", lambda *_: self._on_phi_residual_settings_changed())
        ttk.Label(top2, text="Speed avg pts").pack(side=tk.LEFT, padx=(12, 4))
        self._speed_avg_window_entry = ttk.Entry(top2, textvariable=self._speed_avg_window_var, width=7)
        self._speed_avg_window_entry.pack(side=tk.LEFT)
        self._speed_avg_window_var.trace_add("write", lambda *_: self._on_speed_avg_changed())
        ttk.Checkbutton(
            top2,
            text="Grid 360/23",
            variable=self._phi_grid_23_var,
            command=self._on_phi_grid_changed,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top2, text="FFT max Hz").pack(side=tk.LEFT, padx=(12, 4))
        self._fft_max_hz_entry = ttk.Entry(top2, textvariable=self._fft_max_hz_var, width=7)
        self._fft_max_hz_entry.pack(side=tk.LEFT)
        self._fft_max_hz_var.trace_add("write", lambda *_: self._on_fft_bounds_changed())

        self._abs_var = tk.BooleanVar(value=self._abs_range_filter_enabled)
        ttk.Checkbutton(
            top2,
            text=f"Filter max(range(X),range(Y)) > {self.ABS_RANGE_MIN:.2f}",
            variable=self._abs_var,
            command=self._rerun_analysis,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self._dir_var = tk.BooleanVar(value=self._dir_filter_enabled)
        ttk.Checkbutton(
            top2,
            text=f"Filter unidirectional (|B| > {self.DIR_FILTER_B_MIN:.2f})",
            variable=self._dir_var,
            command=self._rerun_analysis,
        ).pack(side=tk.LEFT, padx=(12, 0))

        nav = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        nav.pack(side=tk.TOP, fill=tk.X)
        self._prev_btn = ttk.Button(nav, text="<", width=3, command=self._prev_spot)
        self._prev_btn.pack(side=tk.LEFT)
        self._next_btn = ttk.Button(nav, text=">", width=3, command=self._next_spot)
        self._next_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._spot_status_var = tk.StringVar(value="Spot 0 / 0")
        ttk.Label(nav, textvariable=self._spot_status_var).pack(side=tk.LEFT, padx=(8, 0))
        self._current_file_var = tk.StringVar(value="File: -")
        ttk.Label(nav, textvariable=self._current_file_var).pack(side=tk.LEFT, padx=(12, 0))
        self._unwrap_phi_chk = ttk.Checkbutton(
            nav,
            text="Unwrap phi by continuity",
            variable=self._unwrap_phi_continuity_var,
            command=self._on_toggle_phi_unwrap,
        )
        self._unwrap_phi_chk.pack(side=tk.LEFT, padx=(12, 0))
        self._fit_circle_sphere_chk = ttk.Checkbutton(
            nav,
            text="Use fitted phi",
            variable=self._fit_circle_sphere_var,
            command=self._on_toggle_fit_circle_sphere,
        )
        self._fit_circle_sphere_chk.pack(side=tk.LEFT, padx=(12, 0))
        self._analysis_btn = ttk.Button(
            nav,
            text="Apply fitted phi",
            command=self._analyze_current_spot_distribution,
        )
        self._analysis_btn.pack(side=tk.RIGHT)
        self._fft_add_peaks_btn = None

        self._status_var = tk.StringVar(value="Load a .npy stack to begin.")
        ttk.Label(self.root, textvariable=self._status_var, padding=(8, 0, 8, 8)).pack(side=tk.TOP, fill=tk.X)

        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._notebook = notebook
        tab_spot = ttk.Frame(notebook)
        tab_sphere = ttk.Frame(notebook)
        tab_arc = ttk.Frame(notebook)
        tab_spherical_heatmap = ttk.Frame(notebook)
        notebook.add(tab_spot, text="Spot View")
        notebook.add(tab_sphere, text="Unit Sphere")
        notebook.add(tab_arc, text="Arc Analysis")
        notebook.add(tab_spherical_heatmap, text="Spherical Heatmap")

        fig = Figure(figsize=(13, 10), dpi=100)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0])
        self._ax_xy = fig.add_subplot(gs[0, 0])
        self._ax_shift = fig.add_subplot(gs[0, 1])
        self._ax_phi = fig.add_subplot(gs[1, :])
        self._ax_hist = fig.add_subplot(gs[2, :])
        self._ax_fft = fig.add_subplot(gs[3, :])
        self._ax_speed = None
        self._ax_phi_dist_fft = None
        self._fig = fig

        canvas = FigureCanvasTkAgg(fig, master=tab_spot)
        self._canvas = canvas
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._mpl_cid_press = canvas.mpl_connect("button_press_event", self._on_phi_press)
        self._mpl_cid_motion = canvas.mpl_connect("motion_notify_event", self._on_phi_motion)
        self._mpl_cid_release = canvas.mpl_connect("button_release_event", self._on_phi_release)
        self._mpl_cid_fft_press = canvas.mpl_connect("button_press_event", self._on_fft_press)
        self._mpl_cid_fft_motion = canvas.mpl_connect("motion_notify_event", self._on_fft_motion)
        self._mpl_cid_fft_release = canvas.mpl_connect("button_release_event", self._on_fft_release)

        sphere_top = ttk.Frame(tab_sphere, padding=(8, 8, 8, 0))
        sphere_top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(sphere_top, text="Trail points").pack(side=tk.LEFT)
        self._sphere_trail_entry = ttk.Entry(sphere_top, textvariable=self._sphere_trail_var, width=7)
        self._sphere_trail_entry.pack(side=tk.LEFT, padx=(4, 0))
        self._sphere_trail_apply_btn = ttk.Button(sphere_top, text="Apply", command=self._on_apply_sphere_trail)
        self._sphere_trail_apply_btn.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(sphere_top, text="Speed x").pack(side=tk.LEFT)
        self._sphere_speed_entry = ttk.Entry(sphere_top, textvariable=self._sphere_speed_var, width=6)
        self._sphere_speed_entry.pack(side=tk.LEFT, padx=(4, 8))
        self._sphere_dual_axis_chk = ttk.Checkbutton(
            sphere_top,
            text="Dual-axis fit",
            variable=self._sphere_dual_axis_var,
            command=self._on_toggle_dual_axis_fit,
        )
        self._sphere_dual_axis_chk.pack(side=tk.LEFT, padx=(0, 8))
        self._sphere_play_btn = ttk.Button(sphere_top, text="Play", command=self._on_sphere_play)
        self._sphere_play_btn.pack(side=tk.LEFT)
        self._sphere_pause_btn = ttk.Button(sphere_top, text="Pause", command=self._on_sphere_pause)
        self._sphere_pause_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._sphere_reset_btn = ttk.Button(sphere_top, text="Reset", command=self._on_sphere_reset)
        self._sphere_reset_btn.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(sphere_top, textvariable=self._sphere_info_var).pack(side=tk.LEFT, padx=(12, 0))

        sphere_fig = Figure(figsize=(13, 10), dpi=100)
        sphere_gs = sphere_fig.add_gridspec(4, 2, height_ratios=(1.1, 1.0, 0.8, 0.8))
        self._ax_sphere_3d = sphere_fig.add_subplot(sphere_gs[0, 0], projection="3d")
        self._ax_sphere_heatmap = sphere_fig.add_subplot(sphere_gs[0, 1], projection="3d")
        self._ax_sphere_plane = sphere_fig.add_subplot(sphere_gs[1, 0])
        self._ax_sphere_phi = sphere_fig.add_subplot(sphere_gs[1, 1])
        self._ax_sphere_theta = sphere_fig.add_subplot(sphere_gs[2, :])
        self._ax_sphere_hist = sphere_fig.add_subplot(sphere_gs[3, :])
        self._sphere_fig = sphere_fig
        sphere_canvas = FigureCanvasTkAgg(sphere_fig, master=tab_sphere)
        self._sphere_canvas = sphere_canvas
        sphere_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        self._sphere_mpl_cid_press = sphere_canvas.mpl_connect("button_press_event", self._on_sphere_phi_press)
        self._sphere_mpl_cid_motion = sphere_canvas.mpl_connect("motion_notify_event", self._on_sphere_phi_motion)
        self._sphere_mpl_cid_release = sphere_canvas.mpl_connect("button_release_event", self._on_sphere_phi_release)

        arc_top = ttk.Frame(tab_arc, padding=(8, 8, 8, 0))
        arc_top.pack(side=tk.TOP, fill=tk.X)
        self._arc_analyse_btn = ttk.Button(arc_top, text="Analyse Arc", command=self._on_analyse_arc)
        self._arc_analyse_btn.pack(side=tk.LEFT)
        ttk.Label(arc_top, textvariable=self._arc_info_var).pack(side=tk.LEFT, padx=(12, 0))

        arc_fig = Figure(figsize=(13, 8), dpi=100)
        arc_gs = arc_fig.add_gridspec(2, 2, height_ratios=(1.2, 0.9))
        self._ax_arc_3d = arc_fig.add_subplot(arc_gs[0, 0], projection="3d")
        self._ax_arc_angle = arc_fig.add_subplot(arc_gs[0, 1])
        self._ax_arc_hist = arc_fig.add_subplot(arc_gs[1, :])
        self._arc_fig = arc_fig
        arc_canvas = FigureCanvasTkAgg(arc_fig, master=tab_arc)
        self._arc_canvas = arc_canvas
        arc_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        self._render_arc_tab()

        heatmap_top = ttk.Frame(tab_spherical_heatmap, padding=(8, 8, 8, 0))
        heatmap_top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(heatmap_top, textvariable=self._heatmap_info_var).pack(side=tk.LEFT)
        heatmap_fig = Figure(figsize=(10, 7), dpi=100)
        self._ax_spherical_heatmap = heatmap_fig.add_subplot(111)
        self._spherical_heatmap_fig = heatmap_fig
        heatmap_canvas = FigureCanvasTkAgg(heatmap_fig, master=tab_spherical_heatmap)
        self._spherical_heatmap_canvas = heatmap_canvas
        heatmap_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        self._render_spherical_heatmap_tab()

    def _ui_call(self, fn, *args, **kwargs) -> None:
        self.root.after(0, lambda: fn(*args, **kwargs))

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        state = tk.DISABLED if busy else tk.NORMAL
        for w in (
            self._load_btn,
            self._load_many_btn,
            self._run_btn,
            self._prev_btn,
            self._next_btn,
            self._analysis_btn,
            self._fft_add_peaks_btn,
            self._unwrap_phi_chk,
            self._fit_circle_sphere_chk,
            self._include_chk,
            self._include_speed_theta_chk,
            self._avg_refresh_btn,
            self._include_upto_btn,
            self._fps_mode_box,
            self._fps_manual_entry,
            self._bin_deg_entry,
            self._theta_model_box,
            self._sphere_trail_entry,
            self._sphere_speed_entry,
            self._sphere_dual_axis_chk,
            self._sphere_trail_apply_btn,
            self._sphere_play_btn,
            self._sphere_pause_btn,
            self._sphere_reset_btn,
            self._arc_analyse_btn,
            self._st_auto_half_btn,
            self._st_add_half_btn,
            self._st_add_full_btn,
            self._st_clear_btn,
            self._bm_source_box,
            self._bm_test_btn,
            self._bm_lag_min_entry,
            self._bm_lag_max_entry,
            self._bm_apply_lag_btn,
            self._bm_show_refs_chk,
            self._fft_peak_clear_btn,
        ):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _on_load_npy(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Open NumPy stack",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if not path:
            return
        self._start_processing(Path(path))

    def _on_load_many_inspection_npy(self) -> None:
        if self._busy:
            return
        paths = filedialog.askopenfilenames(
            title="Open many spot-inspection NumPy stacks",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if not paths:
            return
        self._start_processing_many([Path(p) for p in paths])

    def _rerun_analysis(self) -> None:
        if self._busy:
            return
        if self.source_paths and len(self.source_paths) > 1:
            self._start_processing_many(list(self.source_paths))
            return
        if self.source_path is None:
            return
        self._start_processing(self.source_path)

    def _on_theta_model_changed(self, event=None) -> None:
        if self._busy:
            return
        self._sphere_fit = None
        self._clear_arc_result(render=True)
        self._stop_sphere_animation()
        self._render_sphere_tab()
        self._render_spherical_heatmap_tab()

    def _on_phi_residual_settings_changed(self) -> None:
        if self._busy:
            return
        self._render_all()

    def _on_phi_grid_changed(self) -> None:
        if self._busy:
            return
        self._render_all()

    def _on_speed_avg_changed(self) -> None:
        if self._busy:
            return
        self._render_all()

    def _get_speed_avg_points(self, default: int = 81) -> int:
        try:
            n = int(round(float(self._speed_avg_window_var.get())))
        except Exception:
            n = int(default)
        n = max(1, int(n))
        if (n % 2) == 0:
            n += 1
        return n

    def _phi_guide_step_deg(self) -> float:
        return (360.0 / 23.0) if bool(self._phi_grid_23_var.get()) else 36.0

    def _phi_guide_count(self) -> int:
        return 23 if bool(self._phi_grid_23_var.get()) else 10

    def _phi_guide_linestyle(self) -> str:
        return "--" if bool(self._phi_grid_23_var.get()) else ":"

    def _on_fft_bounds_changed(self) -> None:
        if self._busy:
            return
        self._render_all()

    def _on_toggle_phi_unwrap(self) -> None:
        if self._busy:
            return
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._sphere_fit = None
        self._sphere_anim_idx = 0
        self._clear_arc_result(render=True)
        self._stop_sphere_animation()
        self._render_all()
        self._render_spherical_heatmap_tab()

    def _current_spot_label(self) -> str:
        if self._spot_names and 0 <= int(self._spot_idx) < len(self._spot_names):
            return str(self._spot_names[int(self._spot_idx)])
        if self.source_path is not None:
            return str(self.source_path.name)
        return f"spot_{int(self._spot_idx) + 1}"

    def _current_fft_peak_key(self) -> Optional[str]:
        if not self._spot_xy_series:
            return None
        idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
        arr = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        i0, i1 = self._selected_index_range(arr.shape[0], float(self._current_fps()))
        return f"{self._current_spot_label()}|spot={idx}|range={i0}:{i1}"

    def _current_fft_peak_record(self) -> Optional[dict]:
        key = self._current_fft_peak_key()
        if key is None:
            return None
        for rec in self._fft_peak_records:
            if str(rec.get("key", "")) == key:
                return rec
        return None

    def _current_fft_positive_freqs_psd(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_fft_freqs is None or self._fit_fft_psd is None:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        freqs = np.asarray(self._fit_fft_freqs, dtype=np.float64)
        psd = np.asarray(self._fit_fft_psd, dtype=np.float64)
        valid = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= 0.0) & (psd > 0.0)
        return freqs[valid], psd[valid]

    def _compute_complex_xy_fft(self, xy: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 8:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        z = arr[:, 0].astype(np.complex128) + 1j * arr[:, 1].astype(np.complex128)
        z = z - np.mean(z)
        n = int(z.size)
        if n < 8:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        win = np.hanning(n)
        scale = float(fs) * float(np.sum(win * win))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        nfft_target = max(4096, int(n) * 16)
        nfft = 1 << int(math.ceil(math.log2(max(8, nfft_target))))
        fft = np.fft.fft(z * win, n=nfft)
        freqs = np.fft.fftfreq(nfft, d=1.0 / float(fs))
        psd = (np.abs(fft) ** 2) / scale
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)
        if freqs.size == 0 or psd.size == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        order = np.argsort(freqs)
        return freqs[order], psd[order]

    def _fft_max_hz(self, default: float) -> float:
        val = self._parse_float(self._fft_max_hz_var.get())
        if val is None or not np.isfinite(val) or val <= 0.0:
            return float(default)
        return float(val)

    def _nearest_fft_psd(self, freq_hz: float) -> Optional[float]:
        freqs, psd = self._current_fft_positive_freqs_psd()
        if freqs.size == 0:
            return None
        idx = int(np.argmin(np.abs(freqs - float(freq_hz))))
        return float(psd[idx]) if 0 <= idx < psd.size else None

    def _clear_fft_peak_records(self) -> None:
        self._fft_peak_records = []
        self._fft_peak_info_var.set("No Fourier peaks added yet.")
        self._render_all()
        if self._fft_peak_canvas is not None:
            self._render_frequency_peaks_tab()

    def _add_fft_marker_to_plot(self) -> None:
        if self._fit_fft_freqs is None or self._fit_fft_psd is None or self._fit_fft_freqs.size == 0:
            messagebox.showerror("Fourier marker", "Run Analyze Current Spot Angle Distribution first.")
            return
        if self._fft_marker_freq_hz is None or not np.isfinite(float(self._fft_marker_freq_hz)):
            messagebox.showwarning("Fourier marker", "Place the marker on the Fourier plot first.")
            return
        key = self._current_fft_peak_key()
        if key is None:
            return
        marker_freq = float(self._fft_marker_freq_hz)
        marker_psd = self._nearest_fft_psd(marker_freq)
        record = {
            "key": key,
            "label": self._current_spot_label(),
            "spot_idx": int(self._spot_idx),
            "peak_freq_hz": float(marker_freq),
            "peak_psd": None if marker_psd is None else float(marker_psd),
        }
        replaced = False
        for i, rec in enumerate(self._fft_peak_records):
            if str(rec.get("key", "")) == key:
                self._fft_peak_records[i] = record
                replaced = True
                break
        if not replaced:
            self._fft_peak_records.append(record)
        self._render_all()
        if self._fft_peak_canvas is not None:
            self._render_frequency_peaks_tab()
        action = "Updated" if replaced else "Added"
        self._status_var.set(f"{action} Fourier marker at {marker_freq:.3f} Hz for {self._current_spot_label()}.")

    def _on_brownian_source_changed(self, event=None) -> None:
        self._brownian_last = None
        self._brownian_result_var.set("Source changed. Press test button for straight-line MSD check.")
        if self._bm_canvas is not None:
            self._render_brownian_tab()

    def _on_apply_sphere_trail(self) -> None:
        _ = self._get_sphere_trail_len(50)
        self._update_sphere_animation_artists()

    def _on_toggle_fit_circle_sphere(self) -> None:
        if self._busy:
            return
        self._clear_arc_result(render=True)
        if bool(self._fit_circle_sphere_var.get()):
            self._analyze_current_spot_distribution()
            return
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._sphere_fit = None
        self._sphere_anim_idx = 0
        self._stop_sphere_animation()
        self._render_all()

    def _on_toggle_dual_axis_fit(self) -> None:
        if self._busy:
            return
        self._clear_arc_result(render=True)
        if not bool(self._fit_circle_sphere_var.get()):
            self._sphere_fit = None
            self._stop_sphere_animation()
            self._render_sphere_tab()
            self._render_spherical_heatmap_tab()
            return
        self._refresh_sphere_fit_from_current_selection()
        self._render_spherical_heatmap_tab()

    def _refresh_sphere_fit_from_current_selection(self) -> None:
        if not bool(self._fit_circle_sphere_var.get()):
            self._sphere_fit = None
            self._stop_sphere_animation()
            self._render_sphere_tab()
            return
        if not self._spot_xy_series:
            self._sphere_fit = None
            self._stop_sphere_animation()
            self._render_sphere_tab()
            return
        idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        if arr_full.ndim != 2 or arr_full.shape[1] != 2 or arr_full.shape[0] < 3:
            self._sphere_fit = None
            self._stop_sphere_animation()
            self._render_sphere_tab()
            return
        i0, i1 = self._selected_index_range(arr_full.shape[0], float(self._current_fps()))
        arr = arr_full[i0:i1]
        self._sphere_fit = self._fit_unit_sphere_distribution(arr) if arr.shape[0] >= 3 else None
        self._sphere_fit_i0 = int(i0)
        self._sphere_fit_i1 = int(i1)
        self._sphere_anim_idx = 0
        self._stop_sphere_animation()
        self._render_sphere_tab()

    def _on_sphere_play(self) -> None:
        if self._sphere_fit is None:
            return
        n = int(np.asarray(self._sphere_fit.get("u", np.zeros((0, 3), dtype=np.float64))).shape[0])
        if n <= 0:
            return
        if self._sphere_anim_idx >= (n - 1):
            self._sphere_anim_idx = 0
            self._update_sphere_animation_artists()
        self._sphere_anim_running = True
        self._schedule_sphere_animation_tick()

    def _on_sphere_pause(self) -> None:
        self._stop_sphere_animation()

    def _on_sphere_reset(self) -> None:
        self._stop_sphere_animation()
        self._sphere_anim_idx = 0
        self._update_sphere_animation_artists()

    def _stop_sphere_animation(self) -> None:
        self._sphere_anim_running = False
        if self._sphere_anim_after is not None:
            try:
                self.root.after_cancel(self._sphere_anim_after)
            except Exception:
                pass
            self._sphere_anim_after = None

    def _schedule_sphere_animation_tick(self) -> None:
        if not self._sphere_anim_running:
            return
        if self._sphere_anim_after is not None:
            return
        # Speed is points/second. Do not batch points; each tick advances exactly
        # one point so the trail grows smoothly and predictably.
        speed = max(0.01, float(self._get_sphere_speed(1.0)))
        dt_ms = max(1, int(round(1000.0 / speed)))
        self._sphere_anim_after = self.root.after(dt_ms, self._sphere_animation_tick)

    def _sphere_animation_tick(self) -> None:
        self._sphere_anim_after = None
        if not self._sphere_anim_running or self._sphere_fit is None:
            return
        u = np.asarray(self._sphere_fit.get("u", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        n = int(u.shape[0])
        if n <= 0:
            self._sphere_anim_running = False
            return
        if self._sphere_anim_idx < (n - 1):
            self._sphere_anim_idx += 1
            self._update_sphere_animation_artists()
            self._schedule_sphere_animation_tick()
        else:
            self._sphere_anim_running = False

    def _start_processing(self, path: Path) -> None:
        self._set_busy(True)
        self._status_var.set(f"Processing {path.name} ...")
        t = threading.Thread(target=self._process_npy_worker, args=(path,), daemon=True)
        t.start()

    def _start_processing_many(self, paths: list[Path]) -> None:
        if not paths:
            return
        self._set_busy(True)
        self._status_var.set(f"Processing {len(paths)} inspection file(s) ...")
        t = threading.Thread(target=self._process_many_inspection_worker, args=(paths,), daemon=True)
        t.start()

    def _load_npy(self, path: Path) -> tuple[np.ndarray, bool, int, np.ndarray]:
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        if arr.dtype == object:
            raise RuntimeError("Unsupported NPY: object arrays are not supported.")
        if arr.ndim < 2 or arr.ndim > 4:
            raise RuntimeError("Unsupported NPY shape. Expected (H,W), (H,W,C), (N,H,W), or (N,H,W,C).")

        if arr.ndim == 2:
            frame0 = arr
            frame_count = 1
            has_frames_dim = False
        elif arr.ndim == 3:
            if int(arr.shape[-1]) in (1, 3, 4) and int(arr.shape[0]) > 4 and int(arr.shape[1]) > 4:
                frame0 = arr
                frame_count = 1
                has_frames_dim = False
            else:
                if int(arr.shape[0]) < 1:
                    raise RuntimeError("NPY file has no frames.")
                frame0 = arr[0]
                frame_count = int(arr.shape[0])
                has_frames_dim = True
        else:
            if int(arr.shape[0]) < 1:
                raise RuntimeError("NPY file has no frames.")
            frame0 = arr[0]
            frame_count = int(arr.shape[0])
            has_frames_dim = True

        gray0 = _to_gray_u8(frame0)
        if gray0 is None:
            raise RuntimeError("Could not convert first frame to grayscale.")
        if (gray0.shape[0] % 2) != 0 or (gray0.shape[1] % 2) != 0:
            raise RuntimeError(f"Frame shape must be even (polar mosaic). Got {gray0.shape}.")

        return arr, has_frames_dim, frame_count, gray0

    def _iter_gray_frames(self, arr: np.ndarray, has_frames_dim: bool, frame_count: int):
        if has_frames_dim:
            for i in range(frame_count):
                gray = _to_gray_u8(arr[i])
                if gray is not None:
                    yield gray
        else:
            gray = _to_gray_u8(arr)
            if gray is not None:
                yield gray

    def _inspection_crop_side(self, shape: tuple[int, int]) -> Optional[int]:
        h, w = int(shape[0]), int(shape[1])
        small = min(h, w)
        large = max(h, w)
        if large == 256 and small < 50:
            return int(small)
        return None

    def _read_json_if_exists(self, path: Path) -> Optional[object]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _sidecar_json_objects(self, path: Path) -> list[object]:
        candidates = [
            path.with_suffix(".json"),
            path.parent / f"{path.stem}.json",
            path.parent / f"{path.stem}_meta.json",
            path.parent / "meta.json",
        ]
        stem = path.stem
        if stem.startswith("spotrec_"):
            ts = stem[len("spotrec_") :]
            candidates.append(path.parent / f"spotrec_preview_{ts}.json")
        if stem.startswith("capture_"):
            candidates.append(path.parent / f"{stem}_meta.json")

        objs: list[object] = []
        seen: set[str] = set()
        for c in candidates:
            key = str(c.resolve()) if c.exists() else str(c)
            if key in seen:
                continue
            seen.add(key)
            obj = self._read_json_if_exists(c)
            if obj is not None:
                objs.append(obj)
        return objs

    def _walk_json_values(self, obj: object, path: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                child_path = f"{path}.{key}" if path else key
                yield child_path, v
                yield from self._walk_json_values(v, child_path)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                child_path = f"{path}.{i}" if path else str(i)
                yield child_path, v
                yield from self._walk_json_values(v, child_path)

    def _metadata_background_summary(self, objs: list[object]) -> str:
        keys = (
            "background_subtracted",
            "background_subtract_requested",
            "subtract_background",
            "background_profile_subtracted",
        )
        for obj in objs:
            for key_path, val in self._walk_json_values(obj):
                lname = key_path.lower()
                if not any(k in lname for k in keys):
                    continue
                if isinstance(val, bool):
                    return "on" if val else "off"
                if isinstance(val, (int, float)):
                    return "on" if bool(val) else "off"
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in ("true", "yes", "on", "1"):
                        return "on"
                    if v in ("false", "no", "off", "0", "none"):
                        return "off"
        return "unknown"

    def _metadata_exposure_ms(self, objs: list[object]) -> Optional[float]:
        preferred: list[tuple[int, float]] = []
        for obj in objs:
            for key_path, val in self._walk_json_values(obj):
                try:
                    f = float(val)
                except Exception:
                    continue
                if not np.isfinite(f) or f <= 0.0:
                    continue
                lname = key_path.lower()
                score = None
                out = f
                if lname.endswith("timing_snapshot.exposure_us") or lname.endswith("exposure_us"):
                    score = 0
                    out = f / 1000.0
                elif lname.endswith("actual.exposure_ms") or lname.endswith("exposure_ms"):
                    score = 1
                elif lname.endswith("requested.exp_ms") or lname.endswith("exp_ms"):
                    score = 2
                elif lname.endswith("exposure_time_ms"):
                    score = 3
                if score is not None:
                    preferred.append((int(score), float(out)))
        if not preferred:
            return None
        preferred.sort(key=lambda item: item[0])
        return float(preferred[0][1])

    def _mean_intensity_trace_for_vibration(
        self,
        arr: np.ndarray,
        has_frames_dim: bool,
        frame_count: int,
        shape: tuple[int, int],
        max_samples: int = 20000,
    ) -> np.ndarray:
        if not has_frames_dim or frame_count < 8:
            return np.asarray([], dtype=np.float64)
        stride = max(1, int(math.ceil(float(frame_count) / float(max_samples))))
        means: list[float] = []
        insp_side = self._inspection_crop_side(shape)
        for i in range(0, int(frame_count), stride):
            gray = _to_gray_u8(arr[i])
            if gray is None:
                continue
            if insp_side is not None:
                crop = self._crop_inspection_square(gray, int(insp_side))
                if crop is not None:
                    gray = crop
            means.append(float(np.mean(np.asarray(gray, dtype=np.float64))))
        trace = np.asarray(means, dtype=np.float64)
        if trace.size < 8:
            return np.asarray([], dtype=np.float64)
        return trace

    def _vibration_summary_from_trace(self, trace: np.ndarray, fps: float, frame_count: int) -> str:
        y = np.asarray(trace, dtype=np.float64)
        y = y[np.isfinite(y)]
        if y.size < 64:
            return "unknown"
        fs = float(fps)
        if not np.isfinite(fs) or fs <= 0.0:
            return "unknown"
        stride = max(1.0, float(frame_count) / float(y.size))
        fs_eff = fs / stride
        if fs_eff <= 2.0:
            return "unknown"

        mean_y = float(np.mean(y))
        x = np.arange(y.size, dtype=np.float64)
        try:
            coeff = np.polyfit(x, y, deg=1)
            y = y - ((coeff[0] * x) + coeff[1])
        except Exception:
            y = y - mean_y
        y = y - float(np.mean(y))
        if float(np.std(y)) <= 1e-9:
            return "not obvious"

        win = np.hanning(y.size)
        spec = np.fft.rfft(y * win)
        freqs = np.fft.rfftfreq(y.size, d=1.0 / fs_eff)
        power = np.abs(spec) ** 2
        fmin = max(5.0, fs_eff / max(2.0 * y.size, 1.0))
        fmax = min(0.48 * fs_eff, 1000.0)
        band = np.isfinite(freqs) & np.isfinite(power) & (freqs >= fmin) & (freqs <= fmax)
        if int(np.count_nonzero(band)) < 5:
            return "unknown"
        band_power = power[band]
        band_freqs = freqs[band]
        peak_i = int(np.argmax(band_power))
        peak_f = float(band_freqs[peak_i])
        med = float(np.median(band_power))
        ratio = float(band_power[peak_i] / max(med, 1e-18))
        amp_counts = float((2.0 * np.abs(spec[np.where(band)[0][peak_i]])) / max(float(np.sum(win)), 1e-9))
        amp_pct = float(100.0 * amp_counts / max(abs(mean_y), 1e-9))
        if ratio >= 25.0 and amp_pct >= 0.05:
            verdict = "likely on"
        elif ratio >= 10.0 and amp_pct >= 0.03:
            verdict = "possible"
        else:
            verdict = "not obvious"
        return f"{verdict} (peak {peak_f:.1f} Hz, amp {amp_pct:.2f}%, peak/median {ratio:.0f}x)"

    def _recording_summary(
        self,
        path: Path,
        arr: np.ndarray,
        has_frames_dim: bool,
        frame_count: int,
        fps: float,
        shape: tuple[int, int],
    ) -> str:
        objs = self._sidecar_json_objects(path)
        bg = self._metadata_background_summary(objs)
        exp_ms = self._metadata_exposure_ms(objs)
        exp_txt = f"{exp_ms:.4g} ms" if exp_ms is not None else "unknown"
        trace = self._mean_intensity_trace_for_vibration(arr, has_frames_dim, frame_count, shape)
        vib = self._vibration_summary_from_trace(trace, fps, frame_count)
        return f"BG subtract: {bg} | exposure: {exp_txt} | vibration: {vib}"

    def _extract_fps_from_obj(self, obj: object) -> Optional[float]:
        def parse_positive(v: object) -> Optional[float]:
            try:
                f = float(v)
                if f > 0.0:
                    return f
            except Exception:
                pass
            return None

        if isinstance(obj, dict):
            for k in ("actual_fps", "resulting_fps"):
                f = parse_positive(obj.get(k))
                if f is not None:
                    return f
            actual = obj.get("actual")
            if isinstance(actual, dict):
                f = parse_positive(actual.get("fps"))
                if f is not None:
                    return f
                timing = actual.get("timing_snapshot")
                if isinstance(timing, dict):
                    f = parse_positive(timing.get("fps"))
                    if f is not None:
                        return f
            timing = obj.get("timing_snapshot")
            if isinstance(timing, dict):
                f = parse_positive(timing.get("fps"))
                if f is not None:
                    return f
            f = parse_positive(obj.get("fps"))
            if f is not None:
                return f
            for v in obj.values():
                f = self._extract_fps_from_obj(v)
                if f is not None:
                    return f
        elif isinstance(obj, list):
            for v in obj:
                f = self._extract_fps_from_obj(v)
                if f is not None:
                    return f
        return None

    def _fps_from_sidecar(self, path: Path) -> Optional[float]:
        for obj in self._sidecar_json_objects(path):
            f = self._extract_fps_from_obj(obj)
            if f is not None:
                return f
        candidates = [path.with_suffix(".json"), path.parent / f"{path.stem}.json"]
        stem = path.stem
        if stem.startswith("spotrec_"):
            ts = stem[len("spotrec_") :]
            candidates.append(path.parent / f"spotrec_preview_{ts}.json")
        seen = set()
        for c in candidates:
            key = str(c.resolve()) if c.exists() else str(c)
            if key in seen:
                continue
            seen.add(key)
            obj = self._read_json_if_exists(c)
            if obj is None:
                continue
            f = self._extract_fps_from_obj(obj)
            if f is not None:
                return f
        return None

    def _resolve_fps(self, path: Path, shape: tuple[int, int], inspection_hint: bool) -> float:
        mode = str(self._fps_mode_var.get() or "Auto").strip()
        if mode == "1600":
            return 1600.0
        if mode == "77":
            return 77.0
        if mode == "Manual":
            try:
                f = float(self._fps_manual_var.get())
                if f > 0.0:
                    return f
            except Exception:
                pass
        sidecar_fps = self._fps_from_sidecar(path)
        if sidecar_fps is not None:
            return float(sidecar_fps)
        if inspection_hint or (self._inspection_crop_side(shape) is not None):
            return 1600.0
        return 77.0

    def _current_fps(self) -> float:
        n = len(self._spot_fps)
        if n > 0 and len(self._spot_centers) > 0:
            idx = max(0, min(int(self._spot_idx), len(self._spot_centers) - 1))
            if idx < n:
                f = float(self._spot_fps[idx])
                if f > 0.0:
                    return f
        f = float(self.source_fps)
        return f if f > 0.0 else 1.0

    def _fps_for_spot_idx(self, idx: int) -> float:
        if 0 <= int(idx) < len(self._spot_fps):
            f = float(self._spot_fps[int(idx)])
            if f > 0.0:
                return f
        f = float(self.source_fps)
        return f if f > 0.0 else 1.0

    def _get_bin_deg(self, default: float = 9.0) -> float:
        try:
            b = float(self._bin_deg_var.get())
        except Exception:
            return float(default)
        if not np.isfinite(b) or b <= 0.0:
            return float(default)
        return float(min(360.0, b))

    def _get_sphere_trail_len(self, default: int = 50) -> int:
        try:
            n = int(self._sphere_trail_var.get())
        except Exception:
            return int(default)
        if n <= 0:
            return int(default)
        return int(min(200000, n))

    def _get_sphere_speed(self, default: float = 1.0) -> float:
        try:
            s = float(self._sphere_speed_var.get())
        except Exception:
            return float(default)
        if not np.isfinite(s) or s <= 0.0:
            return float(default)
        return float(min(100.0, max(0.05, s)))

    def _get_dwell_avg_points(self) -> int:
        try:
            n = int(round(float(self._dwell_avg_points_var.get())))
        except Exception:
            n = 51
        n = max(11, int(n))
        if n % 2 == 0:
            n += 1
        return int(min(200001, n))

    def _get_dwell_jump_deg(self) -> float:
        try:
            v = float(self._dwell_jump_deg_var.get())
        except Exception:
            v = 25.0
        if not np.isfinite(v) or v <= 0.0:
            v = 25.0
        return float(min(180.0, max(1.0, v)))

    def _get_dwell_min_points(self) -> int:
        try:
            n = int(round(float(self._dwell_min_points_var.get())))
        except Exception:
            n = 10
        return int(min(200000, max(10, n)))

    def _get_dwell_bin_width_s(self) -> float:
        try:
            v = float(self._dwell_bin_width_s_var.get())
        except Exception:
            v = 0.05
        if not np.isfinite(v) or v <= 0.0:
            v = 0.05
        return float(min(1000.0, max(1e-4, v)))

    def _get_raw_play_speed(self) -> float:
        try:
            v = float(self._raw_play_speed_var.get())
        except Exception:
            v = 100.0
        if not np.isfinite(v) or v <= 0.0:
            v = 100.0
        return float(min(1000.0, max(0.1, v)))

    def _get_raw_play_trail(self) -> int:
        try:
            n = int(round(float(self._raw_play_trail_var.get())))
        except Exception:
            n = 200
        return int(min(200000, max(1, n)))

    def _raw_series_from_xy(self, xy: np.ndarray) -> dict:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = np.zeros((0, 2), dtype=np.float64)
        x = arr[:, 0] if arr.size else np.zeros((0,), dtype=np.float64)
        y = arr[:, 1] if arr.size else np.zeros((0,), dtype=np.float64)
        r = np.sqrt((x * x) + (y * y))
        two_phi = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
        theta, clipped = self._theta_from_radius(r)
        return {
            "xy": arr,
            "r": r.astype(np.float64, copy=False),
            "two_phi_deg": two_phi.astype(np.float64, copy=False),
            "theta_deg": np.degrees(theta).astype(np.float64, copy=False),
            "r_clipped": int(clipped),
        }

    def _current_raw_gallery_series(self) -> tuple[Optional[dict], float, str]:
        if not self._spot_xy_series:
            return None, float(self._current_fps()), "No rod loaded"
        idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
        xy = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        fps = float(self._current_fps())
        name = self._current_spot_label()
        return self._raw_series_from_xy(xy), fps, name

    def _theta_model_params(self) -> dict:
        key = str(self._theta_model_var.get() or "hole+fresnel").strip().lower()
        return dict(self.THETA_MODELS.get(key, self.THETA_MODELS["hole+fresnel"]))

    def _theta_from_radius(self, r: np.ndarray) -> tuple[np.ndarray, int]:
        params = self._theta_model_params()
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        r_max = float(params["r_max"])
        rr = np.asarray(r, dtype=np.float64)
        rr_safe = np.clip(rr, 0.0, max(0.0, r_max - 1e-12))
        val = (a * rr_safe) / np.maximum(1e-12, b - (c * rr_safe))
        val = np.clip(val, 0.0, 1.0)
        theta = np.arcsin(np.sqrt(val))
        clipped = int(np.count_nonzero((rr < 0.0) | (rr >= r_max) | ~np.isfinite(rr)))
        return theta.astype(np.float64, copy=False), clipped

    def _fit_axis_from_unit_sphere(
        self,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
            raise RuntimeError("Need at least 3 reconstructed unit-sphere points.")
        # Density-balanced fit:
        # when the rod gets stuck, many repeated samples pile up in one region.
        # We bin on the sphere and let each occupied bin contribute roughly equally.
        z = np.clip(arr[:, 2], -1.0, 1.0)
        az = np.mod(np.arctan2(arr[:, 1], arr[:, 0]), 2.0 * np.pi)
        nz = int(max(4, self.SPHERE_FIT_BINS_Z))
        nphi = int(max(8, self.SPHERE_FIT_BINS_PHI))
        iz = np.clip(((z + 1.0) * 0.5 * nz).astype(np.int32), 0, nz - 1)
        ip = np.clip((az / (2.0 * np.pi) * nphi).astype(np.int32), 0, nphi - 1)
        cell = iz.astype(np.int64) * np.int64(nphi) + ip.astype(np.int64)
        uniq, inv = np.unique(cell, return_inverse=True)
        occ = int(uniq.size)
        if occ >= 3:
            reps = np.zeros((occ, 3), dtype=np.float64)
            cnt = np.zeros((occ,), dtype=np.int32)
            for i in range(arr.shape[0]):
                j = int(inv[i])
                reps[j] += arr[i]
                cnt[j] += 1
            valid = cnt > 0
            reps = reps[valid]
            cnt = cnt[valid]
            nr = np.linalg.norm(reps, axis=1)
            good = nr > 1e-12
            reps = reps[good]
            if reps.shape[0] >= 3:
                reps = reps / nr[good][:, None]
                m = (reps.T @ reps) / max(1, int(reps.shape[0]))
            else:
                m = (arr.T @ arr) / max(1, int(arr.shape[0]))
                occ = int(arr.shape[0])
        else:
            m = (arr.T @ arr) / max(1, int(arr.shape[0]))
            occ = int(arr.shape[0])
        vals, vecs = np.linalg.eigh(m)
        k = vecs[:, int(np.argmax(vals))]
        if float(np.mean(arr @ k)) < 0.0:
            k = -k
        k = k / max(1e-12, float(np.linalg.norm(k)))

        z_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ref = z_ref if abs(float(np.dot(k, z_ref))) < 0.95 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = np.cross(k, ref)
        e1 = e1 / max(1e-12, float(np.linalg.norm(e1)))
        e2 = np.cross(k, e1)
        e2 = e2 / max(1e-12, float(np.linalg.norm(e2)))

        cos_g = float(np.mean(arr @ k))
        cos_g = max(-1.0, min(1.0, cos_g))
        center = cos_g * k
        gamma = math.acos(cos_g)
        return k, e1, e2, center, float(gamma), int(occ), int(arr.shape[0])

    @staticmethod
    def _wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi

    def _kmeans_two_on_sphere(self, u: np.ndarray, max_iter: int = 24) -> np.ndarray:
        arr = np.asarray(u, dtype=np.float64)
        n = int(arr.shape[0])
        if n < 2:
            return np.zeros((n,), dtype=np.int32)
        c0 = arr[0]
        d2 = np.sum((arr - c0[None, :]) ** 2, axis=1)
        i1 = int(np.argmax(d2))
        c1 = arr[i1]
        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(max(1, int(max_iter))):
            d0 = np.sum((arr - c0[None, :]) ** 2, axis=1)
            d1 = np.sum((arr - c1[None, :]) ** 2, axis=1)
            new_labels = (d1 < d0).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            m0 = arr[labels == 0]
            m1 = arr[labels == 1]
            if m0.shape[0] == 0 or m1.shape[0] == 0:
                break
            c0n = np.mean(m0, axis=0)
            c1n = np.mean(m1, axis=0)
            n0 = float(np.linalg.norm(c0n))
            n1 = float(np.linalg.norm(c1n))
            if n0 > 1e-12:
                c0 = c0n / n0
            if n1 > 1e-12:
                c1 = c1n / n1
        return labels

    def _raw_phi_from_axis(
        self,
        u: np.ndarray,
        k: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(u, dtype=np.float64)
        proj = arr - (arr @ k)[:, None] * k[None, :]
        p1 = proj @ e1
        p2 = proj @ e2
        pn = np.sqrt((p1 * p1) + (p2 * p2))
        raw = np.zeros((arr.shape[0],), dtype=np.float64)
        valid = pn > 1e-12
        raw[valid] = np.mod(np.arctan2(p2[valid], p1[valid]), 2.0 * np.pi)
        if np.any(~valid):
            bad = np.where(~valid)[0]
            for i in bad:
                raw[i] = raw[i - 1] if i > 0 else 0.0
        return raw, p1, p2

    def _phi_from_axis(
        self,
        u: np.ndarray,
        k: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw, p1, p2 = self._raw_phi_from_axis(u, k, e1, e2)
        if not bool(self._unwrap_phi_continuity_var.get()):
            phi_direct = np.mod(raw, np.pi)
            return phi_direct, phi_direct, p1, p2
        phi_unwrapped = np.unwrap(raw)
        phi_wrapped = np.mod(phi_unwrapped, 2.0 * np.pi)
        return phi_wrapped, phi_unwrapped, p1, p2

    def _dual_axis_phi_track(
        self,
        u: np.ndarray,
        ax0: dict,
        ax1: dict,
        labels_hint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        raw0, p10, p20 = self._raw_phi_from_axis(u, ax0["k"], ax0["e1"], ax0["e2"])
        raw1, p11, p21 = self._raw_phi_from_axis(u, ax1["k"], ax1["e1"], ax1["e2"])
        n = int(raw0.size)
        if n == 0:
            z = np.zeros((0,), dtype=np.float64)
            zi = np.zeros((0,), dtype=np.int32)
            return z, z, zi, z, z, 0

        if not bool(self._unwrap_phi_continuity_var.get()):
            axis_idx = labels_hint.astype(np.int32, copy=False) if labels_hint.size == n else np.zeros((n,), dtype=np.int32)
            axis_idx = np.where(axis_idx == 1, 1, 0).astype(np.int32, copy=False)
            direct = np.where(axis_idx == 1, raw1, raw0)
            p1 = np.where(axis_idx == 1, p11, p10)
            p2 = np.where(axis_idx == 1, p21, p20)
            direct = np.mod(direct, np.pi)
            switch_count = int(np.count_nonzero(axis_idx[1:] != axis_idx[:-1])) if n > 1 else 0
            return direct, direct, axis_idx, p1, p2, switch_count

        axis_idx = np.zeros((n,), dtype=np.int32)
        p1 = np.zeros((n,), dtype=np.float64)
        p2 = np.zeros((n,), dtype=np.float64)
        unwrapped = np.zeros((n,), dtype=np.float64)

        start = int(labels_hint[0]) if labels_hint.size else 0
        start = 0 if start not in (0, 1) else start
        axis_idx[0] = start
        unwrapped[0] = float(raw0[0] if start == 0 else raw1[0])
        p1[0] = float(p10[0] if start == 0 else p11[0])
        p2[0] = float(p20[0] if start == 0 else p21[0])
        switches = 0

        for t in range(1, n):
            prev = float(unwrapped[t - 1])
            c0 = prev + float(self._wrap_pi(float(raw0[t] - prev)))
            c1 = prev + float(self._wrap_pi(float(raw1[t] - prev)))
            if abs(c1 - prev) < abs(c0 - prev):
                axis_idx[t] = 1
                unwrapped[t] = c1
                p1[t] = float(p11[t])
                p2[t] = float(p21[t])
            else:
                axis_idx[t] = 0
                unwrapped[t] = c0
                p1[t] = float(p10[t])
                p2[t] = float(p20[t])
            if axis_idx[t] != axis_idx[t - 1]:
                switches += 1

        wrapped = np.mod(unwrapped, 2.0 * np.pi)
        return wrapped, unwrapped, axis_idx, p1, p2, int(switches)

    def _fit_unit_sphere_distribution(self, xy: np.ndarray) -> Optional[dict]:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return None

        r = np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))
        theta, clipped = self._theta_from_radius(r)
        phi = self._physical_phi_from_xy_series(arr, wrap_2pi=True)
        sin_th = np.sin(theta)
        u = np.column_stack((sin_th * np.cos(phi), sin_th * np.sin(phi), np.cos(theta)))

        dual_enabled = bool(self._sphere_dual_axis_var.get())
        try:
            axis_k, e1, e2, center, gamma, fit_cells_used, fit_points_total = self._fit_axis_from_unit_sphere(u)
        except Exception:
            return None
        phi_wrapped, phi_unwrapped, p1, p2 = self._phi_from_axis(u, axis_k, e1, e2)
        axis2_k = None
        axis2_center = None
        axis2_e1 = None
        axis2_e2 = None
        axis2_gamma = None
        axis2_circle = None
        axis_angle_deg = None
        axis_choice = np.zeros((u.shape[0],), dtype=np.int32)
        switch_count = 0
        fit_cells2_used = 0
        fit_points2_total = 0

        if dual_enabled and u.shape[0] >= 8:
            labels = self._kmeans_two_on_sphere(u)
            m0 = int(np.count_nonzero(labels == 0))
            m1 = int(np.count_nonzero(labels == 1))
            if m0 >= 3 and m1 >= 3:
                try:
                    k2, e12, e22, c2, g2, fit_cells2_used, fit_points2_total = self._fit_axis_from_unit_sphere(u[labels == 1])
                    k1, e11, e21, c1, g1, fit_cells_used, fit_points_total = self._fit_axis_from_unit_sphere(u[labels == 0])
                    ax0 = {"k": k1, "e1": e11, "e2": e21}
                    ax1 = {"k": k2, "e1": e12, "e2": e22}
                    phi_wrapped, phi_unwrapped, axis_choice, p1, p2, switch_count = self._dual_axis_phi_track(
                        u, ax0, ax1, labels
                    )
                    axis_k, e1, e2, center, gamma = k1, e11, e21, c1, g1
                    axis2_k, axis2_center, axis2_e1, axis2_e2, axis2_gamma = k2, c2, e12, e22, g2
                    # Axes are undirected lines, so report the acute separation angle.
                    axis_dot = abs(float(np.dot(axis_k, axis2_k)))
                    axis_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, axis_dot))))
                except Exception:
                    pass

        phi_max = 360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0
        phi_deg = np.mod(np.degrees(phi_wrapped), phi_max)
        bin_w = self._get_bin_deg(9.0)
        start_deg = self._best_window_start_deg(phi_deg, window_deg=bin_w) if bool(self._unwrap_phi_continuity_var.get()) else 0.0
        phi_rel = np.mod(phi_deg - start_deg, phi_max)
        edges = np.arange(0.0, phi_max + 1e-9, bin_w)
        counts, _ = np.histogram(phi_rel, bins=edges)
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        circle = (math.cos(gamma) * axis_k[None, :]) + (
            math.sin(gamma) * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :])
        )
        if axis2_k is not None and axis2_e1 is not None and axis2_e2 is not None and axis2_gamma is not None:
            axis2_circle = (math.cos(axis2_gamma) * axis2_k[None, :]) + (
                math.sin(axis2_gamma) * (np.cos(th)[:, None] * axis2_e1[None, :] + np.sin(th)[:, None] * axis2_e2[None, :])
            )
        return {
            "mode": "fitted",
            "u": u.astype(np.float64, copy=False),
            "theta_rad": theta.astype(np.float64, copy=False),
            "phi_input_rad": phi.astype(np.float64, copy=False),
            "dual_axis_enabled": bool(dual_enabled),
            "axis_k": axis_k,
            "axis_center": center,
            "e1": e1,
            "e2": e2,
            "gamma_rad": float(gamma),
            "axis2_k": axis2_k,
            "axis2_center": axis2_center,
            "axis2_e1": axis2_e1,
            "axis2_e2": axis2_e2,
            "axis2_gamma_rad": float(axis2_gamma) if axis2_gamma is not None else None,
            "axis_angle_deg": float(axis_angle_deg) if axis_angle_deg is not None else None,
            "axis_choice": axis_choice.astype(np.int32, copy=False),
            "switch_count": int(switch_count),
            "circle_xyz": circle.astype(np.float64, copy=False),
            "circle2_xyz": axis2_circle.astype(np.float64, copy=False) if axis2_circle is not None else None,
            "phi_axis_wrapped_rad": phi_wrapped.astype(np.float64, copy=False),
            "phi_axis_unwrapped_rad": phi_unwrapped.astype(np.float64, copy=False),
            "phi_axis_rel_deg": phi_rel.astype(np.float64, copy=False),
            "phi_axis_counts": counts.astype(np.float64, copy=False),
            "phi_axis_edges_deg": edges.astype(np.float64, copy=False),
            "plane_p1": p1.astype(np.float64, copy=False),
            "plane_p2": p2.astype(np.float64, copy=False),
            "r_clipped_count": int(clipped),
            "theta_model": str(self._theta_model_var.get() or "hole+fresnel"),
            "align_start_deg": float(start_deg),
            "fit_cells_used": int(fit_cells_used),
            "fit_points_total": int(fit_points_total),
            "fit_cells2_used": int(fit_cells2_used),
            "fit_points2_total": int(fit_points2_total),
        }

    def _raw_unit_sphere_distribution(self, xy: np.ndarray) -> Optional[dict]:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
            return None

        r = np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))
        theta, clipped = self._theta_from_radius(r)
        phi = self._physical_phi_from_xy_series(arr, wrap_2pi=True)
        sin_th = np.sin(theta)
        u = np.column_stack((sin_th * np.cos(phi), sin_th * np.sin(phi), np.cos(theta)))
        phi_max = 360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0
        phi_deg = np.mod(np.degrees(phi), phi_max)
        bin_w = self._get_bin_deg(9.0)
        edges = np.arange(0.0, phi_max + 1e-9, bin_w)
        counts, _ = np.histogram(phi_deg, bins=edges)
        return {
            "mode": "raw",
            "u": u.astype(np.float64, copy=False),
            "theta_rad": theta.astype(np.float64, copy=False),
            "phi_input_rad": phi.astype(np.float64, copy=False),
            "phi_axis_unwrapped_rad": (np.unwrap(phi) if bool(self._unwrap_phi_continuity_var.get()) else np.mod(phi, np.pi)).astype(np.float64, copy=False),
            "phi_axis_rel_deg": phi_deg.astype(np.float64, copy=False),
            "phi_axis_counts": counts.astype(np.float64, copy=False),
            "phi_axis_edges_deg": edges.astype(np.float64, copy=False),
            "plane_p1": arr[:, 0].astype(np.float64, copy=False),
            "plane_p2": arr[:, 1].astype(np.float64, copy=False),
            "r_clipped_count": int(clipped),
            "theta_model": str(self._theta_model_var.get() or "hole+fresnel"),
        }

    @staticmethod
    def _percentile_range_text(vals: np.ndarray, label: str, unit: str = "") -> str:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return f"{label}95-5=n/a {label}99-1=n/a"
        p1, p5, p95, p99 = np.percentile(arr, [1.0, 5.0, 95.0, 99.0])
        suffix = str(unit)
        return (
            f"{label}95-5={float(p95 - p5):.3g}{suffix} "
            f"{label}99-1={float(p99 - p1):.3g}{suffix}"
        )

    def _r_theta_percentile_summary(self, fit: dict) -> str:
        p1 = np.asarray(fit.get("plane_p1", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        p2 = np.asarray(fit.get("plane_p2", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        r = np.sqrt((p1 * p1) + (p2 * p2))
        theta_deg = np.degrees(np.asarray(fit.get("theta_rad", np.zeros((0,), dtype=np.float64)), dtype=np.float64))
        return f"{self._percentile_range_text(r, 'r')} | {self._percentile_range_text(theta_deg, 'theta', ' deg')}"

    def _current_unit_sphere_distribution(self) -> Optional[dict]:
        if not self._spot_xy_series:
            self._sphere_fit_i0 = 0
            self._sphere_fit_i1 = 0
            return None
        idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        if arr_full.ndim != 2 or arr_full.shape[1] != 2 or arr_full.shape[0] < 1:
            self._sphere_fit_i0 = 0
            self._sphere_fit_i1 = 0
            return None
        i0, i1 = self._selected_index_range(arr_full.shape[0], float(self._current_fps()))
        arr = arr_full[i0:i1]
        self._sphere_fit_i0 = int(i0)
        self._sphere_fit_i1 = int(i1)
        if bool(self._fit_circle_sphere_var.get()):
            if self._sphere_fit is None or self._sphere_fit.get("mode") != "fitted":
                self._sphere_fit = self._fit_unit_sphere_distribution(arr) if arr.shape[0] >= 3 else None
            return self._sphere_fit
        self._sphere_fit = self._raw_unit_sphere_distribution(arr)
        return self._sphere_fit

    def _crop_inspection_square(self, gray: np.ndarray, side: int) -> Optional[np.ndarray]:
        if gray is None or gray.ndim != 2:
            return None
        h, w = gray.shape
        n = int(max(1, side))
        n = min(n, int(h), int(w))
        # Spot-rec ROI data can be padded to 256 on one axis; use the first n x n block.
        return gray[:n, :n]

    def _inspection_intensity_bounds(self, n: int) -> tuple[int, int, int, int]:
        win = max(1, int(round(self._spot_window_size / 2.0)))
        if win % 2 == 0:
            win += 1
        half = win // 2
        ih = max(1, int(n) // 2)
        iw = max(1, int(n) // 2)
        cx = int(round((0.5 * (n - 1)) / 2.0))
        cy = int(round((0.5 * (n - 1)) / 2.0))
        x0 = max(0, cx - half)
        x1 = min(iw, cx + half + 1)
        y0 = max(0, cy - half)
        y1 = min(ih, cy + half + 1)
        return (x0, x1, y0, y1)

    def _xy_phi_from_gray_bounds(self, gray: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[float, float, float]:
        I0 = gray[0::2, 0::2]
        I45 = gray[0::2, 1::2]
        I135 = gray[1::2, 0::2]
        I90 = gray[1::2, 1::2]
        x0, x1, y0, y1 = bounds
        a0 = I0[y0:y1, x0:x1]
        a90 = I90[y0:y1, x0:x1]
        a45 = I45[y0:y1, x0:x1]
        a135 = I135[y0:y1, x0:x1]
        eps = 1e-6
        m0 = float(a0.mean()) if a0.size else 0.0
        m90 = float(a90.mean()) if a90.size else 0.0
        m45 = float(a45.mean()) if a45.size else 0.0
        m135 = float(a135.mean()) if a135.size else 0.0
        x = (m0 - m90) / (m0 + m90 + eps)
        y = (m45 - m135) / (m45 + m135 + eps)
        phi = float(0.5 * np.arctan2(y, x))
        return (float(x), float(y), float(phi))

    def _analyze_inspection_path(self, path: Path) -> tuple[list[tuple[float, float]], list[float], int, float, str]:
        arr, has_frames_dim, frame_count, gray0 = self._load_npy(path)
        side = self._inspection_crop_side(tuple(gray0.shape))
        if side is None:
            raise RuntimeError(f"{path.name}: not a small inspection stack (expected 256 x n, n<50).")
        fps = self._resolve_fps(path, tuple(gray0.shape), inspection_hint=True)
        summary = self._recording_summary(path, arr, has_frames_dim, frame_count, float(fps), tuple(gray0.shape))
        n = int(side)
        bounds = self._inspection_intensity_bounds(n)
        xy_series: list[tuple[float, float]] = []
        phi_series: list[float] = []
        for gray in self._iter_gray_frames(arr, has_frames_dim, frame_count):
            crop = self._crop_inspection_square(gray, n)
            if crop is None:
                continue
            x, y, phi = self._xy_phi_from_gray_bounds(crop, bounds)
            xy_series.append((x, y))
            phi_series.append(phi)
        return xy_series, phi_series, frame_count, float(fps), summary

    def _process_many_inspection_worker(self, paths: list[Path]) -> None:
        try:
            self._spot_window_size = int(self._spot_win_var.get())
        except Exception:
            self._ui_call(messagebox.showerror, "Parameters", "Invalid Phi window value.")
            self._ui_call(self._set_busy, False)
            return
        if self._spot_window_size < 3:
            self._ui_call(messagebox.showerror, "Parameters", "Phi window must be >= 3.")
            self._ui_call(self._set_busy, False)
            return
        if self._spot_window_size % 2 == 0:
            self._spot_window_size += 1

        xy_all: list[list[tuple[float, float]]] = []
        phi_all: list[list[float]] = []
        centers: list[tuple[float, float]] = []
        names: list[str] = []
        summaries: list[str] = []
        good_paths: list[Path] = []
        fps_list: list[float] = []
        skipped: list[str] = []
        for i, path in enumerate(paths, start=1):
            self._ui_call(self._status_var.set, f"Processing inspection file {i}/{len(paths)}: {path.name}")
            try:
                xy, phi, _n, fps, summary = self._analyze_inspection_path(path)
            except Exception as e:
                skipped.append(f"{path.name} ({e})")
                continue
            if not xy:
                skipped.append(f"{path.name} (no frames)")
                continue
            xy_all.append(xy)
            phi_all.append(phi)
            centers.append((0.0, 0.0))
            names.append(path.name)
            summaries.append(summary)
            good_paths.append(path)
            fps_list.append(float(fps))

        if not xy_all:
            self._ui_call(messagebox.showerror, "Inspection load", "No valid inspection files were loaded.")
            self._ui_call(self._status_var.set, "Inspection batch load failed.")
            self._ui_call(self._set_busy, False)
            return

        self._analysis_mode = "inspection_batch"
        self.source_path = good_paths[0]
        self.source_paths = list(good_paths)
        self.source_fps = float(np.mean(np.asarray(fps_list, dtype=np.float64))) if fps_list else 1600.0
        self._spot_fps = list(fps_list)
        self.npy_frames = None
        self.npy_has_frames_dim = False
        self.frame_count = 0
        self._source_shape = None
        self._s_map = None
        self._s_map_int = None
        self._spot_centers_all = list(centers)
        self._spot_centers = list(centers)
        self._spot_xy_series_all = list(xy_all)
        self._spot_xy_series = list(xy_all)
        self._spot_phi_series_all = list(phi_all)
        self._spot_phi_series = list(phi_all)
        self._spot_names = list(names)
        self._spot_file_summaries = list(summaries)
        self._arc_result = None
        self._spot_idx = 0
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._fft_peak_records = []
        self._sphere_fit = None
        self._fit_circle_sphere_var.set(False)
        self._stop_sphere_animation()
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1 = None
        self._avg_selected_spots = [False for _ in self._spot_centers]
        self._speed_theta_selected_spots = [False for _ in self._spot_centers]
        self._speed_theta_points = []
        self._brownian_last = None

        msg = f"Loaded {len(good_paths)}/{len(paths)} inspection files."
        if skipped:
            msg += f" Skipped {len(skipped)}."
        self._ui_call(self._status_var.set, msg)
        if skipped:
            self._ui_call(messagebox.showwarning, "Inspection load", "\n".join(skipped[:20]))
        self._ui_call(self._render_all)
        self._ui_call(self._render_arc_tab)
        self._ui_call(self._set_busy, False)

    def _anisotropy_range_s_map(
        self,
        min_x: np.ndarray,
        max_x: np.ndarray,
        min_y: np.ndarray,
        max_y: np.ndarray,
        raw_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        rx = max_x.astype(np.float32) - min_x.astype(np.float32)
        ry = max_y.astype(np.float32) - min_y.astype(np.float32)
        s_int = (rx * rx) + (ry * ry)

        h, w = raw_shape
        si_h, si_w = s_int.shape
        if (si_h, si_w) == (h, w):
            s_full = s_int
        elif (si_h, si_w) == (h - 1, w - 1):
            s_full = np.pad(s_int, ((0, 1), (0, 1)), mode="edge")
        else:
            s_full = cv2.resize(s_int, (w, h), interpolation=cv2.INTER_LINEAR)
        return s_full.astype(np.float32, copy=False), s_int.astype(np.float32, copy=False)

    def _find_centers_on_s_map(self, s_map_full: np.ndarray) -> list[tuple[float, float]]:
        edge = int(detect_spinners.EDGE_EXCLUDE_PX)
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
            min_area=int(detect_spinners.DOG_MIN_AREA),
            max_area=int(detect_spinners.DOG_MAX_AREA),
            connectivity=int(detect_spinners.DOG_CONNECTIVITY),
        )
        if offset:
            centers = [(cx + offset, cy + offset) for (cx, cy) in centers]

        m = int(detect_spinners.EDGE_EXCLUDE_PX)
        return [
            (cx, cy)
            for (cx, cy) in centers
            if (m <= cx <= (w - 1 - m)) and (m <= cy <= (h - 1 - m))
        ]

    def _sort_centers_by_s_int(
        self,
        centers_full: list[tuple[float, float]],
        s_map_full: np.ndarray,
        s_map_int: np.ndarray,
    ) -> list[tuple[float, float]]:
        if not centers_full:
            return []
        scored: list[tuple[float, tuple[float, float]]] = []
        h_int, w_int = s_map_int.shape
        h_full, w_full = s_map_full.shape
        sx = (float(w_int - 1) / float(max(1, w_full - 1))) if w_int > 1 else 0.0
        sy = (float(h_int - 1) / float(max(1, h_full - 1))) if h_int > 1 else 0.0
        for cx, cy in centers_full:
            ix = int(round(float(cx) * sx))
            iy = int(round(float(cy) * sy))
            s_val = float(s_map_int[iy, ix]) if (0 <= ix < w_int and 0 <= iy < h_int) else float("-inf")
            scored.append((s_val, (cx, cy)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored]

    def _update_spot_bounds_intensity(self, raw_shape: tuple[int, int]) -> None:
        h, w = raw_shape
        ih, iw = h // 2, w // 2
        win = max(1, int(round(self._spot_window_size / 2.0)))
        if win % 2 == 0:
            win += 1
        half = win // 2
        bounds = []
        for cx, cy in self._spot_centers_all:
            ix = int(round(cx / 2.0))
            iy = int(round(cy / 2.0))
            x0 = max(0, ix - half)
            x1 = min(iw, ix + half + 1)
            y0 = max(0, iy - half)
            y1 = min(ih, iy + half + 1)
            bounds.append((x0, x1, y0, y1))
        self._spot_bounds_int_all = bounds

    def _append_xy_from_frame(self, gray: np.ndarray) -> None:
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
            phi = float(0.5 * np.arctan2(y, x))
            self._spot_xy_series_all[i].append((x, y))
            self._spot_phi_series_all[i].append(phi)

    def _ring_likeness_score(self, xy_series: list[tuple[float, float]], eps: float = 1e-12) -> float:
        if len(xy_series) < 20:
            return 0.0
        arr = np.asarray(xy_series, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return 0.0
        x = arr[:, 0]
        y = arr[:, 1]
        zx = x - float(np.mean(x))
        zy = y - float(np.mean(y))
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
        Z = np.stack([zx, zy], axis=0)
        U = W @ Z
        r = np.sqrt(U[0] * U[0] + U[1] * U[1])
        q25 = float(np.quantile(r, 0.25))
        q75 = float(np.quantile(r, 0.75))
        score = q25 / (q75 + eps)
        return float(min(1.0, max(0.0, score)))

    def _spot_xy_max_axis_range(self, series: list[tuple[float, float]]) -> float:
        if not series:
            return float("-inf")
        arr = np.asarray(series, dtype=np.float32)
        if arr.size == 0:
            return float("-inf")
        range_x = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
        range_y = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
        return max(range_x, range_y)

    def _spot_xy_range_score(self, series: list[tuple[float, float]]) -> float:
        if not series:
            return float("-inf")
        arr = np.asarray(series, dtype=np.float32)
        if arr.size == 0:
            return float("-inf")
        range_x = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
        range_y = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
        return (range_x * range_x) + (range_y * range_y)

    def _directionality_B_only(self, xy_series: list[tuple[float, float]]) -> Optional[float]:
        if len(xy_series) < 32:
            return None
        arr = np.asarray(xy_series, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        z = arr[:, 0].astype(np.complex64) + 1j * arr[:, 1].astype(np.complex64)
        fs = float(self.source_fps) if self.source_fps > 0.0 else 1.0
        nperseg = int(min(256, len(z)))
        if nperseg < 8:
            return None
        noverlap = nperseg // 2
        freqs, psd = _safe_welch(z, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)
        if freqs.size == 0 or psd.size == 0:
            return None
        order = np.argsort(freqs)
        freqs = freqs[order]
        psd = psd[order]
        p_plus = float(np.sum(psd[(freqs > 0.0) & np.isfinite(psd)]))
        p_minus = float(np.sum(psd[(freqs < 0.0) & np.isfinite(psd)]))
        eps = 1e-18
        return float((p_plus - p_minus) / (p_plus + p_minus + eps))

    def _apply_filters(self) -> None:
        ring_thr = float(self._ring_score_min)
        abs_enabled = bool(self._abs_range_filter_enabled)
        dir_enabled = bool(self._dir_filter_enabled)

        keep = []
        for i, series in enumerate(self._spot_xy_series_all):
            if ring_thr > 0.0 and self._ring_likeness_score(series) < ring_thr:
                continue
            if abs_enabled and self._spot_xy_max_axis_range(series) <= float(self.ABS_RANGE_MIN):
                continue
            if dir_enabled:
                b = self._directionality_B_only(series)
                if b is None or abs(float(b)) < float(self.DIR_FILTER_B_MIN):
                    continue
            keep.append(i)

        centers = [self._spot_centers_all[i] for i in keep]
        xy = [self._spot_xy_series_all[i] for i in keep]
        phi = [self._spot_phi_series_all[i] for i in keep]
        order = list(range(len(centers)))
        if xy:
            scores = [self._spot_xy_range_score(s) for s in xy]
            if not all(v == float("-inf") for v in scores):
                order = sorted(order, key=lambda i: scores[i], reverse=True)
        self._spot_centers = [centers[i] for i in order]
        self._spot_xy_series = [xy[i] for i in order]
        self._spot_phi_series = [phi[i] for i in order]
        self._spot_idx = 0

    def _process_npy_worker(self, path: Path) -> None:
        try:
            self._dog_k_std = float(self._dog_k_var.get())
            self._spot_window_size = int(self._spot_win_var.get())
            self._ring_score_min = float(self._ring_score_min_var.get())
            self._abs_range_filter_enabled = bool(self._abs_var.get())
            self._dir_filter_enabled = bool(self._dir_var.get())
        except Exception:
            self._ui_call(messagebox.showerror, "Parameters", "Invalid numeric analysis parameter(s).")
            self._ui_call(self._set_busy, False)
            return

        if self._spot_window_size < 3:
            self._ui_call(messagebox.showerror, "Parameters", "Phi window must be >= 3.")
            self._ui_call(self._set_busy, False)
            return
        if self._spot_window_size % 2 == 0:
            self._spot_window_size += 1

        try:
            arr, has_frames_dim, frame_count, gray0 = self._load_npy(path)
            shape = tuple(gray0.shape)
            insp_side = self._inspection_crop_side(shape)
            use_fps = self._resolve_fps(path, shape, inspection_hint=(insp_side is not None))
            file_summary = self._recording_summary(path, arr, has_frames_dim, frame_count, float(use_fps), shape)

            if insp_side is not None:
                self._analysis_mode = "inspection"
                n = int(insp_side)
                self._s_map = None
                self._s_map_int = None
                self._spot_centers_all = [(0.5 * (n - 1), 0.5 * (n - 1))]
                self._spot_xy_series_all = [[]]
                self._spot_phi_series_all = [[]]
                self._update_spot_bounds_intensity((n, n))

                total = int(frame_count)
                for i, gray in enumerate(self._iter_gray_frames(arr, has_frames_dim, frame_count)):
                    crop = self._crop_inspection_square(gray, n)
                    if crop is None:
                        continue
                    self._append_xy_from_frame(crop)
                    if total > 1 and ((i + 1) % max(1, total // 20) == 0 or (i + 1) == total):
                        self._ui_call(self._status_var.set, f"Inspection XY/phi: {i + 1}/{total}")

                self._spot_centers = list(self._spot_centers_all)
                self._spot_xy_series = list(self._spot_xy_series_all)
                self._spot_phi_series = list(self._spot_phi_series_all)
                self._spot_idx = 0
            else:
                self._analysis_mode = "widefield"
                qu_recon = make_qu_reconstructor(shape, out_dtype=np.float32)
                s_frames = max(2, int(detect_spinners.S_MAP_FRAMES))

                min_x_sm = max_x_sm = None
                min_y_sm = max_y_sm = None
                x_sm = y_sm = None

                for i, gray in enumerate(self._iter_gray_frames(arr, has_frames_dim, frame_count)):
                    if i >= s_frames:
                        break
                    X, Y = qu_recon(gray)
                    if x_sm is None:
                        x_sm = np.empty_like(X)
                        y_sm = np.empty_like(Y)
                    cv2.boxFilter(
                        X,
                        ddepth=-1,
                        ksize=(int(detect_spinners.S_MAP_SMOOTH_K), int(detect_spinners.S_MAP_SMOOTH_K)),
                        dst=x_sm,
                        normalize=True,
                        borderType=cv2.BORDER_REPLICATE,
                    )
                    cv2.boxFilter(
                        Y,
                        ddepth=-1,
                        ksize=(int(detect_spinners.S_MAP_SMOOTH_K), int(detect_spinners.S_MAP_SMOOTH_K)),
                        dst=y_sm,
                        normalize=True,
                        borderType=cv2.BORDER_REPLICATE,
                    )
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

                if min_x_sm is None:
                    raise RuntimeError("No frames available for S-map.")

                s_full, s_int = self._anisotropy_range_s_map(min_x_sm, max_x_sm, min_y_sm, max_y_sm, shape)
                centers = self._find_centers_on_s_map(s_full)
                centers = self._sort_centers_by_s_int(centers, s_full, s_int)

                self._s_map = s_full
                self._s_map_int = s_int
                self._spot_centers_all = centers
                self._spot_xy_series_all = [[] for _ in centers]
                self._spot_phi_series_all = [[] for _ in centers]
                self._update_spot_bounds_intensity(shape)

                total = int(frame_count)
                for i, gray in enumerate(self._iter_gray_frames(arr, has_frames_dim, frame_count)):
                    self._append_xy_from_frame(gray)
                    if total > 1 and ((i + 1) % max(1, total // 20) == 0 or (i + 1) == total):
                        self._ui_call(self._status_var.set, f"Computing XY/phi: {i + 1}/{total}")

                self._apply_filters()

            self.source_path = path
            self.source_paths = [path]
            self.source_fps = float(use_fps)
            self._spot_fps = [float(use_fps) for _ in self._spot_centers]
            self.npy_frames = arr
            self.npy_has_frames_dim = has_frames_dim
            self.frame_count = frame_count
            self._source_shape = shape
            self._fit_center = None
            self._fit_radius = None
            self._fit_shifted_xy = None
            self._fit_shifted_phi = None
            self._fit_fft_freqs = None
            self._fit_fft_psd = None
            self._fft_peak_records = []
            self._sphere_fit = None
            self._fit_circle_sphere_var.set(False)
            self._stop_sphere_animation()
            self._phi_sel_t0 = 0.0
            self._phi_sel_t1 = None
            self._spot_names = [path.name for _ in self._spot_centers] if self._analysis_mode == "inspection" else []
            self._avg_selected_spots = [False for _ in self._spot_centers]
            self._speed_theta_selected_spots = [False for _ in self._spot_centers]
            self._speed_theta_points = []
            self._brownian_last = None
            self._arc_result = None
            if self._analysis_mode == "inspection":
                self._spot_file_summaries = [file_summary for _ in self._spot_centers]
            else:
                self._spot_file_summaries = [file_summary for _ in self._spot_centers]

            self._ui_call(
                self._status_var.set,
                (
                    f"Loaded {path.name}: inspection mode, single ROI spot analyzed."
                    f" FPS={self.source_fps:.3f}. {file_summary}"
                    if self._analysis_mode == "inspection"
                    else f"Loaded {path.name}: {len(self._spot_centers)} filtered spot(s) "
                    f"from {len(self._spot_centers_all)} candidates. FPS={self.source_fps:.3f}. {file_summary}"
                ),
            )
            self._ui_call(self._render_all)
            self._ui_call(self._render_arc_tab)
        except Exception as e:
            self._ui_call(messagebox.showerror, "Analysis error", str(e))
            self._ui_call(self._status_var.set, "Analysis failed.")
        finally:
            self._ui_call(self._set_busy, False)

    def _prev_spot(self) -> None:
        if not self._spot_centers:
            return
        self._clear_arc_result(render=True)
        self._spot_idx = (self._spot_idx - 1) % len(self._spot_centers)
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._fft_marker_freq_hz = None
        self._sphere_fit = None
        self._fit_circle_sphere_var.set(False)
        self._stop_sphere_animation()
        self._brownian_last = None
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1 = None
        self._render_all()

    def _next_spot(self) -> None:
        if not self._spot_centers:
            return
        self._clear_arc_result(render=True)
        self._spot_idx = (self._spot_idx + 1) % len(self._spot_centers)
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._fft_marker_freq_hz = None
        self._sphere_fit = None
        self._fit_circle_sphere_var.set(False)
        self._stop_sphere_animation()
        self._brownian_last = None
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1 = None
        self._render_all()

    def _analyze_current_spot_distribution(self) -> None:
        if not bool(self._fit_circle_sphere_var.get()):
            self._fit_circle_sphere_var.set(True)
        self._clear_arc_result(render=True)
        if not self._spot_xy_series:
            return
        idx = int(self._spot_idx)
        if idx < 0 or idx >= len(self._spot_xy_series):
            return
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        i0, i1 = self._selected_index_range(arr_full.shape[0], float(self._current_fps()))
        arr = arr_full[i0:i1]
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            messagebox.showerror("Angle distribution", "Need at least 3 XY points in selected time range for circle fitting.")
            return

        fit = self._fit_and_aligned_distribution(arr)
        if fit is None:
            messagebox.showerror("Angle distribution", "Circle fit failed.")
            return
        sphere_fit = self._fit_unit_sphere_distribution(arr)
        if sphere_fit is None:
            messagebox.showerror("Unit sphere", "Unit-sphere reconstruction/axis fit failed.")
            return
        self._fit_center = (fit["cx"], fit["cy"])
        self._fit_radius = fit["radius"]
        self._fit_shifted_xy = fit["xy_shift"]
        self._fit_shifted_phi = fit["phi_shift_rad"]
        self._fit_fft_freqs, self._fit_fft_psd = self._compute_complex_xy_fft(arr, float(self._current_fps()))
        self._sphere_fit = sphere_fit
        self._sphere_fit_i0 = int(i0)
        self._sphere_fit_i1 = int(i1)
        self._sphere_anim_idx = 0
        self._stop_sphere_animation()
        self._brownian_last = None
        self._render_all()

    def _selected_index_range(self, n: int, fps: float) -> tuple[int, int]:
        if n <= 1:
            return (0, max(1, n))
        fs = float(fps) if fps and fps > 0.0 else 1.0
        tmax = float(n - 1) / fs
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1) if self._phi_sel_t1 is not None else float(tmax)
        t0 = max(0.0, min(t0, tmax))
        t1 = max(0.0, min(t1, tmax))
        if t1 < t0:
            t0, t1 = t1, t0
        i0 = int(np.floor(t0 * fs))
        i1 = int(np.ceil(t1 * fs)) + 1
        i0 = max(0, min(i0, n - 1))
        i1 = max(i0 + 1, min(i1, n))
        return (i0, i1)

    def _draw_phi_selection(self, tmax: float) -> None:
        self._phi_sel_tmax = max(0.0, float(tmax))
        if self._phi_sel_t1 is None:
            self._phi_sel_t0 = 0.0
            self._phi_sel_t1 = float(self._phi_sel_tmax)
        self._phi_sel_t0 = max(0.0, min(float(self._phi_sel_t0), float(self._phi_sel_tmax)))
        self._phi_sel_t1 = max(0.0, min(float(self._phi_sel_t1), float(self._phi_sel_tmax)))
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1)
        if t1 < t0:
            t0, t1 = t1, t0
            self._phi_sel_t0, self._phi_sel_t1 = t0, t1
        self._phi_sel_span = self._ax_phi.axvspan(t0, t1, color="tab:blue", alpha=0.07, zorder=0)
        self._phi_sel_line0 = self._ax_phi.axvline(t0, color="tab:blue", linestyle="--", linewidth=1.2)
        self._phi_sel_line1 = self._ax_phi.axvline(t1, color="tab:blue", linestyle="--", linewidth=1.2)

    def _on_fft_press(self, event) -> None:
        if event is None or event.inaxes is not self._ax_fft or event.xdata is None:
            return
        self._drag_fft_marker = True
        self._set_fft_marker_from_x(float(event.xdata))

    def _on_fft_motion(self, event) -> None:
        if not bool(self._drag_fft_marker):
            return
        if event is None or event.inaxes is not self._ax_fft or event.xdata is None:
            return
        self._set_fft_marker_from_x(float(event.xdata))

    def _on_fft_release(self, event) -> None:
        if not bool(self._drag_fft_marker):
            return
        self._drag_fft_marker = False
        if event is None or event.inaxes is not self._ax_fft or event.xdata is None:
            return
        self._set_fft_marker_from_x(float(event.xdata))

    def _set_fft_marker_from_x(self, x_hz: float) -> None:
        freqs, _psd = self._current_fft_positive_freqs_psd()
        if freqs.size == 0:
            return
        xmax = self._fft_max_hz(float(np.max(freqs)))
        self._fft_marker_freq_hz = max(0.0, min(float(x_hz), float(xmax)))
        self._render_all()

    def _draw_sphere_phi_selection(self, tmax: float) -> None:
        self._phi_sel_tmax_sphere = max(0.0, float(tmax))
        if self._phi_sel_t1 is None:
            self._phi_sel_t0 = 0.0
            self._phi_sel_t1 = float(self._phi_sel_tmax_sphere)
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1 if self._phi_sel_t1 is not None else self._phi_sel_tmax_sphere)
        lo = min(t0, t1)
        hi = max(t0, t1)
        lo = max(0.0, min(lo, self._phi_sel_tmax_sphere))
        hi = max(0.0, min(hi, self._phi_sel_tmax_sphere))
        self._phi_sel_span_sphere = self._ax_sphere_phi.axvspan(lo, hi, color="tab:blue", alpha=0.07, zorder=0)
        self._phi_sel_line0_sphere = self._ax_sphere_phi.axvline(lo, color="tab:blue", linestyle="--", linewidth=1.2)
        self._phi_sel_line1_sphere = self._ax_sphere_phi.axvline(hi, color="tab:blue", linestyle="--", linewidth=1.2)

    def _update_phi_selection_overlay(self) -> None:
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1 if self._phi_sel_t1 is not None else self._phi_sel_tmax)
        lo = min(t0, t1)
        hi = max(t0, t1)

        # Spot-view overlay.
        if self._phi_sel_line0 is not None:
            try:
                lo_s = max(0.0, min(lo, float(self._phi_sel_tmax)))
                self._phi_sel_line0.set_xdata([lo_s, lo_s])
            except Exception:
                self._phi_sel_line0 = None
        if self._phi_sel_line1 is not None:
            try:
                hi_s = max(0.0, min(hi, float(self._phi_sel_tmax)))
                self._phi_sel_line1.set_xdata([hi_s, hi_s])
            except Exception:
                self._phi_sel_line1 = None
        if self._phi_sel_span is not None:
            try:
                self._phi_sel_span.remove()
            except Exception:
                pass
            lo_s = max(0.0, min(lo, float(self._phi_sel_tmax)))
            hi_s = max(0.0, min(hi, float(self._phi_sel_tmax)))
            self._phi_sel_span = self._ax_phi.axvspan(lo_s, hi_s, color="tab:blue", alpha=0.07, zorder=0)

        # Unit-sphere overlay.
        if self._phi_sel_line0_sphere is not None:
            try:
                lo_u = max(0.0, min(lo, float(self._phi_sel_tmax_sphere)))
                self._phi_sel_line0_sphere.set_xdata([lo_u, lo_u])
            except Exception:
                self._phi_sel_line0_sphere = None
        if self._phi_sel_line1_sphere is not None:
            try:
                hi_u = max(0.0, min(hi, float(self._phi_sel_tmax_sphere)))
                self._phi_sel_line1_sphere.set_xdata([hi_u, hi_u])
            except Exception:
                self._phi_sel_line1_sphere = None
        if self._phi_sel_span_sphere is not None:
            try:
                self._phi_sel_span_sphere.remove()
            except Exception:
                pass
            lo_u = max(0.0, min(lo, float(self._phi_sel_tmax_sphere)))
            hi_u = max(0.0, min(hi, float(self._phi_sel_tmax_sphere)))
            self._phi_sel_span_sphere = self._ax_sphere_phi.axvspan(lo_u, hi_u, color="tab:blue", alpha=0.07, zorder=0)

    def _on_phi_press(self, event) -> None:
        if event is None or event.inaxes is not self._ax_phi:
            return
        if event.xdata is None or self._phi_sel_t1 is None:
            return
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1)
        tr = max(1e-6, float(self._phi_sel_tmax))
        tol = max(0.02 * tr, 0.03)
        d0 = abs(float(event.xdata) - t0)
        d1 = abs(float(event.xdata) - t1)
        if min(d0, d1) > tol:
            return
        self._drag_phi_handle = "start" if d0 <= d1 else "end"
        self._drag_phi_source = "spot"

    def _on_phi_motion(self, event) -> None:
        if self._drag_phi_handle is None or self._drag_phi_source != "spot":
            return
        if event is None or event.inaxes is not self._ax_phi or event.xdata is None:
            return
        x = max(0.0, min(float(event.xdata), float(self._phi_sel_tmax)))
        if self._drag_phi_handle == "start":
            self._phi_sel_t0 = x
        else:
            self._phi_sel_t1 = x
        self._update_phi_selection_overlay()
        self._canvas.draw_idle()
        self._sphere_canvas.draw_idle()

    def _on_phi_release(self, event) -> None:
        if self._drag_phi_handle is None or self._drag_phi_source != "spot":
            return
        self._clear_arc_result(render=True)
        self._drag_phi_handle = None
        self._drag_phi_source = None
        # Range changed; previous fit is no longer valid for "further analysis".
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._fft_marker_freq_hz = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        self._refresh_sphere_fit_from_current_selection()
        self._render_all()

    def _on_sphere_phi_press(self, event) -> None:
        if event is None or event.inaxes is not self._ax_sphere_phi:
            return
        if event.xdata is None or self._phi_sel_t1 is None:
            return
        t0 = float(self._phi_sel_t0)
        t1 = float(self._phi_sel_t1)
        tr = max(1e-6, float(self._phi_sel_tmax_sphere))
        tol = max(0.02 * tr, 0.03)
        d0 = abs(float(event.xdata) - t0)
        d1 = abs(float(event.xdata) - t1)
        if min(d0, d1) > tol:
            return
        self._drag_phi_handle = "start" if d0 <= d1 else "end"
        self._drag_phi_source = "sphere"

    def _on_sphere_phi_motion(self, event) -> None:
        if self._drag_phi_handle is None or self._drag_phi_source != "sphere":
            return
        if event is None or event.inaxes is not self._ax_sphere_phi or event.xdata is None:
            return
        x = max(0.0, min(float(event.xdata), float(self._phi_sel_tmax_sphere)))
        if self._drag_phi_handle == "start":
            self._phi_sel_t0 = x
        else:
            self._phi_sel_t1 = x
        self._update_phi_selection_overlay()
        self._sphere_canvas.draw_idle()
        self._canvas.draw_idle()

    def _on_sphere_phi_release(self, event) -> None:
        if self._drag_phi_handle is None or self._drag_phi_source != "sphere":
            return
        self._clear_arc_result(render=True)
        self._drag_phi_handle = None
        self._drag_phi_source = None
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._fit_fft_freqs = None
        self._fit_fft_psd = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        self._refresh_sphere_fit_from_current_selection()
        self._render_all()

    def _fit_and_aligned_distribution(self, xy: np.ndarray) -> Optional[dict]:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return None
        x = arr[:, 0]
        y = arr[:, 1]
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = (x * x) + (y * y)
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        cx = float(sol[0])
        cy = float(sol[1])
        c0 = float(sol[2])
        r2 = c0 + (cx * cx) + (cy * cy)
        r = float(np.sqrt(max(0.0, r2)))
        x_shift = x - cx
        y_shift = y - cy
        xy_shift = np.column_stack((x_shift, y_shift))
        phi_shift = self._physical_phi_from_xy_series(xy_shift, wrap_2pi=True)
        phi_max = 360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0
        phi_deg = np.mod(np.degrees(phi_shift), phi_max)
        bin_w = self._get_bin_deg(9.0)
        start_deg = self._best_window_start_deg(phi_deg, window_deg=bin_w) if bool(self._unwrap_phi_continuity_var.get()) else 0.0
        phi_rel = np.mod(phi_deg - start_deg, phi_max)
        bins_deg = np.arange(0.0, phi_max + 1e-9, bin_w)
        counts, edges = np.histogram(phi_rel, bins=bins_deg)
        return {
            "cx": cx,
            "cy": cy,
            "radius": r,
            "xy_shift": xy_shift,
            "phi_shift_rad": phi_shift,
            "phi_rel_deg": phi_rel,
            "counts": counts.astype(np.float64, copy=False),
            "edges": edges.astype(np.float64, copy=False),
        }

    def _physical_phi_from_xy_series(self, xy: np.ndarray, wrap_2pi: bool = True) -> np.ndarray:
        """
        Convert XY trajectory to physical angle phi.
        XY is a double cover: physical phi = 0.5 * angle(X+iY).
        In continuity mode, unwrap XY angle first. In direct mode, fold physical phi to [0, pi).
        """
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return np.asarray([], dtype=np.float64)
        xy_ang = np.arctan2(arr[:, 1], arr[:, 0])  # [-pi, pi]
        if not bool(self._unwrap_phi_continuity_var.get()):
            return np.mod(0.5 * xy_ang, np.pi)
        xy_ang_unwrapped = np.unwrap(xy_ang)  # continuous XY phase
        phi = 0.5 * xy_ang_unwrapped  # physical phase
        if wrap_2pi:
            phi = np.mod(phi, 2.0 * np.pi)
        return phi

    def _phi_series_after_time_average_subtraction(self, phi_wrapped: np.ndarray, fps: float) -> np.ndarray:
        phi_arr = np.asarray(phi_wrapped, dtype=np.float64)
        if phi_arr.ndim != 1 or phi_arr.size == 0:
            return np.asarray([], dtype=np.float64)
        if not bool(self._phi_residual_avg_var.get()):
            return phi_arr
        win = self._parse_int(self._phi_residual_window_var.get())
        if win is None or win <= 0:
            return phi_arr
        if bool(self._unwrap_phi_continuity_var.get()):
            phi_unwrapped = np.unwrap(phi_arr)
        else:
            phi_unwrapped = np.mod(phi_arr, np.pi)
        if win <= 1 or phi_unwrapped.size <= 1:
            return phi_unwrapped - np.mean(phi_unwrapped)
        if (win % 2) == 0:
            win += 1
        kernel = np.ones((win,), dtype=np.float64) / float(win)
        avg = np.convolve(phi_unwrapped, kernel, mode="same")
        return phi_unwrapped - avg

    def _plot_speed_vs_time(self, phi_wrapped: np.ndarray, fps: float) -> None:
        if self._ax_speed is None:
            return
        self._ax_speed.clear()
        phi_arr = np.asarray(phi_wrapped, dtype=np.float64)
        if phi_arr.ndim != 1 or phi_arr.size < 3:
            self._ax_speed.set_title("Angular speed vs time")
            self._ax_speed.text(0.5, 0.5, "Need phi(t)", ha="center", va="center", transform=self._ax_speed.transAxes)
            return
        fs = max(1e-9, float(fps))
        phi_unwrapped = np.unwrap(phi_arr)
        speed_deg_s = np.degrees(np.gradient(phi_unwrapped) * fs)
        win = min(int(self._get_speed_avg_points()), int(speed_deg_s.size))
        if win > 1 and (win % 2) == 0:
            win -= 1
        if win > 1:
            kernel = np.ones((win,), dtype=np.float64) / float(win)
            speed_plot = np.convolve(speed_deg_s, kernel, mode="same")
        else:
            speed_plot = speed_deg_s
        t = np.arange(speed_plot.size, dtype=np.float64) / fs
        self._ax_speed.plot(t, speed_plot, color="tab:orange", lw=1.0)
        self._ax_speed.axhline(0.0, color="0.75", lw=1.0, linestyle=":")
        self._ax_speed.set_xlabel("time (s)")
        self._ax_speed.set_ylabel("dphi/dt (deg/s)")
        self._ax_speed.set_title(f"Angular speed vs time (moving average {win} points)")
        self._ax_speed.grid(alpha=0.25)

    def _plot_phi_distribution_fft(self, phi_deg: np.ndarray) -> None:
        if self._ax_phi_dist_fft is None:
            return
        self._ax_phi_dist_fft.clear()
        vals = np.mod(np.asarray(phi_deg, dtype=np.float64), 360.0)
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            self._ax_phi_dist_fft.set_title("Fourier transform of phi distribution")
            self._ax_phi_dist_fft.text(
                0.5,
                0.5,
                "Need shifted phi distribution",
                ha="center",
                va="center",
                transform=self._ax_phi_dist_fft.transAxes,
            )
            return
        n_bins = 360
        counts, _edges = np.histogram(vals, bins=np.linspace(0.0, 360.0, n_bins + 1))
        counts = counts.astype(np.float64)
        counts = counts - float(np.mean(counts))
        coeff = np.fft.rfft(counts)
        harmonics = np.arange(coeff.size, dtype=np.int32)
        amp = np.abs(coeff) / max(1.0, float(n_bins))
        if amp.size > 1:
            amp[1:-1] *= 2.0
        hmax = int(min(80, harmonics[-1])) if harmonics.size else 0
        keep = (harmonics >= 1) & (harmonics <= hmax)
        if np.any(keep):
            self._ax_phi_dist_fft.plot(
                harmonics[keep],
                amp[keep],
                marker="o",
                ms=3.0,
                lw=1.0,
                color="tab:red",
            )
        if amp.size > 23:
            phase23 = float(np.angle(coeff[23]))
            self._ax_phi_dist_fft.axvline(23, color="black", linestyle="--", lw=1.2)
            self._ax_phi_dist_fft.text(
                23,
                0.98,
                f"23: amp={amp[23]:.3g}, phase={phase23:.2f} rad",
                ha="center",
                va="top",
                fontsize=8,
                transform=self._ax_phi_dist_fft.get_xaxis_transform(),
            )
        self._ax_phi_dist_fft.set_xlim(1, max(24, hmax))
        self._ax_phi_dist_fft.set_xlabel("distribution harmonic number")
        self._ax_phi_dist_fft.set_ylabel("amplitude")
        self._ax_phi_dist_fft.set_title("Fourier transform of phi distribution")
        self._ax_phi_dist_fft.grid(alpha=0.25)

    def _render_all(self) -> None:
        n = len(self._spot_centers)
        self._spot_status_var.set(f"Spot {self._spot_idx + 1} / {n}" if n > 0 else "Spot 0 / 0")
        if n <= 0:
            self._current_file_var.set("File: -")
        else:
            idx_name = max(0, min(int(self._spot_idx), n - 1))
            extra = ""
            if self._spot_file_summaries and idx_name < len(self._spot_file_summaries):
                summary = str(self._spot_file_summaries[idx_name]).strip()
                if summary:
                    extra = f" | {summary}"
            if self._spot_names and idx_name < len(self._spot_names):
                self._current_file_var.set(f"File: {self._spot_names[idx_name]}{extra}")
            elif self.source_path is not None:
                self._current_file_var.set(f"File: {self.source_path.name}{extra}")
            else:
                self._current_file_var.set("File: -")

        self._ax_xy.clear()
        self._ax_shift.clear()
        self._ax_phi.clear()
        self._ax_hist.clear()
        self._ax_fft.clear()

        if n <= 0:
            self._ax_xy.set_title("X/Y scatter")
            self._ax_phi.set_title("Raw phi(t)")
            self._ax_shift.set_title("Circle fit")
            self._ax_hist.set_title("Raw phi distribution")
            self._ax_fft.set_title("X+iY Fourier transform")
            self._fig.tight_layout()
            self._canvas.draw_idle()
            self._render_sphere_tab()
            self._render_spherical_heatmap_tab()
            return

        idx = max(0, min(int(self._spot_idx), n - 1))
        xy = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        fps = float(self._current_fps())
        phi_raw = self._physical_phi_from_xy_series(xy, wrap_2pi=True)

        if xy.size > 0:
            self._ax_xy.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.8)
        self._ax_xy.axhline(0.0, color="0.8", lw=1)
        self._ax_xy.axvline(0.0, color="0.8", lw=1)
        self._ax_xy.set_xlim(-1.0, 1.0)
        self._ax_xy.set_ylim(-1.0, 1.0)
        self._ax_xy.set_xlabel("X")
        self._ax_xy.set_ylabel("Y")
        self._ax_xy.set_title("X/Y scatter")

        phi_plot_series = self._phi_series_after_time_average_subtraction(phi_raw, fps)
        if phi_plot_series.size > 0:
            t = np.arange(phi_plot_series.size, dtype=np.float64) / max(1e-9, fps)
            phi_plot = phi_plot_series.astype(np.float64, copy=True)
            if not bool(self._phi_residual_avg_var.get()) and phi_plot.size >= 2:
                d = np.abs(np.diff(phi_plot))
                jump_idx = np.where(d > np.pi)[0]
                if jump_idx.size > 0:
                    phi_plot[jump_idx + 1] = np.nan
            self._ax_phi.plot(t, phi_plot, lw=1.0)
            tmax = float(t[-1]) if t.size else 0.0
            self._draw_phi_selection(tmax)
        else:
            self._draw_phi_selection(0.0)
        self._ax_phi.set_xlabel("time (s)")
        if bool(self._phi_residual_avg_var.get()):
            self._ax_phi.set_ylabel("phi residual (rad)")
            self._ax_phi.set_title("phi(t) residual after subtracting time average")
        else:
            if bool(self._unwrap_phi_continuity_var.get()):
                self._ax_phi.set_ylabel("phi (rad, continuity, 0..2pi)")
                self._ax_phi.set_ylim(0.0, 2.0 * np.pi)
                self._ax_phi.set_title("Raw phi(t) from anisotropy X/Y")
            else:
                self._ax_phi.set_ylabel("phi (rad, direct 0..pi)")
                self._ax_phi.set_ylim(0.0, np.pi)
                self._ax_phi.set_title("Raw direct phi(t) from 0.5*atan2(Y,X)")

        bin_w = self._get_bin_deg(9.0)
        if bool(self._phi_residual_avg_var.get()):
            phi_res_deg = np.degrees(phi_plot_series)
            if phi_res_deg.size > 0:
                lim = max(bin_w, float(np.ceil(np.max(np.abs(phi_res_deg)) / bin_w) * bin_w))
                lim = max(lim, 30.0)
                bins_deg = np.arange(-lim, lim + bin_w + 1e-9, bin_w)
                self._ax_hist.hist(phi_res_deg, bins=bins_deg, color="tab:blue", alpha=0.85)
                self._ax_hist.set_xlim(-lim, lim)
            self._ax_hist.set_xlabel("raw phi residual (deg)")
            self._ax_hist.set_ylabel("count")
            self._ax_hist.set_title(f"Raw phi residual distribution ({bin_w:g} deg bins)")
        else:
            phi_max = 360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0
            phi_deg = np.mod(np.degrees(phi_raw), phi_max)
            bins_deg = np.arange(0.0, phi_max + 1e-9, bin_w)
            if phi_deg.size > 0:
                self._ax_hist.hist(phi_deg, bins=bins_deg, color="tab:blue", alpha=0.85)
            self._ax_hist.set_xlim(0.0, phi_max)
            self._ax_hist.set_xlabel("raw phi (deg)")
            self._ax_hist.set_ylabel("count")
            self._ax_hist.set_title(f"Raw phi distribution ({bin_w:g} deg bins)")

        fit_enabled = bool(self._fit_circle_sphere_var.get())
        if fit_enabled and self._fit_shifted_xy is not None and self._fit_shifted_phi is not None:
            sh = self._fit_shifted_xy
            self._ax_shift.scatter(sh[:, 0], sh[:, 1], s=4, alpha=0.8, label="shifted points")
            if self._fit_radius is not None and np.isfinite(self._fit_radius) and self._fit_radius > 0.0:
                th = np.linspace(0.0, 2.0 * np.pi, 400)
                self._ax_shift.plot(
                    self._fit_radius * np.cos(th),
                    self._fit_radius * np.sin(th),
                    color="tab:red",
                    lw=1.0,
                    label="fitted circle",
                )
            self._ax_shift.axhline(0.0, color="0.8", lw=1)
            self._ax_shift.axvline(0.0, color="0.8", lw=1)
            self._ax_shift.set_xlabel("X shifted")
            self._ax_shift.set_ylabel("Y shifted")
            self._ax_shift.set_title("Shifted XY after circle fit")
            self._ax_shift.set_aspect("equal", adjustable="box")
            self._ax_shift.legend(loc="best", fontsize=8)

            if self._fit_center is not None and self._fit_radius is not None:
                cx, cy = self._fit_center
                self._status_var.set(
                    f"Spot {idx + 1}: fitted circle center=({cx:.4f}, {cy:.4f}), radius={self._fit_radius:.4f}"
                )
        else:
            self._ax_shift.set_title("Circle fit disabled")
            self._ax_shift.text(
                0.5,
                0.5,
                "Tick 'Use fitted phi' and Apply fitted phi\nto compute shifted XY and fitted-axis phi.",
                ha="center",
                va="center",
                transform=self._ax_shift.transAxes,
            )

        if fit_enabled and self._fit_fft_freqs is not None and self._fit_fft_psd is not None and self._fit_fft_freqs.size > 0:
            freqs, psd = self._current_fft_positive_freqs_psd()
            if freqs.size > 0:
                self._ax_fft.plot(freqs, psd, color="tab:purple", lw=1.0)
                self._ax_fft.axvline(0.0, color="0.7", lw=1.0, linestyle=":")
                xmax = self._fft_max_hz(float(np.max(freqs)))
                if np.isfinite(xmax) and xmax > 0.0:
                    self._ax_fft.set_xlim(0.0, min(xmax, float(np.max(freqs))))
                self._ax_fft.set_xlabel("frequency (Hz)")
                self._ax_fft.set_ylabel("PSD of X+iY")
                self._ax_fft.set_title("X+iY Fourier transform")
                self._ax_fft.set_yscale("log")
                self._ax_fft.grid(alpha=0.22)
                marker_freq = self._fft_marker_freq_hz
                if marker_freq is None:
                    current_record = self._current_fft_peak_record()
                    if current_record is not None:
                        marker_freq = current_record.get("peak_freq_hz", None)
                if marker_freq is not None and np.isfinite(float(marker_freq)):
                    marker_freq = max(0.0, min(float(marker_freq), float(xmax)))
                    marker_psd = self._nearest_fft_psd(marker_freq)
                    self._ax_fft.axvline(marker_freq, color="tab:red", lw=1.8, linestyle="--", zorder=4)
                    if marker_psd is not None and np.isfinite(marker_psd) and marker_psd > 0.0:
                        self._ax_fft.scatter(
                            [marker_freq],
                            [marker_psd],
                            s=36,
                            color="tab:red",
                            edgecolors="white",
                            linewidths=0.6,
                            zorder=5,
                        )
                    self._ax_fft.text(
                        marker_freq,
                        0.98,
                        f"{marker_freq:.2f} Hz",
                        color="tab:red",
                        fontsize=8,
                        ha="center",
                        va="top",
                        transform=self._ax_fft.get_xaxis_transform(),
                    )
        else:
            self._ax_fft.set_title("X+iY Fourier transform")
            self._ax_fft.text(
                0.5,
                0.5,
                "FFT is computed only in fit mode.",
                ha="center",
                va="center",
                transform=self._ax_fft.transAxes,
            )

        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._render_sphere_tab()
        self._render_spherical_heatmap_tab()

    def _render_frequency_peaks_tab(self) -> None:
        if self._fft_peak_canvas is None:
            return
        self._fft_peak_ax_hist.clear()
        self._fft_peak_ax_scatter.clear()

        if not self._fft_peak_records:
            self._fft_peak_ax_hist.set_title("Fourier peak frequency distribution")
            self._fft_peak_ax_scatter.set_title("Peaks by added recording")
            self._fft_peak_ax_hist.text(
                0.5, 0.5, "Place marker in Spot View, then click 'Add Marker To Plot'.",
                ha="center", va="center", transform=self._fft_peak_ax_hist.transAxes,
            )
            self._fft_peak_ax_scatter.text(
                0.5, 0.5, "No added recordings yet.",
                ha="center", va="center", transform=self._fft_peak_ax_scatter.transAxes,
            )
            self._fft_peak_info_var.set("No Fourier peaks added yet.")
            self._fft_peak_fig.tight_layout()
            self._fft_peak_canvas.draw_idle()
            return

        xmax = 100.0
        all_freqs: list[float] = []
        labels: list[str] = []
        for rec in self._fft_peak_records:
            labels.append(str(rec.get("label", f"recording {len(labels) + 1}")))
            freq = rec.get("peak_freq_hz", None)
            if freq is not None and np.isfinite(float(freq)):
                all_freqs.append(float(freq))

        all_freqs_arr = np.asarray(all_freqs, dtype=np.float64)
        bins = np.arange(0.0, xmax + 1.0, 1.0)
        if all_freqs_arr.size > 0:
            self._fft_peak_ax_hist.hist(all_freqs_arr, bins=bins, color="tab:purple", alpha=0.85)
        self._fft_peak_ax_hist.set_xlim(0.0, xmax)
        self._fft_peak_ax_hist.set_xlabel("frequency (Hz)")
        self._fft_peak_ax_hist.set_ylabel("count")
        self._fft_peak_ax_hist.set_title("Fourier peak frequency distribution (0-100 Hz)")
        self._fft_peak_ax_hist.grid(alpha=0.22)

        y_positions = np.arange(1, len(self._fft_peak_records) + 1, dtype=np.float64)
        for y, rec in zip(y_positions, self._fft_peak_records):
            freq = rec.get("peak_freq_hz", None)
            if freq is not None and np.isfinite(float(freq)):
                self._fft_peak_ax_scatter.scatter(
                    [float(freq)],
                    [float(y)],
                    s=26,
                    color="tab:red",
                    alpha=0.9,
                )
        self._fft_peak_ax_scatter.set_xlim(0.0, xmax)
        self._fft_peak_ax_scatter.set_xlabel("frequency (Hz)")
        self._fft_peak_ax_scatter.set_ylabel("added recording")
        self._fft_peak_ax_scatter.set_title("Peaks by added recording")
        if len(labels) <= 16:
            self._fft_peak_ax_scatter.set_yticks(y_positions)
            self._fft_peak_ax_scatter.set_yticklabels(labels)
        self._fft_peak_ax_scatter.grid(alpha=0.22)

        total_peaks = int(all_freqs_arr.size)
        self._fft_peak_info_var.set(f"{len(self._fft_peak_records)} recording(s) added, {total_peaks} marker value(s).")
        self._fft_peak_fig.tight_layout()
        self._fft_peak_canvas.draw_idle()

    def _draw_unit_sphere_occupancy_heatmap(self, u: np.ndarray) -> None:
        ax = self._ax_sphere_heatmap
        if ax is None:
            return
        ax.clear()
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            ax.set_title("Unit-sphere occupancy heatmap")
            ax.text2D(0.5, 0.5, "No unit-sphere points", ha="center", va="center", transform=ax.transAxes)
            return

        z = np.clip(arr[:, 2], -1.0, 1.0)
        theta = np.arccos(z)
        phi = np.mod(np.arctan2(arr[:, 1], arr[:, 0]), 2.0 * np.pi)
        theta_bins = np.linspace(0.0, np.pi, 37)
        phi_bins = np.linspace(0.0, 2.0 * np.pi, 73)
        counts, _, _ = np.histogram2d(theta, phi, bins=(theta_bins, phi_bins))
        ti = np.clip(np.searchsorted(theta_bins, theta, side="right") - 1, 0, counts.shape[0] - 1)
        pi = np.clip(np.searchsorted(phi_bins, phi, side="right") - 1, 0, counts.shape[1] - 1)
        density = counts[ti, pi].astype(np.float64, copy=False)

        max_points = 25000
        if arr.shape[0] > max_points:
            step = int(math.ceil(float(arr.shape[0]) / float(max_points)))
            keep = np.arange(0, arr.shape[0], step, dtype=np.int64)
            arr_plot = arr[keep]
            density_plot = density[keep]
        else:
            arr_plot = arr
            density_plot = density
        order = np.argsort(density_plot)
        arr_plot = arr_plot[order]
        density_plot = density_plot[order]

        uu = np.linspace(0.0, 2.0 * np.pi, 28)
        vv = np.linspace(0.0, np.pi, 14)
        xs = np.outer(np.cos(uu), np.sin(vv))
        ys = np.outer(np.sin(uu), np.sin(vv))
        zs = np.outer(np.ones_like(uu), np.cos(vv))
        ax.plot_wireframe(xs, ys, zs, color="0.88", linewidth=0.35, alpha=0.45)
        ax.scatter(
            arr_plot[:, 0],
            arr_plot[:, 1],
            arr_plot[:, 2],
            c=density_plot,
            cmap="inferno",
            s=6,
            alpha=0.88,
            depthshade=False,
        )
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        max_count = float(np.max(density)) if density.size else 0.0
        ax.set_title(f"Unit-sphere occupancy heatmap ({arr.shape[0]} pts, max bin {max_count:.0f})")

    @staticmethod
    def _draw_spherical_grid(ax) -> None:
        uu = np.linspace(0.0, 2.0 * np.pi, 40)
        vv = np.linspace(0.0, np.pi, 20)
        xs = np.outer(np.cos(uu), np.sin(vv))
        ys = np.outer(np.sin(uu), np.sin(vv))
        zs = np.outer(np.ones_like(uu), np.cos(vv))
        ax.plot_wireframe(xs, ys, zs, color="0.84", linewidth=0.4, alpha=0.55)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    @staticmethod
    def _shortest_occupied_arc_coordinates(angle: np.ndarray) -> tuple[np.ndarray, float, float]:
        vals = np.mod(np.asarray(angle, dtype=np.float64), 2.0 * np.pi)
        valid = np.isfinite(vals)
        if int(np.count_nonzero(valid)) == 0:
            z = np.zeros_like(vals, dtype=np.float64)
            return z, 0.0, 0.0
        good_vals = vals[valid]
        if good_vals.size == 1:
            coords = np.zeros_like(vals, dtype=np.float64)
            coords[~valid] = np.nan
            return coords, float(good_vals[0]), float(good_vals[0])
        sorted_vals = np.sort(good_vals)
        gaps = np.diff(np.concatenate([sorted_vals, sorted_vals[:1] + (2.0 * np.pi)]))
        gap_i = int(np.argmax(gaps))
        start = float(sorted_vals[(gap_i + 1) % sorted_vals.size])
        coords = np.mod(vals - start, 2.0 * np.pi)
        coords[~valid] = np.nan
        span = float(np.nanmax(coords)) if np.any(np.isfinite(coords)) else 0.0
        return coords, start, start + span

    def _clear_arc_result(self, render: bool = False) -> None:
        self._arc_result = None
        self._arc_info_var.set("Press Analyse Arc to fit the selected unit-sphere window.")
        if render:
            self._render_arc_tab()

    def _fit_selected_origin_arc(self) -> Optional[dict]:
        fit = self._current_unit_sphere_distribution()
        if fit is None:
            return None
        u = np.asarray(fit.get("u", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 3 or u.shape[0] < 3:
            return None
        good = np.all(np.isfinite(u), axis=1)
        u = u[good]
        if u.shape[0] < 3:
            return None
        norm = np.linalg.norm(u, axis=1)
        keep = norm > 1e-12
        u = u[keep] / norm[keep][:, None]
        if u.shape[0] < 3:
            return None

        # Best great-circle plane through the origin: normal is smallest-eigenvalue vector.
        m = (u.T @ u) / float(u.shape[0])
        vals, vecs = np.linalg.eigh(m)
        normal = vecs[:, int(np.argmin(vals))]
        normal = normal / max(1e-12, float(np.linalg.norm(normal)))

        first = u[0] - (float(np.dot(u[0], normal)) * normal)
        if float(np.linalg.norm(first)) <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, normal))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            first = ref - (float(np.dot(ref, normal)) * normal)
        e1 = first / max(1e-12, float(np.linalg.norm(first)))
        e2 = np.cross(normal, e1)
        e2 = e2 / max(1e-12, float(np.linalg.norm(e2)))

        p1 = u @ e1
        p2 = u @ e2
        raw_angle = np.arctan2(p2, p1)
        if bool(self._unwrap_phi_continuity_var.get()):
            angle = np.unwrap(raw_angle)
            opening_deg = np.degrees(angle - float(angle[0]))
            arc_min = float(np.min(angle))
            arc_max = float(np.max(angle))
        else:
            angle, arc_min, arc_max = self._shortest_occupied_arc_coordinates(raw_angle)
            opening_deg = np.degrees(angle)
        plane_dist = np.clip(np.abs(u @ normal), 0.0, 1.0)
        residual_deg = np.degrees(np.arcsin(plane_dist))
        fs = max(1e-9, float(self._current_fps()))
        t0_abs = float(max(0, int(self._sphere_fit_i0))) / fs
        t = t0_abs + (np.arange(u.shape[0], dtype=np.float64) / fs)
        th = np.linspace(arc_min, arc_max, 361)
        arc_xyz = (np.cos(th)[:, None] * e1[None, :]) + (np.sin(th)[:, None] * e2[None, :])
        return {
            "u": u,
            "normal": normal,
            "e1": e1,
            "e2": e2,
            "arc_xyz": arc_xyz,
            "opening_deg": opening_deg.astype(np.float64, copy=False),
            "time_s": t.astype(np.float64, copy=False),
            "residual_deg": residual_deg.astype(np.float64, copy=False),
            "source_mode": str(fit.get("mode", "raw")),
            "theta_model": str(fit.get("theta_model", self._theta_model_var.get() or "hole+fresnel")),
        }

    def _on_analyse_arc(self) -> None:
        res = self._fit_selected_origin_arc()
        if res is None:
            messagebox.showerror("Arc analysis", "Need at least 3 selected unit-sphere points to fit an origin-centred arc.")
            self._clear_arc_result(render=True)
            return
        self._arc_result = res
        self._render_arc_tab()

    def _render_arc_tab(self) -> None:
        if self._arc_canvas is None:
            return
        self._ax_arc_3d.clear()
        self._ax_arc_angle.clear()
        self._ax_arc_hist.clear()

        res = self._arc_result
        if res is None:
            self._draw_spherical_grid(self._ax_arc_3d)
            self._ax_arc_3d.set_title("Origin-centred arc fit")
            self._ax_arc_3d.text2D(
                0.5,
                0.5,
                "Press Analyse Arc",
                ha="center",
                va="center",
                transform=self._ax_arc_3d.transAxes,
            )
            self._ax_arc_angle.set_title("Opening angle vs time")
            self._ax_arc_angle.text(0.5, 0.5, "No arc fit yet", ha="center", va="center", transform=self._ax_arc_angle.transAxes)
            self._ax_arc_hist.set_title("Opening angle distribution")
            self._ax_arc_hist.text(0.5, 0.5, "No arc fit yet", ha="center", va="center", transform=self._ax_arc_hist.transAxes)
            self._arc_info_var.set("Press Analyse Arc to fit the selected unit-sphere window.")
            self._arc_fig.tight_layout()
            self._arc_canvas.draw_idle()
            return

        u = np.asarray(res["u"], dtype=np.float64)
        arc = np.asarray(res["arc_xyz"], dtype=np.float64)
        opening = np.asarray(res["opening_deg"], dtype=np.float64)
        t = np.asarray(res["time_s"], dtype=np.float64)
        residual = np.asarray(res["residual_deg"], dtype=np.float64)

        self._draw_spherical_grid(self._ax_arc_3d)
        c = np.arange(u.shape[0], dtype=np.float64)
        self._ax_arc_3d.scatter(u[:, 0], u[:, 1], u[:, 2], c=c, cmap="viridis", s=10, alpha=0.9, depthshade=False)
        self._ax_arc_3d.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="tab:red", lw=2.0, label="fitted origin arc")
        nvec = np.asarray(res["normal"], dtype=np.float64)
        self._ax_arc_3d.plot([-nvec[0], nvec[0]], [-nvec[1], nvec[1]], [-nvec[2], nvec[2]], color="black", lw=1.5, linestyle="--", label="plane normal")
        self._ax_arc_3d.set_title("Raw points on spherical grid")
        self._ax_arc_3d.legend(loc="upper left", fontsize=8)

        self._ax_arc_angle.plot(t, opening, color="tab:blue", lw=1.0)
        self._ax_arc_angle.set_xlabel("time (s)")
        self._ax_arc_angle.set_ylabel("opening angle (deg)")
        self._ax_arc_angle.set_title("Opening angle along fitted origin-centred arc")
        self._ax_arc_angle.grid(alpha=0.25)

        bin_w = self._get_bin_deg(9.0)
        if opening.size > 0:
            lo = float(np.floor(float(np.min(opening)) / bin_w) * bin_w)
            hi = float(np.ceil(float(np.max(opening)) / bin_w) * bin_w)
            if hi <= lo:
                hi = lo + bin_w
            bins = np.arange(lo, hi + bin_w + 1e-9, bin_w)
            self._ax_arc_hist.hist(opening, bins=bins, color="tab:blue", alpha=0.85)
        self._ax_arc_hist.set_xlabel("opening angle (deg)")
        self._ax_arc_hist.set_ylabel("count")
        self._ax_arc_hist.set_title("Opening angle distribution")

        open_txt = self._percentile_range_text(opening, "opening", " deg")
        resid_med = float(np.nanmedian(residual)) if residual.size else float("nan")
        resid_p95 = float(np.nanpercentile(residual, 95.0)) if residual.size else float("nan")
        self._arc_info_var.set(
            f"Arc fit | source={res.get('source_mode', 'raw')} | points={u.shape[0]} | {open_txt} | residual median={resid_med:.3g} deg p95={resid_p95:.3g} deg"
        )
        self._arc_fig.tight_layout()
        self._arc_canvas.draw_idle()

    def _render_spherical_heatmap_tab(self) -> None:
        if self._spherical_heatmap_canvas is None:
            return
        ax = self._ax_spherical_heatmap
        ax.clear()
        fit = self._current_unit_sphere_distribution()
        if fit is None:
            ax.set_title("Spherical-coordinate heatmap")
            ax.text(0.5, 0.5, "Load data first", ha="center", va="center", transform=ax.transAxes)
            self._heatmap_info_var.set("Spherical-coordinate density for the current selected window.")
            self._spherical_heatmap_fig.tight_layout()
            self._spherical_heatmap_canvas.draw_idle()
            return

        theta_deg = np.degrees(np.asarray(fit.get("theta_rad", np.zeros((0,), dtype=np.float64)), dtype=np.float64))
        phi_rad = np.asarray(fit.get("phi_input_rad", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        if bool(self._unwrap_phi_continuity_var.get()):
            phi_deg = np.mod(np.degrees(phi_rad), 360.0)
            phi_max = 360.0
        else:
            phi_deg = np.mod(np.degrees(phi_rad), 180.0)
            phi_max = 180.0
        valid = np.isfinite(theta_deg) & np.isfinite(phi_deg)
        theta_deg = theta_deg[valid]
        phi_deg = phi_deg[valid]
        if theta_deg.size == 0:
            ax.set_title("Spherical-coordinate heatmap")
            ax.text(0.5, 0.5, "No valid theta/phi points", ha="center", va="center", transform=ax.transAxes)
            self._heatmap_info_var.set("No valid spherical-coordinate points in current selected window.")
            self._spherical_heatmap_fig.tight_layout()
            self._spherical_heatmap_canvas.draw_idle()
            return

        phi_bins = np.linspace(0.0, phi_max, 73 if phi_max > 180.0 else 37)
        theta_bins = np.linspace(0.0, 90.0, 37)
        counts, theta_edges, phi_edges = np.histogram2d(theta_deg, phi_deg, bins=(theta_bins, phi_bins))
        im = ax.imshow(
            counts,
            origin="lower",
            aspect="auto",
            extent=[float(phi_edges[0]), float(phi_edges[-1]), float(theta_edges[0]), float(theta_edges[-1])],
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("theta (deg)")
        mode = "continuity-unwrapped" if bool(self._unwrap_phi_continuity_var.get()) else "direct folded 0..180"
        ax.set_title(f"Spherical-coordinate density ({mode})")
        self._heatmap_info_var.set(
            f"Spherical heatmap | points={theta_deg.size} | phi range 0-{phi_max:.0f} deg | theta model={fit.get('theta_model', '-')}"
        )
        self._spherical_heatmap_fig.tight_layout()
        self._spherical_heatmap_canvas.draw_idle()

    def _render_sphere_tab(self) -> None:
        self._ax_sphere_3d.clear()
        if self._ax_sphere_heatmap is not None:
            self._ax_sphere_heatmap.clear()
        self._ax_sphere_plane.clear()
        self._ax_sphere_phi.clear()
        self._ax_sphere_theta.clear()
        self._ax_sphere_hist.clear()
        self._sphere_trail_artist = None
        self._sphere_head_artist = None
        self._sphere_plane_trail_artist = None
        self._sphere_plane_head_artist = None
        self._phi_sel_line0_sphere = None
        self._phi_sel_line1_sphere = None
        self._phi_sel_span_sphere = None

        fit = self._current_unit_sphere_distribution()
        if fit is None:
            prompt = "Load data to reconstruct unit-sphere trajectory"
            self._ax_sphere_3d.set_title("Unit-sphere trajectory")
            if self._ax_sphere_heatmap is not None:
                self._ax_sphere_heatmap.set_title("Unit-sphere occupancy heatmap")
                self._ax_sphere_heatmap.text2D(
                    0.5,
                    0.5,
                    prompt,
                    ha="center",
                    va="center",
                    transform=self._ax_sphere_heatmap.transAxes,
                )
            self._ax_sphere_plane.set_title("Projection in fitted plane")
            self._ax_sphere_phi.set_title("phi_fromaxis(t)")
            self._ax_sphere_theta.set_title("theta(t)")
            self._ax_sphere_hist.set_title("phi_fromaxis distribution")
            self._ax_sphere_3d.text2D(0.5, 0.5, prompt, ha="center", va="center", transform=self._ax_sphere_3d.transAxes)
            self._ax_sphere_plane.text(
                0.5, 0.5, prompt, ha="center", va="center", transform=self._ax_sphere_plane.transAxes
            )
            self._ax_sphere_phi.text(
                0.5, 0.5, prompt, ha="center", va="center", transform=self._ax_sphere_phi.transAxes
            )
            self._ax_sphere_theta.text(
                0.5, 0.5, prompt, ha="center", va="center", transform=self._ax_sphere_theta.transAxes
            )
            self._ax_sphere_hist.text(
                0.5, 0.5, prompt, ha="center", va="center", transform=self._ax_sphere_hist.transAxes
            )
            self._sphere_info_var.set(f"{prompt} to reconstruct unit-sphere trajectory.")
            self._sphere_fig.tight_layout()
            self._sphere_canvas.draw_idle()
            return

        u = np.asarray(fit.get("u", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 3 or u.shape[0] == 0:
            self._sphere_info_var.set("No unit-sphere points for current selection.")
            self._sphere_fig.tight_layout()
            self._sphere_canvas.draw_idle()
            return

        if str(fit.get("mode", "fitted")) == "raw":
            uu = np.linspace(0.0, 2.0 * np.pi, 40)
            vv = np.linspace(0.0, np.pi, 20)
            xs = np.outer(np.cos(uu), np.sin(vv))
            ys = np.outer(np.sin(uu), np.sin(vv))
            zs = np.outer(np.ones_like(uu), np.cos(vv))
            self._ax_sphere_3d.plot_wireframe(xs, ys, zs, color="0.82", linewidth=0.4, alpha=0.6)
            self._sphere_trail_artist = self._ax_sphere_3d.scatter([], [], [], s=16, color="tab:blue", alpha=0.85)
            (self._sphere_head_artist,) = self._ax_sphere_3d.plot([], [], [], marker="o", markersize=5, color="black", linestyle="")
            self._ax_sphere_3d.set_box_aspect((1, 1, 1))
            self._ax_sphere_3d.set_xlim(-1.05, 1.05)
            self._ax_sphere_3d.set_ylim(-1.05, 1.05)
            self._ax_sphere_3d.set_zlim(-1.05, 1.05)
            self._ax_sphere_3d.set_xlabel("x")
            self._ax_sphere_3d.set_ylabel("y")
            self._ax_sphere_3d.set_zlabel("z")
            self._ax_sphere_3d.set_title("Raw Rod Path On Unit Sphere")

            self._draw_unit_sphere_occupancy_heatmap(u)

            p1 = np.asarray(fit["plane_p1"], dtype=np.float64)
            p2 = np.asarray(fit["plane_p2"], dtype=np.float64)
            self._ax_sphere_plane.plot(p1, p2, color="0.85", lw=0.8)
            self._sphere_plane_trail_artist = self._ax_sphere_plane.scatter([], [], s=16, color="tab:blue", alpha=0.85)
            (self._sphere_plane_head_artist,) = self._ax_sphere_plane.plot([], [], marker="o", markersize=5, color="black", linestyle="")
            self._ax_sphere_plane.axhline(0.0, color="0.85", lw=1)
            self._ax_sphere_plane.axvline(0.0, color="0.85", lw=1)
            self._ax_sphere_plane.set_xlim(-1.0, 1.0)
            self._ax_sphere_plane.set_ylim(-1.0, 1.0)
            self._ax_sphere_plane.set_aspect("equal", adjustable="box")
            self._ax_sphere_plane.set_xlabel("raw X")
            self._ax_sphere_plane.set_ylabel("raw Y")
            self._ax_sphere_plane.set_title("Raw Anisotropy Plane")

            phi_unwrapped_deg = np.degrees(np.asarray(fit["phi_axis_unwrapped_rad"], dtype=np.float64))
            fs = max(1e-9, float(self._current_fps()))
            t0_abs = float(max(0, int(self._sphere_fit_i0))) / fs
            t = t0_abs + (np.arange(u.shape[0], dtype=np.float64) / fs)
            self._ax_sphere_phi.plot(t, phi_unwrapped_deg, color="tab:green", lw=1.0)
            full_tmax = 0.0
            if self._spot_xy_series:
                idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
                n_full = int(np.asarray(self._spot_xy_series[idx], dtype=np.float64).shape[0])
                if n_full > 0:
                    full_tmax = float(max(0, n_full - 1)) / fs
            self._draw_sphere_phi_selection(full_tmax)
            self._ax_sphere_phi.set_xlabel("time (s)")
            self._ax_sphere_phi.set_ylabel("raw phi (deg, unwrapped)")
            self._ax_sphere_phi.set_title("Raw phi(t) on unit sphere")
            self._ax_sphere_phi.grid(alpha=0.25)

            theta_deg = np.degrees(np.asarray(fit["theta_rad"], dtype=np.float64))
            self._ax_sphere_theta.plot(t, theta_deg, color="tab:red", lw=1.0)
            self._ax_sphere_theta.set_xlabel("time (s)")
            self._ax_sphere_theta.set_ylabel("theta (deg)")
            self._ax_sphere_theta.set_title("theta(t) from raw r")
            self._ax_sphere_theta.grid(alpha=0.25)

            phi_deg = np.asarray(fit["phi_axis_rel_deg"], dtype=np.float64)
            hist_edges = np.asarray(fit["phi_axis_edges_deg"], dtype=np.float64)
            self._ax_sphere_hist.hist(phi_deg, bins=hist_edges, color="tab:green", alpha=0.85)
            phi_max = float(hist_edges[-1]) if hist_edges.size else (360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0)
            self._ax_sphere_hist.set_xlim(0.0, phi_max)
            self._ax_sphere_hist.set_xlabel("raw phi (deg)")
            self._ax_sphere_hist.set_ylabel("count")
            self._ax_sphere_hist.set_title("Raw phi Distribution")

            self._sphere_anim_idx = max(0, min(int(self._sphere_anim_idx), int(u.shape[0]) - 1))
            self._update_sphere_animation_artists()
            clipped = int(fit["r_clipped_count"])
            range_txt = self._r_theta_percentile_summary(fit)
            self._sphere_info_var.set(
                f"Raw unit sphere | Theta model={fit['theta_model']} | points={u.shape[0]} | clipped-r={clipped} | {range_txt} | tick Use fitted phi for axis/circle fit"
            )
            self._sphere_fig.tight_layout()
            self._sphere_canvas.draw_idle()
            if self._sphere_anim_running:
                self._schedule_sphere_animation_tick()
            return

        axis_k = np.asarray(fit["axis_k"], dtype=np.float64)
        axis_center = np.asarray(fit["axis_center"], dtype=np.float64)
        circle_xyz = np.asarray(fit["circle_xyz"], dtype=np.float64)
        axis2_k = fit.get("axis2_k", None)
        axis2_center = fit.get("axis2_center", None)
        circle2_xyz = fit.get("circle2_xyz", None)
        e1 = np.asarray(fit["e1"], dtype=np.float64)
        e2 = np.asarray(fit["e2"], dtype=np.float64)
        p1 = np.asarray(fit["plane_p1"], dtype=np.float64)
        p2 = np.asarray(fit["plane_p2"], dtype=np.float64)
        phi_unwrapped_deg = np.degrees(np.asarray(fit["phi_axis_unwrapped_rad"], dtype=np.float64))
        phi_rel_deg = np.asarray(fit["phi_axis_rel_deg"], dtype=np.float64)
        hist_edges = np.asarray(fit["phi_axis_edges_deg"], dtype=np.float64)

        # Unit sphere + fitted small circle/axis.
        uu = np.linspace(0.0, 2.0 * np.pi, 40)
        vv = np.linspace(0.0, np.pi, 20)
        xs = np.outer(np.cos(uu), np.sin(vv))
        ys = np.outer(np.sin(uu), np.sin(vv))
        zs = np.outer(np.ones_like(uu), np.cos(vv))
        self._ax_sphere_3d.plot_wireframe(xs, ys, zs, color="0.82", linewidth=0.4, alpha=0.6)
        self._ax_sphere_3d.plot(
            [-axis_k[0], axis_k[0]],
            [-axis_k[1], axis_k[1]],
            [-axis_k[2], axis_k[2]],
            color="tab:red",
            lw=2.0,
            label="axis",
        )
        self._ax_sphere_3d.plot(
            circle_xyz[:, 0],
            circle_xyz[:, 1],
            circle_xyz[:, 2],
            color="tab:orange",
            lw=1.6,
            label="circle 1",
        )
        self._ax_sphere_3d.scatter(
            [axis_center[0]],
            [axis_center[1]],
            [axis_center[2]],
            s=28,
            color="tab:red",
            marker="x",
            label="center 1",
        )
        if axis2_k is not None and circle2_xyz is not None and axis2_center is not None:
            axis2_k = np.asarray(axis2_k, dtype=np.float64)
            axis2_center = np.asarray(axis2_center, dtype=np.float64)
            circle2_xyz = np.asarray(circle2_xyz, dtype=np.float64)
            self._ax_sphere_3d.plot(
                [-axis2_k[0], axis2_k[0]],
                [-axis2_k[1], axis2_k[1]],
                [-axis2_k[2], axis2_k[2]],
                color="tab:purple",
                lw=2.0,
                label="axis 2",
            )
            self._ax_sphere_3d.plot(
                circle2_xyz[:, 0],
                circle2_xyz[:, 1],
                circle2_xyz[:, 2],
                color="tab:cyan",
                lw=1.4,
                label="circle 2",
            )
            self._ax_sphere_3d.scatter(
                [axis2_center[0]],
                [axis2_center[1]],
                [axis2_center[2]],
                s=24,
                color="tab:purple",
                marker="x",
                label="center 2",
            )
        self._sphere_trail_artist = self._ax_sphere_3d.scatter([], [], [], s=16, color="tab:blue", alpha=0.85)
        (self._sphere_head_artist,) = self._ax_sphere_3d.plot([], [], [], marker="o", markersize=5, color="black", linestyle="")
        self._ax_sphere_3d.set_box_aspect((1, 1, 1))
        self._ax_sphere_3d.set_xlim(-1.05, 1.05)
        self._ax_sphere_3d.set_ylim(-1.05, 1.05)
        self._ax_sphere_3d.set_zlim(-1.05, 1.05)
        self._ax_sphere_3d.set_xlabel("x")
        self._ax_sphere_3d.set_ylabel("y")
        self._ax_sphere_3d.set_zlabel("z")
        self._ax_sphere_3d.set_title("Rod Path On Unit Sphere")
        self._ax_sphere_3d.legend(loc="upper left", fontsize=8)

        self._draw_unit_sphere_occupancy_heatmap(u)

        # Projection in plane perpendicular to fitted axis.
        dual_used = bool(fit.get("axis2_k", None) is not None)
        if dual_used:
            self._ax_sphere_plane.set_title("Fitted Circle Plane")
            self._ax_sphere_plane.text(
                0.5,
                0.5,
                "Projection disabled in dual-axis mode\nfor faster animation.",
                ha="center",
                va="center",
                transform=self._ax_sphere_plane.transAxes,
            )
            self._ax_sphere_plane.set_xticks([])
            self._ax_sphere_plane.set_yticks([])
        else:
            self._ax_sphere_plane.plot(p1, p2, color="0.85", lw=0.8)
            rr = np.sqrt((p1 * p1) + (p2 * p2))
            r0 = float(np.median(rr)) if rr.size else 0.0
            th = np.linspace(0.0, 2.0 * np.pi, 361)
            self._ax_sphere_plane.plot(r0 * np.cos(th), r0 * np.sin(th), color="tab:orange", lw=1.2, label="median radius")
            self._sphere_plane_trail_artist = self._ax_sphere_plane.scatter([], [], s=16, color="tab:blue", alpha=0.85)
            (self._sphere_plane_head_artist,) = self._ax_sphere_plane.plot([], [], marker="o", markersize=5, color="black", linestyle="")
            self._ax_sphere_plane.axhline(0.0, color="0.85", lw=1)
            self._ax_sphere_plane.axvline(0.0, color="0.85", lw=1)
            self._ax_sphere_plane.set_aspect("equal", adjustable="box")
            self._ax_sphere_plane.set_xlabel("component along e1")
            self._ax_sphere_plane.set_ylabel("component along e2")
            self._ax_sphere_plane.set_title("Fitted Circle Plane")
            self._ax_sphere_plane.legend(loc="best", fontsize=8)

        # phi_fromaxis trace in absolute spot time coordinates.
        fs = max(1e-9, float(self._current_fps()))
        t0_abs = float(max(0, int(self._sphere_fit_i0))) / fs
        t = t0_abs + (np.arange(u.shape[0], dtype=np.float64) / fs)
        self._ax_sphere_phi.plot(t, phi_unwrapped_deg, color="tab:green", lw=1.0)
        full_tmax = 0.0
        if self._spot_xy_series:
            idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
            n_full = int(np.asarray(self._spot_xy_series[idx], dtype=np.float64).shape[0])
            if n_full > 0:
                full_tmax = float(max(0, n_full - 1)) / fs
        self._draw_sphere_phi_selection(full_tmax)
        self._ax_sphere_phi.set_xlabel("time (s)")
        self._ax_sphere_phi.set_ylabel("phi_fromaxis (deg, unwrapped)")
        self._ax_sphere_phi.set_title("phi_fromaxis(t)")
        self._ax_sphere_phi.grid(alpha=0.25)

        theta_deg = np.degrees(np.asarray(fit["theta_rad"], dtype=np.float64))
        self._ax_sphere_theta.plot(t, theta_deg, color="tab:red", lw=1.0)
        self._ax_sphere_theta.set_xlabel("time (s)")
        self._ax_sphere_theta.set_ylabel("theta (deg)")
        self._ax_sphere_theta.set_title("theta(t) from raw r")
        self._ax_sphere_theta.grid(alpha=0.25)

        # phi_fromaxis histogram aligned to densest bin window.
        self._ax_sphere_hist.hist(phi_rel_deg, bins=hist_edges, color="tab:green", alpha=0.85)
        counts, edges = np.histogram(phi_rel_deg, bins=hist_edges)
        if counts.size > 0 and int(np.max(counts)) > 0:
            i_peak = int(np.argmax(counts))
            peak_center = 0.5 * (float(edges[i_peak]) + float(edges[i_peak + 1]))
            self._ax_sphere_hist.axvline(peak_center, color="black", linestyle=":", linewidth=1.4, alpha=0.95)
            guide_step = float(self._phi_guide_step_deg())
            guide_n = int(self._phi_guide_count())
            guide_style = str(self._phi_guide_linestyle())
            guide_mod = float(hist_edges[-1]) if hist_edges.size else 360.0
            guide_angles = sorted({(peak_center + guide_step * k) % guide_mod for k in range(guide_n)})
            for ang in guide_angles:
                self._ax_sphere_hist.axvline(
                    ang,
                    color="0.35",
                    linestyle=guide_style,
                    linewidth=1.0,
                    alpha=0.8,
                )
        phi_max = float(hist_edges[-1]) if hist_edges.size else (360.0 if bool(self._unwrap_phi_continuity_var.get()) else 180.0)
        self._ax_sphere_hist.set_xlim(0.0, phi_max)
        self._ax_sphere_hist.set_xlabel("phi_fromaxis (deg, aligned)")
        self._ax_sphere_hist.set_ylabel("count")
        self._ax_sphere_hist.set_title("phi_fromaxis Distribution")

        # Keep animation index in range and repaint active window.
        self._sphere_anim_idx = max(0, min(int(self._sphere_anim_idx), int(u.shape[0]) - 1))
        self._update_sphere_animation_artists()

        alpha_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(axis_k[2])))))
        beta_deg = math.degrees(math.atan2(float(axis_k[1]), float(axis_k[0]))) % 360.0
        gamma_deg = math.degrees(float(fit["gamma_rad"]))
        clipped = int(fit["r_clipped_count"])
        fit_cells = int(fit.get("fit_cells_used", 0))
        fit_pts = int(fit.get("fit_points_total", 0))
        switches = int(fit.get("switch_count", 0))
        fit_cells2 = int(fit.get("fit_cells2_used", 0))
        fit_pts2 = int(fit.get("fit_points2_total", 0))
        axis_angle = fit.get("axis_angle_deg", None)
        range_txt = self._r_theta_percentile_summary(fit)
        info = (
            f"Theta model={fit['theta_model']} | alpha={alpha_deg:.2f} deg beta={beta_deg:.2f} deg "
            f"gamma={gamma_deg:.2f} deg | clipped-r={clipped} | fit1 bins={fit_cells}/{fit_pts} | {range_txt}"
        )
        if dual_used:
            info += (
                f"\nfit2 bins={fit_cells2}/{fit_pts2}"
                + (f" | axis angle={float(axis_angle):.2f} deg" if axis_angle is not None else "")
                + f" | switches={switches}"
            )
        self._sphere_info_var.set(info)
        self._sphere_fig.tight_layout()
        self._sphere_canvas.draw_idle()
        if self._sphere_anim_running:
            self._schedule_sphere_animation_tick()

    def _update_sphere_animation_artists(self) -> None:
        fit = self._sphere_fit
        if fit is None:
            return
        if self._sphere_trail_artist is None or self._sphere_head_artist is None:
            return

        u = np.asarray(fit.get("u", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        p1 = np.asarray(fit.get("plane_p1", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        p2 = np.asarray(fit.get("plane_p2", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        n = int(u.shape[0])
        if n <= 0:
            return
        idx = max(0, min(int(self._sphere_anim_idx), n - 1))
        trail = self._get_sphere_trail_len(50)
        i0 = max(0, idx - trail + 1)
        seg = u[i0 : idx + 1]
        seg_p = np.column_stack((p1[i0 : idx + 1], p2[i0 : idx + 1]))

        self._sphere_trail_artist._offsets3d = (seg[:, 0], seg[:, 1], seg[:, 2])
        self._sphere_head_artist.set_data_3d([u[idx, 0]], [u[idx, 1]], [u[idx, 2]])
        if self._sphere_plane_trail_artist is not None and self._sphere_plane_head_artist is not None:
            self._sphere_plane_trail_artist.set_offsets(seg_p if seg_p.size else np.zeros((0, 2), dtype=np.float64))
            self._sphere_plane_head_artist.set_data([p1[idx]], [p2[idx]])
        self._sphere_canvas.draw_idle()

    def _sync_include_checkbox(self) -> None:
        if self._include_chk is None and self._include_speed_theta_chk is None:
            return
        n = len(self._spot_centers)
        idx = int(self._spot_idx) if n > 0 else -1
        val = False
        if 0 <= idx < len(self._avg_selected_spots):
            val = bool(self._avg_selected_spots[idx])
        self._syncing_include_var = True
        try:
            self._include_current_var.set(val)
        finally:
            self._syncing_include_var = False
        st_val = False
        if 0 <= idx < len(self._speed_theta_selected_spots):
            st_val = bool(self._speed_theta_selected_spots[idx])
        self._syncing_speed_theta_var = True
        try:
            self._include_speed_theta_var.set(st_val)
        finally:
            self._syncing_speed_theta_var = False

    def _on_toggle_include_current(self) -> None:
        if self._syncing_include_var:
            return
        n = len(self._spot_centers)
        if n <= 0:
            return
        if len(self._avg_selected_spots) != n:
            self._avg_selected_spots = [False for _ in range(n)]
        idx = max(0, min(int(self._spot_idx), n - 1))
        self._avg_selected_spots[idx] = bool(self._include_current_var.get())
        self._render_average_distribution()

    def _on_toggle_include_speed_theta_current(self) -> None:
        if self._syncing_speed_theta_var:
            return
        n = len(self._spot_centers)
        if n <= 0:
            return
        if len(self._speed_theta_selected_spots) != n:
            self._speed_theta_selected_spots = [False for _ in range(n)]
        idx = max(0, min(int(self._spot_idx), n - 1))
        keep = bool(self._include_speed_theta_var.get())
        self._speed_theta_selected_spots[idx] = keep
        if keep:
            ok = self._upsert_speed_theta_point_for_spot(idx, show_error=True)
            if not ok:
                self._speed_theta_selected_spots[idx] = False
                self._sync_include_checkbox()
        else:
            before = len(self._speed_theta_points)
            self._speed_theta_points = [p for p in self._speed_theta_points if int(p.get("spot_idx", -1)) != idx]
            removed = before - len(self._speed_theta_points)
            self._status_var.set(f"Removed {removed} phi-span point(s) for spot {idx + 1}.")
        self._render_speed_theta_tab()

    def _render_average_distribution(self) -> None:
        if self._avg_canvas is None:
            return
        self._avg_ax.clear()
        n = len(self._spot_xy_series)
        if n <= 0 or not self._avg_selected_spots:
            self._avg_ax.set_title("Average aligned distribution")
            self._avg_ax.text(0.5, 0.5, "No selected spots.", ha="center", va="center", transform=self._avg_ax.transAxes)
            self._avg_info_var.set("No selected spots yet.")
            self._avg_fig.tight_layout()
            self._avg_canvas.draw_idle()
            return

        selected_idxs = [i for i, flag in enumerate(self._avg_selected_spots[:n]) if flag]
        if not selected_idxs:
            self._avg_ax.set_title("Average aligned distribution")
            self._avg_ax.text(0.5, 0.5, "Tick spots to include them.", ha="center", va="center", transform=self._avg_ax.transAxes)
            self._avg_info_var.set("No spots ticked.")
            self._avg_fig.tight_layout()
            self._avg_canvas.draw_idle()
            return

        all_probs = []
        used = 0
        bin_w = self._get_bin_deg(float(self.AVG_BIN_DEG))
        avg_edges = np.arange(0.0, 360.0 + 1e-9, bin_w)
        for idx in selected_idxs:
            arr = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
            fit = self._fit_and_aligned_distribution(arr)
            if fit is None:
                continue
            phi_rel = np.asarray(fit["phi_rel_deg"], dtype=np.float64)
            counts, _ = np.histogram(phi_rel, bins=avg_edges)
            counts = counts.astype(np.float64, copy=False)
            s = float(np.sum(counts))
            if s <= 0.0:
                continue
            all_probs.append(counts / s)
            used += 1

        if not all_probs:
            self._avg_ax.set_title("Average aligned distribution")
            self._avg_ax.text(0.5, 0.5, "Selected spots had no valid distributions.", ha="center", va="center", transform=self._avg_ax.transAxes)
            self._avg_info_var.set(f"Selected={len(selected_idxs)}  Used=0")
            self._avg_fig.tight_layout()
            self._avg_canvas.draw_idle()
            return

        P = np.vstack(all_probs)
        mean_p = np.mean(P, axis=0)
        centers = avg_edges[:-1] + (0.5 * bin_w)
        self._avg_ax.bar(centers, mean_p, width=bin_w * 0.94, align="center", color="tab:orange", alpha=0.9)
        self._avg_ax.set_xlim(0.0, 360.0)
        self._avg_ax.set_xlabel("aligned phi (deg)")
        self._avg_ax.set_ylabel("mean probability")
        self._avg_ax.set_title(f"Average aligned distribution (checked spots, {bin_w:.0f} deg bins)")
        self._avg_info_var.set(f"Selected={len(selected_idxs)}  Used={used}  Bin={bin_w:.0f} deg")
        self._avg_fig.tight_layout()
        self._avg_canvas.draw_idle()

    def _compute_brownian_metrics(self, phi_unwrapped: np.ndarray, fs: float) -> Optional[dict]:
        phi = np.asarray(phi_unwrapped, dtype=np.float64)
        if phi.ndim != 1 or phi.size < 20:
            return None

        nphi = int(phi.size)
        max_lag = int(max(5, min(nphi - 1, int(round(0.9 * (nphi - 1))))))
        if max_lag < 5:
            return None
        if max_lag <= 2500:
            # Use all lags when feasible.
            lags = np.arange(1, max_lag + 1, dtype=np.int32)
        else:
            # For long traces, keep a manageable count but bias density toward long lags.
            m = 2200
            u = np.linspace(0.0, 1.0, m)
            frac = 1.0 - np.power(1.0 - u, 3.0)  # denser near long-lag end
            core = 1 + np.round(frac * float(max_lag - 1)).astype(np.int32)
            head = np.arange(1, min(140, max_lag) + 1, dtype=np.int32)
            tail = np.arange(max(1, max_lag - 320), max_lag + 1, dtype=np.int32)
            lags = np.unique(np.concatenate([head, core, tail]).astype(np.int32))

        tau = lags.astype(np.float64) / max(1e-9, float(fs))
        msd = np.empty((lags.size,), dtype=np.float64)
        for i, k in enumerate(lags):
            delta = phi[k:] - phi[:-k]
            msd[i] = float(np.mean(delta * delta)) if delta.size else np.nan

        valid = np.isfinite(msd) & np.isfinite(tau) & (msd > 0.0) & (tau > 0.0)
        alpha = float("nan")
        r2_log = float("nan")
        msd_fit = None
        if int(np.count_nonzero(valid)) >= 5:
            lt = np.log(tau[valid])
            lm = np.log(msd[valid])
            p = np.polyfit(lt, lm, 1)
            alpha = float(p[0])
            pred = np.polyval(p, lt)
            ss_res = float(np.sum((lm - pred) ** 2))
            ss_tot = float(np.sum((lm - np.mean(lm)) ** 2))
            r2_log = (1.0 - (ss_res / ss_tot)) if ss_tot > 1e-18 else 1.0
            msd_fit = np.exp(np.polyval(p, np.log(tau)))

        alpha_tol = 0.25
        r2_min = 0.95
        is_brownian = bool(np.isfinite(alpha) and np.isfinite(r2_log) and (abs(alpha - 1.0) <= alpha_tol) and (r2_log >= r2_min))
        return {
            "phi_unwrapped": phi,
            "tau": tau,
            "msd": msd,
            "msd_fit": msd_fit,
            "max_lag": int(max_lag),
            "lag_count": int(lags.size),
            "alpha": alpha,
            "r2_log": r2_log,
            "alpha_tol": float(alpha_tol),
            "r2_min": float(r2_min),
            "is_brownian": is_brownian,
        }

    def _get_brownian_lag_range(self, show_error: bool = False) -> tuple[Optional[float], Optional[float], bool]:
        s0 = str(self._brownian_lag_min_var.get() or "").strip()
        s1 = str(self._brownian_lag_max_var.get() or "").strip()
        lag_min = None
        lag_max = None
        try:
            if s0:
                lag_min = float(s0)
            if s1:
                lag_max = float(s1)
        except Exception:
            if show_error:
                messagebox.showerror("Brownian test", "Lag range must be numeric (seconds).")
            return None, None, False
        if lag_min is not None and (not np.isfinite(lag_min) or lag_min <= 0.0):
            if show_error:
                messagebox.showerror("Brownian test", "Lag min must be > 0.")
            return None, None, False
        if lag_max is not None and (not np.isfinite(lag_max) or lag_max <= 0.0):
            if show_error:
                messagebox.showerror("Brownian test", "Lag max must be > 0.")
            return None, None, False
        if lag_min is not None and lag_max is not None and lag_max <= lag_min:
            if show_error:
                messagebox.showerror("Brownian test", "Lag max must be greater than lag min.")
            return None, None, False
        return lag_min, lag_max, True

    def _fit_brownian_loglog_in_range(
        self,
        tau: np.ndarray,
        msd: np.ndarray,
        lag_min: Optional[float],
        lag_max: Optional[float],
    ) -> dict:
        t = np.asarray(tau, dtype=np.float64)
        y = np.asarray(msd, dtype=np.float64)
        valid = np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > 0.0)
        if lag_min is not None:
            valid &= (t >= float(lag_min))
        if lag_max is not None:
            valid &= (t <= float(lag_max))
        fit_ok = int(np.count_nonzero(valid)) >= 5
        out = {
            "mask": valid,
            "fit_ok": bool(fit_ok),
            "alpha": float("nan"),
            "r2_log": float("nan"),
            "intercept_log": float("nan"),
            "fit_curve": None,
        }
        if not fit_ok:
            return out
        lt = np.log(t[valid])
        lm = np.log(y[valid])
        p = np.polyfit(lt, lm, 1)
        alpha = float(p[0])
        intercept = float(p[1])
        pred = np.polyval(p, lt)
        ss_res = float(np.sum((lm - pred) ** 2))
        ss_tot = float(np.sum((lm - np.mean(lm)) ** 2))
        r2_log = (1.0 - (ss_res / ss_tot)) if ss_tot > 1e-18 else 1.0
        fit_curve = np.exp(intercept + (alpha * np.log(t)))
        out["alpha"] = alpha
        out["r2_log"] = r2_log
        out["intercept_log"] = intercept
        out["fit_curve"] = fit_curve
        return out

    def _update_brownian_result_label(self) -> None:
        rec = self._brownian_last
        if rec is None:
            return
        m = rec["metrics"]
        lag_min, lag_max, ok = self._get_brownian_lag_range(show_error=False)
        if not ok:
            lag_min = None
            lag_max = None
        fit = self._fit_brownian_loglog_in_range(np.asarray(m["tau"]), np.asarray(m["msd"]), lag_min, lag_max)
        rec["fit_view"] = fit
        nfit = int(np.count_nonzero(fit["mask"]))
        if not fit["fit_ok"]:
            self._brownian_result_var.set(
                f"Need >=5 lag samples in selected range (currently {nfit})."
            )
            return
        alpha = float(fit["alpha"])
        r2 = float(fit["r2_log"])
        tol = float(m["alpha_tol"])
        r2min = float(m["r2_min"])
        is_brownian = bool((abs(alpha - 1.0) <= tol) and (r2 >= r2min))
        verdict = "Likely Brownian" if is_brownian else "Not Brownian"
        if lag_min is None and lag_max is None:
            rtxt = "full lag range"
        else:
            a = f"{lag_min:.3g}" if lag_min is not None else "min"
            b = f"{lag_max:.3g}" if lag_max is not None else "max"
            rtxt = f"lag {a}..{b} s"
        total_lags = int(m.get("lag_count", 0))
        max_lag_s = float(np.max(np.asarray(m["tau"], dtype=np.float64))) if np.asarray(m["tau"]).size else 0.0
        self._brownian_result_var.set(
            f"{verdict} | source={rec['source']} | alpha={alpha:.3f}, R2={r2:.3f} "
            f"| criterion: |alpha-1|<={tol:.2f} and R2>={r2min:.2f} | fit={rtxt} (n={nfit}) "
            f"| lag pts={total_lags}, max lag={max_lag_s:.3g}s"
        )

    def _on_apply_brownian_lag_range(self) -> None:
        _, _, ok = self._get_brownian_lag_range(show_error=True)
        if not ok:
            return
        if self._brownian_last is None:
            self._brownian_result_var.set("Lag range saved. Press Brownian test button.")
            self._render_brownian_tab()
            return
        self._update_brownian_result_label()
        self._render_brownian_tab()

    def _test_current_spot_brownian(self) -> None:
        n = len(self._spot_xy_series)
        if n <= 0:
            messagebox.showerror("Brownian test", "No spots are loaded.")
            return
        idx = max(0, min(int(self._spot_idx), n - 1))
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        if arr_full.ndim != 2 or arr_full.shape[1] != 2 or arr_full.shape[0] < 20:
            messagebox.showerror("Brownian test", "Need at least 20 XY samples for Brownian testing.")
            return

        fs = float(self._fps_for_spot_idx(idx))
        i0, i1 = self._selected_index_range(arr_full.shape[0], fs)
        arr = arr_full[i0:i1]
        if arr.shape[0] < 20:
            messagebox.showerror("Brownian test", "Selected window needs at least 20 samples.")
            return

        src = str(self._brownian_phi_source_var.get() or "Shifted+fitted phi")
        if src == "Shifted+fitted phi":
            fit = self._fit_and_aligned_distribution(arr)
            if fit is None:
                messagebox.showerror("Brownian test", "Shifted+fitted phi unavailable (circle fit failed).")
                return
            phi_wrapped = np.asarray(fit["phi_shift_rad"], dtype=np.float64)
            phi_unwrapped = np.unwrap(phi_wrapped)
        else:
            phi_unwrapped = self._physical_phi_from_xy_series(arr, wrap_2pi=False)

        metrics = self._compute_brownian_metrics(phi_unwrapped, fs)
        if metrics is None:
            messagebox.showerror("Brownian test", "Not enough valid data for Brownian metrics.")
            return

        _, _, ok = self._get_brownian_lag_range(show_error=True)
        if not ok:
            return
        self._brownian_last = {
            "spot_idx": int(idx),
            "source": src,
            "n": int(arr.shape[0]),
            "fs": float(fs),
            "metrics": metrics,
        }
        self._update_brownian_result_label()
        self._status_var.set(f"Brownian test spot {idx + 1}: updated.")
        self._render_brownian_tab()

    def _render_brownian_tab(self) -> None:
        if self._bm_canvas is None:
            return
        self._bm_ax_msd.clear()
        self._bm_ax_inc.clear()
        rec = self._brownian_last
        if rec is None:
            self._bm_ax_msd.set_title("MSD vs lag")
            self._bm_ax_inc.set_title("Log-log fit residuals")
            self._bm_ax_msd.text(0.5, 0.5, "Press Brownian test button", ha="center", va="center", transform=self._bm_ax_msd.transAxes)
            self._bm_ax_inc.text(0.5, 0.5, "Straight-line check view", ha="center", va="center", transform=self._bm_ax_inc.transAxes)
            self._bm_fig.tight_layout()
            self._bm_canvas.draw_idle()
            return

        m = rec["metrics"]
        tau = np.asarray(m["tau"], dtype=np.float64)
        msd = np.asarray(m["msd"], dtype=np.float64)
        lag_min, lag_max, ok = self._get_brownian_lag_range(show_error=False)
        if not ok:
            lag_min = None
            lag_max = None
        fit = self._fit_brownian_loglog_in_range(tau, msd, lag_min, lag_max)
        rec["fit_view"] = fit
        mask = np.asarray(fit["mask"], dtype=bool)
        if np.any(mask):
            self._bm_ax_msd.loglog(tau[mask], msd[mask], "o", ms=3.0, alpha=0.85, label="MSD (selected lag range)")
            fitv = fit.get("fit_curve", None)
            if fit["fit_ok"] and fitv is not None:
                fitv = np.asarray(fitv, dtype=np.float64)
                ok_fit = mask & np.isfinite(fitv) & (fitv > 0.0)
                if np.any(ok_fit):
                    self._bm_ax_msd.loglog(tau[ok_fit], fitv[ok_fit], "-", lw=1.4, color="tab:red", label="log-log fit")
                if bool(self._brownian_show_refs_var.get()):
                    tfit = tau[ok_fit] if np.any(ok_fit) else tau[mask]
                    if tfit.size > 0:
                        t0 = float(np.exp(np.mean(np.log(tfit))))
                        y0 = float(np.exp(float(fit["intercept_log"]) + (float(fit["alpha"]) * np.log(t0))))
                        y_ref1 = y0 * (tau[mask] / max(1e-18, t0))
                        y_ref2 = y0 * ((tau[mask] / max(1e-18, t0)) ** 2)
                        self._bm_ax_msd.loglog(tau[mask], y_ref1, "--", lw=1.0, color="0.35", label="slope 1 ref")
                        self._bm_ax_msd.loglog(tau[mask], y_ref2, "--", lw=1.0, color="0.55", label="slope 2 ref")
        self._bm_ax_msd.set_xlabel("lag (s)")
        self._bm_ax_msd.set_ylabel("MSD(phi) (rad^2)")
        if fit["fit_ok"]:
            self._bm_ax_msd.set_title(f"MSD slope alpha={fit['alpha']:.3f}, R2={fit['r2_log']:.3f}")
        else:
            self._bm_ax_msd.set_title("MSD slope alpha=nan, R2=nan (need >=5 lag points)")
        self._bm_ax_msd.grid(alpha=0.25)
        self._bm_ax_msd.legend(loc="best", fontsize=8)

        # Straight-line residual view on log-log scale.
        if fit["fit_ok"] and np.any(mask):
            lt = np.log(tau[mask])
            lm = np.log(msd[mask])
            pred = float(fit["intercept_log"]) + (float(fit["alpha"]) * lt)
            res = lm - pred
            self._bm_ax_inc.plot(tau[mask], res, "o", ms=3.0, alpha=0.8)
            self._bm_ax_inc.axhline(0.0, color="tab:red", lw=1.0, linestyle=":")
        self._bm_ax_inc.set_xscale("log")
        self._bm_ax_inc.set_xlabel("lag (s)")
        self._bm_ax_inc.set_ylabel("log-MSD residual")
        if fit["fit_ok"]:
            tol = float(m["alpha_tol"])
            r2min = float(m["r2_min"])
            verdict = "Likely Brownian" if (abs(float(fit["alpha"]) - 1.0) <= tol and float(fit["r2_log"]) >= r2min) else "Not Brownian"
            self._bm_ax_inc.set_title(
                f"{verdict} | criterion: |alpha-1|<={tol:.2f}, R2>={r2min:.2f}"
            )
        else:
            self._bm_ax_inc.set_title("Need >=5 lag points in selected range")
        self._bm_ax_inc.grid(alpha=0.2)
        self._bm_fig.tight_layout()
        self._bm_canvas.draw_idle()

    def _add_speed_theta_half_rotation(self) -> None:
        self._update_current_speed_theta_point()

    def _add_speed_theta_full_rotation(self) -> None:
        self._update_current_speed_theta_point()

    def _update_current_speed_theta_point(self) -> None:
        n = len(self._spot_xy_series)
        if n <= 0:
            messagebox.showerror("Phi Span vs sin(theta)", "No spots are loaded.")
            return
        idx = max(0, min(int(self._spot_idx), n - 1))
        if len(self._speed_theta_selected_spots) != n:
            self._speed_theta_selected_spots = [False for _ in range(n)]
        self._speed_theta_selected_spots[idx] = True
        ok = self._upsert_speed_theta_point_for_spot(idx, show_error=True)
        if ok:
            self._sync_include_checkbox()
            self._render_speed_theta_tab()

    def _recompute_all_included_speed_theta_points(self) -> None:
        n = len(self._spot_centers)
        if n <= 0:
            return
        if len(self._speed_theta_selected_spots) != n:
            self._speed_theta_selected_spots = [False for _ in range(n)]
        keep_idxs = [i for i, keep in enumerate(self._speed_theta_selected_spots) if keep]
        if not keep_idxs:
            self._status_var.set("No rods ticked for phi-span vs sin(theta).")
            self._render_speed_theta_tab()
            return
        kept = []
        for i in keep_idxs:
            if self._upsert_speed_theta_point_for_spot(i, show_error=False):
                kept.append(i)
        self._status_var.set(f"Recomputed phi-span metric for {len(kept)}/{len(keep_idxs)} included rods.")
        self._render_speed_theta_tab()

    def _exclude_current_speed_theta_rod(self) -> None:
        n = len(self._spot_centers)
        if n <= 0:
            return
        if len(self._speed_theta_selected_spots) != n:
            self._speed_theta_selected_spots = [False for _ in range(n)]
        idx = max(0, min(int(self._spot_idx), n - 1))
        self._speed_theta_selected_spots[idx] = False
        self._speed_theta_points = [p for p in self._speed_theta_points if int(p.get("spot_idx", -1)) != idx]
        self._sync_include_checkbox()
        self._render_speed_theta_tab()

    def _clear_speed_theta_points(self) -> None:
        self._speed_theta_points = []
        self._speed_theta_selected_spots = [False for _ in self._spot_centers]
        self._sync_include_checkbox()
        self._render_speed_theta_tab()

    def _current_phi_fromaxis_for_dwell(self) -> tuple[Optional[np.ndarray], float]:
        if not self._spot_xy_series:
            return None, float(self._current_fps())
        idx = max(0, min(int(self._spot_idx), len(self._spot_xy_series) - 1))
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        fps = float(self._current_fps())
        if arr_full.ndim != 2 or arr_full.shape[1] != 2 or arr_full.shape[0] < 3:
            return None, fps
        i0, i1 = self._selected_index_range(arr_full.shape[0], fps)
        arr = arr_full[i0:i1]
        if arr.shape[0] < 3:
            return None, fps
        fit = self._sphere_fit
        if fit is None or int(np.asarray(fit.get("u", np.zeros((0, 3), dtype=np.float64))).shape[0]) != int(arr.shape[0]):
            fit = self._fit_unit_sphere_distribution(arr)
        if fit is None:
            return None, fps
        phi = np.asarray(fit.get("phi_axis_unwrapped_rad", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        if phi.ndim != 1 or phi.size < 3:
            return None, fps
        return phi, fps

    def _dwell_detect_plateaus(self, phi_rad: np.ndarray, fps: float) -> Optional[dict]:
        phi = np.asarray(phi_rad, dtype=np.float64)
        if phi.ndim != 1 or phi.size < 5:
            return None
        n_avg = min(self._get_dwell_avg_points(), int(phi.size))
        if n_avg % 2 == 0:
            n_avg -= 1
        n_avg = max(11, n_avg)
        min_points = max(10, self._get_dwell_min_points())
        jump_deg = float(self._get_dwell_jump_deg())
        # phi_fromaxis(t) from the Unit Sphere tab is already unwrapped. Use
        # local average levels before/after each possible boundary so the
        # threshold means "minimum change in average phi" rather than a noisy
        # one-frame derivative.
        phi_unwrapped = np.asarray(phi, dtype=np.float64)
        if n_avg > 1:
            kernel = np.ones((n_avg,), dtype=np.float64) / float(n_avg)
            phi_smooth = np.convolve(phi_unwrapped, kernel, mode="same")
            half = n_avg // 2
            if phi.size > (2 * half + 1):
                phi_smooth[:half] = phi_smooth[half]
                phi_smooth[-half:] = phi_smooth[-half - 1]
        else:
            phi_smooth = phi_unwrapped

        half_win = max(5, int(n_avg // 2))
        raw_jumps_list = []
        raw_score_list = []
        for j in range(half_win, int(phi_unwrapped.size) - half_win):
            before = phi_unwrapped[j - half_win : j]
            after = phi_unwrapped[j : j + half_win]
            if before.size < half_win or after.size < half_win:
                continue
            delta_deg = abs(float(np.degrees(np.mean(after) - np.mean(before))))
            if delta_deg >= jump_deg:
                raw_jumps_list.append(int(j))
                raw_score_list.append(float(delta_deg))
        raw_jumps = np.asarray(raw_jumps_list, dtype=np.int64)
        raw_scores = np.asarray(raw_score_list, dtype=np.float64)
        if raw_jumps.size:
            merged = [int(raw_jumps[0])]
            merged_scores = [float(raw_scores[0])]
            merge_gap = max(1, n_avg // 2)
            for j, score in zip(raw_jumps[1:], raw_scores[1:]):
                jj = int(j)
                if jj - merged[-1] <= merge_gap:
                    # Keep the largest average-level-change point inside a transition cluster.
                    if float(score) > float(merged_scores[-1]):
                        merged[-1] = jj
                        merged_scores[-1] = float(score)
                else:
                    merged.append(jj)
                    merged_scores.append(float(score))
            jumps = np.asarray(merged, dtype=np.int64)
            jump_scores = np.asarray(merged_scores, dtype=np.float64)
        else:
            jumps = np.zeros((0,), dtype=np.int64)
            jump_scores = np.zeros((0,), dtype=np.float64)

        edges = np.r_[0, jumps, phi.size]
        plateaus = []
        dwell_s = []
        for a, b in zip(edges[:-1], edges[1:]):
            a = int(a)
            b = int(b)
            if (b - a) < min_points:
                continue
            dwell = float(b - a) / max(1e-9, float(fps))
            val = float(np.median(np.degrees(phi_smooth[a:b]))) if b > a else 0.0
            plateaus.append({"start_i": a, "end_i": b, "duration_s": dwell, "phi_deg": val})
            dwell_s.append(dwell)
        return {
            "phi_smooth_rad": phi_smooth,
            "jump_indices": jumps,
            "jump_delta_deg": jump_scores,
            "plateaus": plateaus,
            "dwell_s": np.asarray(dwell_s, dtype=np.float64),
            "avg_points": int(n_avg),
            "jump_deg": float(jump_deg),
            "min_points": int(min_points),
        }

    def _fit_dwell_distributions(self, dwell_s: np.ndarray) -> list[dict]:
        x = np.asarray(dwell_s, dtype=np.float64)
        x = x[np.isfinite(x) & (x > 0.0)]
        if x.size < 2:
            return []
        fits: list[dict] = []

        def add_fit(name: str, params: tuple, pdf_fn, k: int, logpdf: np.ndarray) -> None:
            ok = np.isfinite(logpdf)
            if int(np.count_nonzero(ok)) <= 0:
                return
            ll = float(np.sum(logpdf[ok]))
            aic = float((2 * int(k)) - (2.0 * ll))
            fits.append({"name": name, "params": params, "k": int(k), "log_likelihood": ll, "aic": aic, "pdf_fn": pdf_fn})

        mean_x = float(np.mean(x))
        if mean_x > 0.0:
            scale = mean_x
            logpdf = -np.log(scale) - (x / scale)
            add_fit("exponential", (scale,), lambda z, s=scale: (1.0 / s) * np.exp(-np.asarray(z) / s), 1, logpdf)

        if _scipy_stats is not None and x.size >= 3:
            try:
                shape, loc, scale = _scipy_stats.gamma.fit(x, floc=0.0)
                logpdf = _scipy_stats.gamma.logpdf(x, shape, loc=0.0, scale=scale)
                add_fit("gamma", (float(shape), float(scale)), lambda z, a=shape, s=scale: _scipy_stats.gamma.pdf(z, a, loc=0.0, scale=s), 2, logpdf)
            except Exception:
                pass
            try:
                shape, loc, scale = _scipy_stats.weibull_min.fit(x, floc=0.0)
                logpdf = _scipy_stats.weibull_min.logpdf(x, shape, loc=0.0, scale=scale)
                add_fit(
                    "weibull",
                    (float(shape), float(scale)),
                    lambda z, c=shape, s=scale: _scipy_stats.weibull_min.pdf(z, c, loc=0.0, scale=s),
                    2,
                    logpdf,
                )
            except Exception:
                pass
            try:
                shape, loc, scale = _scipy_stats.lognorm.fit(x, floc=0.0)
                logpdf = _scipy_stats.lognorm.logpdf(x, shape, loc=0.0, scale=scale)
                add_fit("lognormal", (float(shape), float(scale)), lambda z, a=shape, s=scale: _scipy_stats.lognorm.pdf(z, a, loc=0.0, scale=s), 2, logpdf)
            except Exception:
                pass
        fits.sort(key=lambda r: float(r["aic"]))
        return fits

    def _render_dwell_tab(self) -> None:
        if self._dwell_canvas is None:
            return
        self._dwell_ax_phi.clear()
        self._dwell_ax_hist.clear()
        phi, fps = self._current_phi_fromaxis_for_dwell()
        if phi is None or phi.size < 5:
            self._dwell_ax_phi.set_title("Plateau detection")
            self._dwell_ax_phi.text(0.5, 0.5, "Need current unit-sphere phi_fromaxis(t)", ha="center", va="center", transform=self._dwell_ax_phi.transAxes)
            self._dwell_ax_hist.set_title("Dwell-time distribution")
            self._dwell_info_var.set("No dwell data.")
            self._dwell_fig.tight_layout()
            self._dwell_canvas.draw_idle()
            return
        rec = self._dwell_detect_plateaus(phi, fps)
        if rec is None:
            self._dwell_info_var.set("Could not detect dwell data.")
            self._dwell_canvas.draw_idle()
            return
        smooth_deg = np.degrees(np.asarray(rec["phi_smooth_rad"], dtype=np.float64))
        t = np.arange(phi.size, dtype=np.float64) / max(1e-9, float(fps))
        jumps = np.asarray(rec["jump_indices"], dtype=np.int64)
        jump_delta = np.asarray(rec.get("jump_delta_deg", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        dwell_s = np.asarray(rec["dwell_s"], dtype=np.float64)

        self._dwell_ax_phi.plot(t, np.degrees(phi), color="0.75", lw=0.65, label="phi_fromaxis(t)")
        self._dwell_ax_phi.plot(t, smooth_deg, color="tab:blue", lw=1.2, label=f"moving average {rec['avg_points']} pts")
        for k, j in enumerate(jumps):
            if 0 <= int(j) < t.size:
                self._dwell_ax_phi.axvline(float(t[int(j)]), color="tab:red", lw=1.0, alpha=0.75)
                if k < jump_delta.size:
                    self._dwell_ax_phi.text(
                        float(t[int(j)]),
                        0.98,
                        f"{float(jump_delta[k]):.0f} deg",
                        color="tab:red",
                        fontsize=7,
                        ha="center",
                        va="top",
                        rotation=90,
                        transform=self._dwell_ax_phi.get_xaxis_transform(),
                    )
        for p in rec["plateaus"]:
            a = int(p["start_i"])
            b = int(p["end_i"])
            if 0 <= a < t.size and 0 < b <= t.size:
                self._dwell_ax_phi.hlines(float(p["phi_deg"]), t[a], t[b - 1], color="black", lw=2.0, alpha=0.65)
        self._dwell_ax_phi.set_xlabel("time (s)")
        self._dwell_ax_phi.set_ylabel("phi (deg)")
        self._dwell_ax_phi.set_title("Plateaus and detected transitions in unit-sphere phi_fromaxis(t)")
        self._dwell_ax_phi.grid(alpha=0.25)
        self._dwell_ax_phi.legend(loc="best", fontsize=8)

        fits = self._fit_dwell_distributions(dwell_s)
        if dwell_s.size > 0:
            bin_w = self._get_dwell_bin_width_s()
            x_max = float(np.max(dwell_s)) * 1.08
            bins = np.arange(0.0, max(x_max + bin_w, bin_w * 2.0), bin_w)
            if bins.size < 3:
                bins = np.linspace(0.0, max(x_max, bin_w), 6)
            self._dwell_ax_hist.hist(dwell_s, bins=bins, density=True, color="tab:green", alpha=0.65, label=f"dwell durations ({bin_w:g} s bins)")
            xx = np.linspace(0.0, max(x_max, 1e-6), 400)
            for fit in fits[:4]:
                try:
                    yy = fit["pdf_fn"](xx)
                    self._dwell_ax_hist.plot(xx, yy, lw=1.4, label=f"{fit['name']} AIC={fit['aic']:.1f}")
                except Exception:
                    continue
            self._dwell_ax_hist.set_xlabel("time spent on plateau before transition (s)")
            self._dwell_ax_hist.set_ylabel("probability density")
            self._dwell_ax_hist.set_title("Dwell-time distribution and candidate fits")
            self._dwell_ax_hist.grid(alpha=0.25)
            self._dwell_ax_hist.legend(loc="best", fontsize=8)
        else:
            self._dwell_ax_hist.text(0.5, 0.5, "No plateaus survived min-length filter", ha="center", va="center", transform=self._dwell_ax_hist.transAxes)
            self._dwell_ax_hist.set_title("Dwell-time distribution")

        if fits:
            best = fits[0]
            self._dwell_info_var.set(
                f"{dwell_s.size} dwell(s), {jumps.size} jump(s). Best fit: {best['name']} AIC={best['aic']:.1f}"
            )
        else:
            self._dwell_info_var.set(f"{dwell_s.size} dwell(s), {jumps.size} jump(s). Need more dwell times for fits.")
        self._dwell_fig.tight_layout()
        self._dwell_canvas.draw_idle()

    def _render_raw_gallery_tab(self) -> None:
        if self._raw_canvas is None:
            return
        self._raw_ax_xy.clear()
        self._raw_ax_2phi.clear()
        self._raw_ax_r.clear()
        self._raw_ax_theta.clear()
        self._raw_xy_path_artist = None
        self._raw_xy_trail_artist = None
        self._raw_xy_head_artist = None
        self._raw_2phi_marker = None
        self._raw_r_marker = None
        self._raw_theta_marker = None

        series, fps, name = self._current_raw_gallery_series()
        if series is None:
            self._raw_ax_xy.set_title("Raw X/Y playback")
            self._raw_ax_xy.text(0.5, 0.5, "Load data first", ha="center", va="center", transform=self._raw_ax_xy.transAxes)
            self._raw_play_info_var.set("No raw XY series loaded.")
            self._raw_fig.tight_layout()
            self._raw_canvas.draw_idle()
            return

        xy = np.asarray(series["xy"], dtype=np.float64)
        r = np.asarray(series["r"], dtype=np.float64)
        two_phi = np.asarray(series["two_phi_deg"], dtype=np.float64)
        theta = np.asarray(series["theta_deg"], dtype=np.float64)
        n = int(xy.shape[0])
        self._raw_play_idx = max(0, min(int(self._raw_play_idx), max(0, n - 1)))
        t = np.arange(n, dtype=np.float64) / max(1e-9, float(fps))

        if n > 0:
            (self._raw_xy_path_artist,) = self._raw_ax_xy.plot(xy[:, 0], xy[:, 1], color="0.82", lw=0.8)
        self._raw_xy_trail_artist = self._raw_ax_xy.scatter([], [], s=14, color="tab:blue", alpha=0.8)
        (self._raw_xy_head_artist,) = self._raw_ax_xy.plot([], [], marker="o", markersize=6, color="black", linestyle="")
        self._raw_ax_xy.axhline(0.0, color="0.82", lw=1)
        self._raw_ax_xy.axvline(0.0, color="0.82", lw=1)
        self._raw_ax_xy.set_xlim(-1.0, 1.0)
        self._raw_ax_xy.set_ylim(-1.0, 1.0)
        self._raw_ax_xy.set_aspect("equal", adjustable="box")
        self._raw_ax_xy.set_xlabel("X")
        self._raw_ax_xy.set_ylabel("Y")
        self._raw_ax_xy.set_title("Raw anisotropy X/Y playback")

        self._raw_ax_2phi.plot(t, two_phi, color="tab:orange", lw=0.8)
        self._raw_ax_2phi.set_ylim(0.0, 360.0)
        self._raw_ax_2phi.set_xlabel("time (s)")
        self._raw_ax_2phi.set_ylabel("2phi raw (deg)")
        self._raw_ax_2phi.set_title("Raw 2phi(t) from atan2(Y, X)")
        self._raw_ax_2phi.grid(alpha=0.25)
        self._raw_2phi_marker = self._raw_ax_2phi.axvline(0.0, color="black", lw=1.0)

        self._raw_ax_r.plot(t, r, color="tab:green", lw=0.8)
        self._raw_ax_r.set_ylim(0.0, max(1.0, float(np.nanmax(r)) * 1.05 if r.size else 1.0))
        self._raw_ax_r.set_xlabel("time (s)")
        self._raw_ax_r.set_ylabel("r")
        self._raw_ax_r.set_title("r(t) from raw X/Y")
        self._raw_ax_r.grid(alpha=0.25)
        self._raw_r_marker = self._raw_ax_r.axvline(0.0, color="black", lw=1.0)

        self._raw_ax_theta.plot(t, theta, color="tab:red", lw=0.8)
        self._raw_ax_theta.set_ylim(0.0, 90.0)
        self._raw_ax_theta.set_xlabel("time (s)")
        self._raw_ax_theta.set_ylabel("theta (deg)")
        self._raw_ax_theta.set_title("theta(t) from r(t)")
        self._raw_ax_theta.grid(alpha=0.25)
        self._raw_theta_marker = self._raw_ax_theta.axvline(0.0, color="black", lw=1.0)

        self._raw_play_info_var.set(f"{name} | {n} points | FPS={fps:.3f}")
        self._update_raw_play_artists()
        self._raw_fig.tight_layout()
        self._raw_canvas.draw_idle()

    def _update_raw_play_artists(self) -> None:
        if self._raw_canvas is None:
            return
        series, fps, _name = self._current_raw_gallery_series()
        if series is None:
            return
        xy = np.asarray(series["xy"], dtype=np.float64)
        n = int(xy.shape[0])
        if n <= 0:
            return
        idx = max(0, min(int(self._raw_play_idx), n - 1))
        trail = self._get_raw_play_trail()
        i0 = max(0, idx - trail + 1)
        seg = xy[i0 : idx + 1]
        if self._raw_xy_trail_artist is not None:
            self._raw_xy_trail_artist.set_offsets(seg if seg.size else np.zeros((0, 2), dtype=np.float64))
        if self._raw_xy_head_artist is not None:
            self._raw_xy_head_artist.set_data([xy[idx, 0]], [xy[idx, 1]])
        t = float(idx) / max(1e-9, float(fps))
        for marker in (self._raw_2phi_marker, self._raw_r_marker, self._raw_theta_marker):
            if marker is not None:
                marker.set_xdata([t, t])
        self._raw_canvas.draw_idle()

    def _on_raw_play(self) -> None:
        if self._raw_canvas is None:
            return
        series, _fps, _name = self._current_raw_gallery_series()
        if series is None:
            return
        n = int(np.asarray(series["xy"], dtype=np.float64).shape[0])
        if n <= 0:
            return
        if self._raw_play_idx >= (n - 1):
            self._raw_play_idx = 0
            self._update_raw_play_artists()
        self._raw_play_running = True
        self._schedule_raw_play_tick()

    def _on_raw_pause(self) -> None:
        self._stop_raw_playback()

    def _on_raw_reset(self) -> None:
        self._stop_raw_playback()
        self._raw_play_idx = 0
        self._update_raw_play_artists()

    def _stop_raw_playback(self) -> None:
        self._raw_play_running = False
        if self._raw_play_after is not None:
            try:
                self.root.after_cancel(self._raw_play_after)
            except Exception:
                pass
            self._raw_play_after = None

    def _schedule_raw_play_tick(self) -> None:
        if self._raw_canvas is None:
            self._raw_play_running = False
            return
        if not self._raw_play_running:
            return
        if self._raw_play_after is not None:
            return
        speed = self._get_raw_play_speed()
        dt_ms = max(1, int(round(1000.0 / speed)))
        self._raw_play_after = self.root.after(dt_ms, self._raw_play_tick)

    def _raw_play_tick(self) -> None:
        self._raw_play_after = None
        if self._raw_canvas is None:
            self._raw_play_running = False
            return
        if not self._raw_play_running:
            return
        series, _fps, _name = self._current_raw_gallery_series()
        if series is None:
            self._raw_play_running = False
            return
        n = int(np.asarray(series["xy"], dtype=np.float64).shape[0])
        if self._raw_play_idx < (n - 1):
            self._raw_play_idx += 1
            self._update_raw_play_artists()
            self._schedule_raw_play_tick()
        else:
            self._raw_play_running = False

    def _raw_gallery_quality_score(self, xy: np.ndarray) -> float:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            return float("-inf")
        dx = float(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0]))
        dy = float(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]))
        if not np.isfinite(dx) or not np.isfinite(dy):
            return float("-inf")
        return float((dx * dx) + (dy * dy))

    def _raw_gallery_default_path(self) -> Path:
        if self.source_path is not None:
            base_dir = Path(self.source_path).parent
            stem = Path(self.source_path).stem
        elif self.source_paths:
            base_dir = Path(self.source_paths[0]).parent
            stem = "inspection_batch"
        else:
            base_dir = Path.cwd()
            stem = "raw_xy_gallery"
        return base_dir / f"{stem}_raw_xy_gallery.tiff"

    def _make_raw_gallery_page_image(self, idx: int) -> Optional["Image.Image"]:
        if Image is None:
            return None
        if idx < 0 or idx >= len(self._spot_xy_series):
            return None
        xy = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        series = self._raw_series_from_xy(xy)
        xy = np.asarray(series["xy"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 1:
            return None
        fps = float(self._fps_for_spot_idx(idx))
        t = np.arange(xy.shape[0], dtype=np.float64) / max(1e-9, fps)
        r = np.asarray(series["r"], dtype=np.float64)
        two_phi = np.asarray(series["two_phi_deg"], dtype=np.float64)
        theta = np.asarray(series["theta_deg"], dtype=np.float64)
        label = str(self._spot_names[idx]) if self._spot_names and idx < len(self._spot_names) else f"spot {idx + 1}"
        score = self._raw_gallery_quality_score(xy)

        fig = Figure(figsize=(15.5, 4.2), dpi=120)
        ax_xy = fig.add_subplot(141)
        ax_phi = fig.add_subplot(142)
        ax_r = fig.add_subplot(143)
        ax_theta = fig.add_subplot(144)

        ax_xy.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=0.75)
        ax_xy.scatter(xy[0, 0], xy[0, 1], s=18, color="tab:green", label="start")
        ax_xy.scatter(xy[-1, 0], xy[-1, 1], s=18, color="tab:red", label="end")
        ax_xy.axhline(0.0, color="0.82", lw=0.8)
        ax_xy.axvline(0.0, color="0.82", lw=0.8)
        ax_xy.set_xlim(-1.0, 1.0)
        ax_xy.set_ylim(-1.0, 1.0)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("raw X/Y")
        ax_xy.legend(loc="best", fontsize=7)

        ax_phi.plot(t, two_phi, color="tab:orange", lw=0.75)
        ax_phi.set_ylim(0.0, 360.0)
        ax_phi.set_xlabel("time (s)")
        ax_phi.set_ylabel("2phi raw (deg)")
        ax_phi.set_title("2phi(t)")
        ax_phi.grid(alpha=0.2)

        ax_r.plot(t, r, color="tab:green", lw=0.75)
        ax_r.set_ylim(0.0, max(1.0, float(np.nanmax(r)) * 1.05 if r.size else 1.0))
        ax_r.set_xlabel("time (s)")
        ax_r.set_ylabel("r")
        ax_r.set_title("r(t)")
        ax_r.grid(alpha=0.2)

        ax_theta.plot(t, theta, color="tab:red", lw=0.75)
        ax_theta.set_ylim(0.0, 90.0)
        ax_theta.set_xlabel("time (s)")
        ax_theta.set_ylabel("theta (deg)")
        ax_theta.set_title("theta(t)")
        ax_theta.grid(alpha=0.2)

        fig.suptitle(f"{label} | quality={score:.4g} | points={xy.shape[0]} | FPS={fps:.3f}", fontsize=11)
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba()).copy()
        return Image.fromarray(rgba).convert("RGB")

    def _export_raw_gallery_tiff(self) -> None:
        if Image is None:
            messagebox.showerror("Export TIFF gallery", "Pillow is not available, so TIFF export cannot be written.")
            return
        if not self._spot_xy_series:
            messagebox.showerror("Export TIFF gallery", "Load rods before exporting a gallery.")
            return
        default_path = self._raw_gallery_default_path()
        out = filedialog.asksaveasfilename(
            title="Save raw XY gallery TIFF stack",
            initialdir=str(default_path.parent),
            initialfile=str(default_path.name),
            defaultextension=".tiff",
            filetypes=[("TIFF stack", "*.tiff *.tif"), ("All files", "*.*")],
        )
        if not out:
            return
        order = sorted(
            range(len(self._spot_xy_series)),
            key=lambda i: self._raw_gallery_quality_score(np.asarray(self._spot_xy_series[i], dtype=np.float64)),
            reverse=True,
        )
        images = []
        for idx in order:
            img = self._make_raw_gallery_page_image(idx)
            if img is not None:
                images.append(img)
        if not images:
            messagebox.showerror("Export TIFF gallery", "No valid rods were available for export.")
            return
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(out_path, save_all=True, append_images=images[1:], compression="tiff_deflate")
        self._status_var.set(f"Saved raw XY gallery TIFF with {len(images)} rod page(s): {out_path}")
        self._raw_play_info_var.set(f"Saved TIFF gallery: {out_path}")

    def _upsert_speed_theta_point_for_spot(self, idx: int, show_error: bool = True) -> bool:
        n = len(self._spot_xy_series)
        if n <= 0:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "No spots are loaded.")
            return False
        if len(self._speed_theta_selected_spots) != n:
            self._speed_theta_selected_spots = [False for _ in range(n)]
        idx = max(0, min(int(idx), n - 1))
        arr_full = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        if arr_full.ndim != 2 or arr_full.shape[1] != 2 or arr_full.shape[0] < 3:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "Current spot has insufficient XY points.")
            return False
        fs = float(self._fps_for_spot_idx(idx))
        i0, i1 = self._selected_index_range(arr_full.shape[0], fs)
        if (i1 - i0) < 3:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "Select a longer phi(t) interval (>=3 points).")
            return False
        dt = float(i1 - i0 - 1) / max(1e-9, fs)
        if dt <= 0.0:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "Selected time interval is too short.")
            return False
        arr = arr_full[i0:i1]
        fit = self._fit_and_aligned_distribution(arr)
        if fit is None:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "Circle fit failed for selected interval.")
            return False
        r = float(fit["radius"])
        r_seg = np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))
        th_seg, clipped_seg = self._theta_from_radius(r_seg)
        th_seg_deg = np.degrees(th_seg) if th_seg.size else np.asarray([], dtype=np.float64)
        th_min = float(np.min(th_seg_deg)) if th_seg_deg.size else float("nan")
        th_max = float(np.max(th_seg_deg)) if th_seg_deg.size else float("nan")
        theta_arr, clipped = self._theta_from_radius(np.asarray([r], dtype=np.float64))
        theta = float(theta_arr[0]) if theta_arr.size else 0.0
        sin_theta = float(np.sin(theta))
        phi_rel = np.asarray(fit["phi_rel_deg"], dtype=np.float64)
        if phi_rel.size < 3:
            if show_error:
                messagebox.showerror("Phi Span vs sin(theta)", "Not enough phi samples in selected interval.")
            return False
        p5, p95 = np.percentile(phi_rel, [5.0, 95.0])
        phi_span_deg = float(p95 - p5)
        self._speed_theta_selected_spots[idx] = True
        self._speed_theta_points = [p for p in self._speed_theta_points if int(p.get("spot_idx", -1)) != idx]
        self._speed_theta_points.append(
            {
                "sin_theta": sin_theta,
                "phi_span_deg": phi_span_deg,
                "phi_p5_deg": float(p5),
                "phi_p95_deg": float(p95),
                "theta_rad": theta,
                "theta_min_deg": th_min,
                "theta_max_deg": th_max,
                "radius": r,
                "spot_idx": int(idx),
                "dt_s": dt,
                "r_clipped": bool((clipped > 0) or (clipped_seg > 0)),
            }
        )
        self._sync_include_checkbox()
        self._status_var.set(
            f"Updated spot {idx + 1}: phi_span(5-95)={phi_span_deg:.2f} deg, "
            f"sin(theta)={sin_theta:.4f}, theta range=[{th_min:.2f}, {th_max:.2f}] deg"
        )
        return True

    def _render_speed_theta_tab(self) -> None:
        if self._st_canvas is None:
            return
        self._st_ax.clear()
        visible = [
            p
            for p in self._speed_theta_points
            if (0 <= int(p.get("spot_idx", -1)) < len(self._speed_theta_selected_spots))
            and bool(self._speed_theta_selected_spots[int(p.get("spot_idx", -1))])
        ]
        if not visible:
            self._st_ax.set_title("Phi Span (P95-P5) vs sin(theta)")
            self._st_ax.text(
                0.5,
                0.5,
                "No visible points yet.\nTick rods to include, then update current rod.",
                ha="center",
                va="center",
                transform=self._st_ax.transAxes,
            )
            self._speed_theta_info_var.set("No visible points yet. Tick rods to include.")
            self._st_fig.tight_layout()
            self._st_canvas.draw_idle()
            return

        s = np.asarray([p["sin_theta"] for p in visible], dtype=np.float64)
        v = np.asarray([p["phi_span_deg"] for p in visible], dtype=np.float64)
        self._st_ax.scatter(s, v, s=44, alpha=0.9, color="tab:blue")
        self._st_ax.set_xlabel("sin(theta) from fitted-circle radius")
        self._st_ax.set_ylabel("phi spread (deg) = P95-P5 after circle fit")
        self._st_ax.set_title("Phi Span (P95-P5) vs sin(theta)")
        self._st_ax.grid(alpha=0.25)
        n = len(visible)
        smin = float(np.min(s)) if s.size else 0.0
        smax = float(np.max(s)) if s.size else 0.0
        vmin = float(np.min(v)) if v.size else 0.0
        vmax = float(np.max(v)) if v.size else 0.0
        clipped_n = sum(1 for p in visible if bool(p.get("r_clipped", False)))
        self._speed_theta_info_var.set(
            f"Visible rods={n}  sin(theta)=[{smin:.3f},{smax:.3f}]  phi_spread=[{vmin:.2f},{vmax:.2f}] deg  clipped-r={clipped_n}"
        )
        self._st_fig.tight_layout()
        self._st_canvas.draw_idle()

    def _include_up_to_spot(self) -> None:
        n = len(self._spot_centers)
        if n <= 0:
            return
        if len(self._avg_selected_spots) != n:
            self._avg_selected_spots = [False for _ in range(n)]
        try:
            k = int(self._include_upto_var.get())
        except Exception:
            messagebox.showerror("Average selection", "Spot number must be an integer.")
            return
        k = max(0, min(k, n))
        self._avg_selected_spots = [i < k for i in range(n)]
        self._sync_include_checkbox()
        self._render_average_distribution()

    def _best_window_start_deg(self, phi_deg: np.ndarray, window_deg: float = 9.0) -> float:
        vals = np.mod(np.asarray(phi_deg, dtype=np.float64), 360.0)
        if vals.size == 0:
            return 0.0
        vals.sort()
        ext = np.concatenate([vals, vals + 360.0])
        n = int(vals.size)
        j = 0
        best_i = 0
        best_count = -1
        w = float(window_deg)
        for i in range(n):
            if j < i:
                j = i
            end = vals[i] + w
            while j < (i + n) and ext[j] < end:
                j += 1
            c = j - i
            if c > best_count:
                best_count = c
                best_i = i
        return float(vals[best_i])


def main() -> None:
    root = tk.Tk()
    app = AngleDistributionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
