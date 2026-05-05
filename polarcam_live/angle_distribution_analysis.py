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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import Detection_alg_offline as detect_spinners
from pol_reconstruction import make_qu_reconstructor

try:
    from scipy.signal import welch as _welch  # type: ignore
except Exception:
    _welch = None


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
        self._analysis_mode = "widefield"
        self._spot_names: list[str] = []
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
        self._sphere_dual_axis_var = tk.BooleanVar(value=False)

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
        self._sphere_info_var = tk.StringVar(value="Press analysis button to reconstruct unit-sphere trajectory.")

        self._busy = False

        self._build_ui()
        self._render_all()
        self._render_average_distribution()
        self._render_speed_theta_tab()
        self._render_brownian_tab()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

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

        self._abs_var = tk.BooleanVar(value=self._abs_range_filter_enabled)
        ttk.Checkbutton(
            top,
            text=f"Filter max(range(X),range(Y)) > {self.ABS_RANGE_MIN:.2f}",
            variable=self._abs_var,
            command=self._rerun_analysis,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self._dir_var = tk.BooleanVar(value=self._dir_filter_enabled)
        ttk.Checkbutton(
            top,
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
        self._include_chk = ttk.Checkbutton(
            nav,
            text="Include this spot in average",
            variable=self._include_current_var,
            command=self._on_toggle_include_current,
        )
        self._include_chk.pack(side=tk.LEFT, padx=(12, 0))
        self._include_speed_theta_chk = ttk.Checkbutton(
            nav,
            text="Include this spot in phi-span plot",
            variable=self._include_speed_theta_var,
            command=self._on_toggle_include_speed_theta_current,
        )
        self._include_speed_theta_chk.pack(side=tk.LEFT, padx=(8, 0))
        self._analysis_btn = ttk.Button(
            nav,
            text="Analyze Current Spot Angle Distribution",
            command=self._analyze_current_spot_distribution,
        )
        self._analysis_btn.pack(side=tk.RIGHT)

        self._status_var = tk.StringVar(value="Load a .npy stack to begin.")
        ttk.Label(self.root, textvariable=self._status_var, padding=(8, 0, 8, 8)).pack(side=tk.TOP, fill=tk.X)

        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._notebook = notebook
        tab_spot = ttk.Frame(notebook)
        tab_avg = ttk.Frame(notebook)
        tab_sphere = ttk.Frame(notebook)
        tab_speed_theta = ttk.Frame(notebook)
        tab_brownian = ttk.Frame(notebook)
        notebook.add(tab_spot, text="Spot View")
        notebook.add(tab_avg, text="Average Distribution")
        notebook.add(tab_sphere, text="Unit Sphere")
        notebook.add(tab_speed_theta, text="Phi Span vs sin(theta)")
        notebook.add(tab_brownian, text="Brownian Test")

        fig = Figure(figsize=(12, 8), dpi=100)
        self._ax_xy = fig.add_subplot(221)
        self._ax_phi = fig.add_subplot(222)
        self._ax_shift = fig.add_subplot(223)
        self._ax_hist = fig.add_subplot(224)
        self._fig = fig

        canvas = FigureCanvasTkAgg(fig, master=tab_spot)
        self._canvas = canvas
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._mpl_cid_press = canvas.mpl_connect("button_press_event", self._on_phi_press)
        self._mpl_cid_motion = canvas.mpl_connect("motion_notify_event", self._on_phi_motion)
        self._mpl_cid_release = canvas.mpl_connect("button_release_event", self._on_phi_release)

        avg_top = ttk.Frame(tab_avg, padding=(8, 8, 8, 0))
        avg_top.pack(side=tk.TOP, fill=tk.X)
        self._avg_info_var = tk.StringVar(value="No selected spots yet.")
        ttk.Label(avg_top, textvariable=self._avg_info_var).pack(side=tk.LEFT)
        ttk.Label(avg_top, text="Include spots 1..").pack(side=tk.RIGHT, padx=(8, 4))
        self._include_upto_var = tk.StringVar(value="0")
        ttk.Entry(avg_top, textvariable=self._include_upto_var, width=6).pack(side=tk.RIGHT)
        self._include_upto_btn = ttk.Button(avg_top, text="Apply", command=self._include_up_to_spot)
        self._include_upto_btn.pack(side=tk.RIGHT, padx=(4, 8))
        self._avg_refresh_btn = ttk.Button(avg_top, text="Refresh average", command=self._render_average_distribution)
        self._avg_refresh_btn.pack(side=tk.RIGHT)

        avg_fig = Figure(figsize=(10, 6), dpi=100)
        self._avg_ax = avg_fig.add_subplot(111)
        self._avg_fig = avg_fig
        avg_canvas = FigureCanvasTkAgg(avg_fig, master=tab_avg)
        self._avg_canvas = avg_canvas
        avg_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))

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

        sphere_fig = Figure(figsize=(12, 8), dpi=100)
        self._ax_sphere_3d = sphere_fig.add_subplot(221, projection="3d")
        self._ax_sphere_plane = sphere_fig.add_subplot(222)
        self._ax_sphere_phi = sphere_fig.add_subplot(223)
        self._ax_sphere_hist = sphere_fig.add_subplot(224)
        self._sphere_fig = sphere_fig
        sphere_canvas = FigureCanvasTkAgg(sphere_fig, master=tab_sphere)
        self._sphere_canvas = sphere_canvas
        sphere_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        self._sphere_mpl_cid_press = sphere_canvas.mpl_connect("button_press_event", self._on_sphere_phi_press)
        self._sphere_mpl_cid_motion = sphere_canvas.mpl_connect("motion_notify_event", self._on_sphere_phi_motion)
        self._sphere_mpl_cid_release = sphere_canvas.mpl_connect("button_release_event", self._on_sphere_phi_release)

        st_top = ttk.Frame(tab_speed_theta, padding=(8, 8, 8, 0))
        st_top.pack(side=tk.TOP, fill=tk.X)
        self._st_auto_half_btn = ttk.Button(
            st_top,
            text="Update Current Rod",
            command=self._update_current_speed_theta_point,
        )
        self._st_auto_half_btn.pack(side=tk.LEFT)
        self._st_add_half_btn = ttk.Button(st_top, text="Recompute Included", command=self._recompute_all_included_speed_theta_points)
        self._st_add_half_btn.pack(side=tk.LEFT)
        self._st_add_full_btn = ttk.Button(st_top, text="Untick Current Rod", command=self._exclude_current_speed_theta_rod)
        self._st_add_full_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._st_clear_btn = ttk.Button(st_top, text="Clear All", command=self._clear_speed_theta_points)
        self._st_clear_btn.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(st_top, textvariable=self._speed_theta_info_var).pack(side=tk.LEFT, padx=(12, 0))

        st_fig = Figure(figsize=(10, 6), dpi=100)
        self._st_ax = st_fig.add_subplot(111)
        self._st_fig = st_fig
        st_canvas = FigureCanvasTkAgg(st_fig, master=tab_speed_theta)
        self._st_canvas = st_canvas
        st_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))

        bm_top = ttk.Frame(tab_brownian, padding=(8, 8, 8, 0))
        bm_top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bm_top, text="Phi source").pack(side=tk.LEFT)
        self._bm_source_box = ttk.Combobox(
            bm_top,
            textvariable=self._brownian_phi_source_var,
            values=["Shifted+fitted phi", "Raw phi"],
            width=18,
            state="readonly",
        )
        self._bm_source_box.pack(side=tk.LEFT, padx=(4, 8))
        self._bm_source_box.bind("<<ComboboxSelected>>", self._on_brownian_source_changed)
        self._bm_test_btn = ttk.Button(
            bm_top,
            text="Test Brownian Motion (selected window)",
            command=self._test_current_spot_brownian,
        )
        self._bm_test_btn.pack(side=tk.LEFT)
        ttk.Label(bm_top, text="Lag range (s)").pack(side=tk.LEFT, padx=(12, 4))
        self._bm_lag_min_entry = ttk.Entry(bm_top, textvariable=self._brownian_lag_min_var, width=8)
        self._bm_lag_min_entry.pack(side=tk.LEFT)
        ttk.Label(bm_top, text="to").pack(side=tk.LEFT, padx=(4, 4))
        self._bm_lag_max_entry = ttk.Entry(bm_top, textvariable=self._brownian_lag_max_var, width=8)
        self._bm_lag_max_entry.pack(side=tk.LEFT)
        self._bm_apply_lag_btn = ttk.Button(bm_top, text="Apply Lag Range", command=self._on_apply_brownian_lag_range)
        self._bm_apply_lag_btn.pack(side=tk.LEFT, padx=(6, 8))
        self._bm_show_refs_chk = ttk.Checkbutton(
            bm_top,
            text="Show slope-1 & slope-2 refs",
            variable=self._brownian_show_refs_var,
            command=self._render_brownian_tab,
        )
        self._bm_show_refs_chk.pack(side=tk.LEFT)
        ttk.Label(bm_top, textvariable=self._brownian_result_var).pack(side=tk.LEFT, padx=(12, 0))

        bm_fig = Figure(figsize=(10, 6), dpi=100)
        self._bm_ax_msd = bm_fig.add_subplot(121)
        self._bm_ax_inc = bm_fig.add_subplot(122)
        self._bm_fig = bm_fig
        bm_canvas = FigureCanvasTkAgg(bm_fig, master=tab_brownian)
        self._bm_canvas = bm_canvas
        bm_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))

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
        self._stop_sphere_animation()
        self._render_sphere_tab()

    def _on_brownian_source_changed(self, event=None) -> None:
        self._brownian_last = None
        self._brownian_result_var.set("Source changed. Press test button for straight-line MSD check.")
        self._render_brownian_tab()

    def _on_apply_sphere_trail(self) -> None:
        _ = self._get_sphere_trail_len(50)
        self._update_sphere_animation_artists()

    def _on_toggle_dual_axis_fit(self) -> None:
        if self._busy:
            return
        self._refresh_sphere_fit_from_current_selection()

    def _refresh_sphere_fit_from_current_selection(self) -> None:
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
        speed = self._get_sphere_speed(1.0)
        dt_ms = max(5, int(round(40.0 / max(0.05, speed))))
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

    def _extract_fps_from_obj(self, obj: object) -> Optional[float]:
        if isinstance(obj, dict):
            for k in ("actual_fps", "resulting_fps", "fps"):
                v = obj.get(k)
                try:
                    f = float(v)
                    if f > 0.0:
                        return f
                except Exception:
                    pass
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
                except Exception:
                    pass

        phi_deg = np.mod(np.degrees(phi_wrapped), 360.0)
        bin_w = self._get_bin_deg(9.0)
        start_deg = self._best_window_start_deg(phi_deg, window_deg=bin_w)
        phi_rel = np.mod(phi_deg - start_deg, 360.0)
        edges = np.arange(0.0, 360.0 + 1e-9, bin_w)
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

    def _analyze_inspection_path(self, path: Path) -> tuple[list[tuple[float, float]], list[float], int]:
        arr, has_frames_dim, frame_count, gray0 = self._load_npy(path)
        side = self._inspection_crop_side(tuple(gray0.shape))
        if side is None:
            raise RuntimeError(f"{path.name}: not a small inspection stack (expected 256 x n, n<50).")
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
        return xy_series, phi_series, frame_count

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
        good_paths: list[Path] = []
        fps_list: list[float] = []
        skipped: list[str] = []
        for i, path in enumerate(paths, start=1):
            self._ui_call(self._status_var.set, f"Processing inspection file {i}/{len(paths)}: {path.name}")
            try:
                xy, phi, _n = self._analyze_inspection_path(path)
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
            good_paths.append(path)
            fps_list.append(self._resolve_fps(path, (int(max(1, len(phi))), 256), inspection_hint=True))

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
        self._spot_idx = 0
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._sphere_fit = None
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
        self._ui_call(self._render_average_distribution)
        self._ui_call(self._render_speed_theta_tab)
        self._ui_call(self._render_brownian_tab)
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
            self._sphere_fit = None
            self._stop_sphere_animation()
            self._phi_sel_t0 = 0.0
            self._phi_sel_t1 = None
            self._spot_names = [path.name for _ in self._spot_centers] if self._analysis_mode == "inspection" else []
            self._avg_selected_spots = [False for _ in self._spot_centers]
            self._speed_theta_selected_spots = [False for _ in self._spot_centers]
            self._speed_theta_points = []
            self._brownian_last = None

            self._ui_call(
                self._status_var.set,
                (
                    f"Loaded {path.name}: inspection mode, single ROI spot analyzed."
                    f" FPS={self.source_fps:.3f}"
                    if self._analysis_mode == "inspection"
                    else f"Loaded {path.name}: {len(self._spot_centers)} filtered spot(s) "
                    f"from {len(self._spot_centers_all)} candidates. FPS={self.source_fps:.3f}"
                ),
            )
            self._ui_call(self._render_all)
            self._ui_call(self._render_average_distribution)
            self._ui_call(self._render_speed_theta_tab)
            self._ui_call(self._render_brownian_tab)
        except Exception as e:
            self._ui_call(messagebox.showerror, "Analysis error", str(e))
            self._ui_call(self._status_var.set, "Analysis failed.")
        finally:
            self._ui_call(self._set_busy, False)

    def _prev_spot(self) -> None:
        if not self._spot_centers:
            return
        self._spot_idx = (self._spot_idx - 1) % len(self._spot_centers)
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1 = None
        self._render_all()
        self._render_average_distribution()
        self._render_brownian_tab()

    def _next_spot(self) -> None:
        if not self._spot_centers:
            return
        self._spot_idx = (self._spot_idx + 1) % len(self._spot_centers)
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        self._phi_sel_t0 = 0.0
        self._phi_sel_t1 = None
        self._render_all()
        self._render_average_distribution()
        self._render_brownian_tab()

    def _analyze_current_spot_distribution(self) -> None:
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
        self._sphere_fit = sphere_fit
        self._sphere_fit_i0 = int(i0)
        self._sphere_fit_i1 = int(i1)
        self._sphere_anim_idx = 0
        self._stop_sphere_animation()
        self._brownian_last = None
        if idx < len(self._speed_theta_selected_spots) and bool(self._speed_theta_selected_spots[idx]):
            self._upsert_speed_theta_point_for_spot(idx, show_error=False)
        self._render_all()
        self._render_average_distribution()
        self._render_speed_theta_tab()
        self._render_brownian_tab()

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
        self._drag_phi_handle = None
        self._drag_phi_source = None
        # Range changed; previous fit is no longer valid for "further analysis".
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        idx = int(self._spot_idx)
        if 0 <= idx < len(self._speed_theta_selected_spots) and bool(self._speed_theta_selected_spots[idx]):
            self._upsert_speed_theta_point_for_spot(idx, show_error=False)
            self._render_speed_theta_tab()
        self._refresh_sphere_fit_from_current_selection()
        self._render_all()
        self._render_brownian_tab()

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
        self._drag_phi_handle = None
        self._drag_phi_source = None
        self._fit_center = None
        self._fit_radius = None
        self._fit_shifted_xy = None
        self._fit_shifted_phi = None
        self._sphere_fit = None
        self._stop_sphere_animation()
        self._brownian_last = None
        idx = int(self._spot_idx)
        if 0 <= idx < len(self._speed_theta_selected_spots) and bool(self._speed_theta_selected_spots[idx]):
            self._upsert_speed_theta_point_for_spot(idx, show_error=False)
            self._render_speed_theta_tab()
        self._refresh_sphere_fit_from_current_selection()
        self._render_all()
        self._render_brownian_tab()

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
        phi_deg = np.mod(np.degrees(phi_shift), 360.0)
        bin_w = self._get_bin_deg(9.0)
        start_deg = self._best_window_start_deg(phi_deg, window_deg=bin_w)
        phi_rel = np.mod(phi_deg - start_deg, 360.0)
        bins_deg = np.arange(0.0, 360.0 + 1e-9, bin_w)
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
        We unwrap XY angle first for continuity, then optionally wrap physical phi to [0, 2pi).
        """
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return np.asarray([], dtype=np.float64)
        xy_ang = np.arctan2(arr[:, 1], arr[:, 0])  # [-pi, pi]
        xy_ang_unwrapped = np.unwrap(xy_ang)  # continuous XY phase
        phi = 0.5 * xy_ang_unwrapped  # physical phase
        if wrap_2pi:
            phi = np.mod(phi, 2.0 * np.pi)
        return phi

    def _render_all(self) -> None:
        n = len(self._spot_centers)
        self._spot_status_var.set(f"Spot {self._spot_idx + 1} / {n}" if n > 0 else "Spot 0 / 0")
        self._sync_include_checkbox()
        if n <= 0:
            self._current_file_var.set("File: -")
        else:
            idx_name = max(0, min(int(self._spot_idx), n - 1))
            if self._spot_names and idx_name < len(self._spot_names):
                self._current_file_var.set(f"File: {self._spot_names[idx_name]}")
            elif self.source_path is not None:
                self._current_file_var.set(f"File: {self.source_path.name}")
            else:
                self._current_file_var.set("File: -")

        self._ax_xy.clear()
        self._ax_phi.clear()
        self._ax_shift.clear()
        self._ax_hist.clear()

        if n <= 0:
            self._ax_xy.set_title("X/Y scatter")
            self._ax_phi.set_title("phi(t)")
            self._ax_shift.set_title("Shifted XY after circle fit")
            self._ax_hist.set_title("Shifted phi distribution")
            self._fig.tight_layout()
            self._canvas.draw_idle()
            self._render_sphere_tab()
            return

        idx = max(0, min(int(self._spot_idx), n - 1))
        xy = np.asarray(self._spot_xy_series[idx], dtype=np.float64)
        phi = self._physical_phi_from_xy_series(xy, wrap_2pi=True)

        if xy.size > 0:
            self._ax_xy.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.8)
        self._ax_xy.axhline(0.0, color="0.8", lw=1)
        self._ax_xy.axvline(0.0, color="0.8", lw=1)
        self._ax_xy.set_xlim(-1.0, 1.0)
        self._ax_xy.set_ylim(-1.0, 1.0)
        self._ax_xy.set_xlabel("X")
        self._ax_xy.set_ylabel("Y")
        self._ax_xy.set_title("X/Y scatter")

        phi_t = self._fit_shifted_phi if self._fit_shifted_phi is not None else phi
        if phi_t.size > 0:
            t = np.arange(phi_t.size, dtype=np.float64) / max(1e-9, float(self._current_fps()))
            phi_plot = phi_t.astype(np.float64, copy=True)
            if phi_plot.size >= 2:
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
        self._ax_phi.set_ylabel("phi (rad, 0..2pi)")
        self._ax_phi.set_ylim(0.0, 2.0 * np.pi)
        self._ax_phi.set_title(
            "phi(t) after fit+shift" if self._fit_shifted_phi is not None
            else "phi(t) continuity-unwrapped, wrapped to 0..2pi"
        )

        if self._fit_shifted_xy is not None and self._fit_shifted_phi is not None:
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

            phi_deg = np.mod(np.degrees(self._fit_shifted_phi), 360.0)
            bin_w = self._get_bin_deg(9.0)
            start_deg = self._best_window_start_deg(phi_deg, window_deg=bin_w)
            phi_rel = np.mod(phi_deg - start_deg, 360.0)
            bins_deg = np.arange(0.0, 360.0 + 1e-9, bin_w)
            self._ax_hist.hist(phi_rel, bins=bins_deg, color="tab:blue", alpha=0.85)
            counts, edges = np.histogram(phi_rel, bins=bins_deg)
            if counts.size > 0 and int(np.max(counts)) > 0:
                i_peak = int(np.argmax(counts))
                peak_center_deg = 0.5 * (float(edges[i_peak]) + float(edges[i_peak + 1]))
                self._ax_hist.axvline(
                    peak_center_deg,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.95,
                )
                guide_angles = sorted({(peak_center_deg + 36.0 * k) % 360.0 for k in range(10)})
                for ang in guide_angles:
                    self._ax_hist.axvline(
                        ang,
                        color="0.35",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.8,
                    )
            self._ax_hist.set_xlim(0.0, 360.0)
            self._ax_hist.set_xlabel("shifted phi (deg, 0 deg = densest-bin window start)")
            self._ax_hist.set_ylabel("count")
            self._ax_hist.set_title(f"Shifted phi distribution ({bin_w:g} deg bins, aligned to densest window)")

            if self._fit_center is not None and self._fit_radius is not None:
                cx, cy = self._fit_center
                self._status_var.set(
                    f"Spot {idx + 1}: fitted circle center=({cx:.4f}, {cy:.4f}), radius={self._fit_radius:.4f}"
                )
        else:
            self._ax_shift.set_title("Shifted XY after circle fit")
            self._ax_hist.set_title("Shifted phi distribution")
            self._ax_shift.text(0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_shift.transAxes)
            self._ax_hist.text(0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_hist.transAxes)

        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._render_sphere_tab()

    def _render_sphere_tab(self) -> None:
        self._ax_sphere_3d.clear()
        self._ax_sphere_plane.clear()
        self._ax_sphere_phi.clear()
        self._ax_sphere_hist.clear()
        self._sphere_trail_artist = None
        self._sphere_head_artist = None
        self._sphere_plane_trail_artist = None
        self._sphere_plane_head_artist = None
        self._phi_sel_line0_sphere = None
        self._phi_sel_line1_sphere = None
        self._phi_sel_span_sphere = None

        fit = self._sphere_fit
        if fit is None:
            self._ax_sphere_3d.set_title("Unit-sphere trajectory")
            self._ax_sphere_plane.set_title("Projection in fitted plane")
            self._ax_sphere_phi.set_title("phi_fromaxis(t)")
            self._ax_sphere_hist.set_title("phi_fromaxis distribution")
            self._ax_sphere_3d.text2D(0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_sphere_3d.transAxes)
            self._ax_sphere_plane.text(
                0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_sphere_plane.transAxes
            )
            self._ax_sphere_phi.text(
                0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_sphere_phi.transAxes
            )
            self._ax_sphere_hist.text(
                0.5, 0.5, "Press analysis button", ha="center", va="center", transform=self._ax_sphere_hist.transAxes
            )
            self._sphere_info_var.set("Press analysis button to reconstruct unit-sphere trajectory.")
            self._sphere_fig.tight_layout()
            self._sphere_canvas.draw_idle()
            return

        u = np.asarray(fit.get("u", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 3 or u.shape[0] == 0:
            self._sphere_info_var.set("No unit-sphere points for current selection.")
            self._sphere_fig.tight_layout()
            self._sphere_canvas.draw_idle()
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

        # Projection in plane perpendicular to fitted axis.
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

        # phi_fromaxis histogram aligned to densest bin window.
        self._ax_sphere_hist.hist(phi_rel_deg, bins=hist_edges, color="tab:green", alpha=0.85)
        counts, edges = np.histogram(phi_rel_deg, bins=hist_edges)
        if counts.size > 0 and int(np.max(counts)) > 0:
            i_peak = int(np.argmax(counts))
            peak_center = 0.5 * (float(edges[i_peak]) + float(edges[i_peak + 1]))
            self._ax_sphere_hist.axvline(peak_center, color="black", linestyle=":", linewidth=1.4, alpha=0.95)
            guide_angles = sorted({(peak_center + 36.0 * k) % 360.0 for k in range(10)})
            for ang in guide_angles:
                self._ax_sphere_hist.axvline(
                    ang,
                    color="0.35",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
        self._ax_sphere_hist.set_xlim(0.0, 360.0)
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
        dual_used = bool(fit.get("axis2_k", None) is not None)
        switches = int(fit.get("switch_count", 0))
        fit_cells2 = int(fit.get("fit_cells2_used", 0))
        fit_pts2 = int(fit.get("fit_points2_total", 0))
        self._sphere_info_var.set(
            f"Theta model={fit['theta_model']} | alpha={alpha_deg:.2f} deg beta={beta_deg:.2f} deg "
            f"gamma={gamma_deg:.2f} deg | clipped-r={clipped} | fit1 bins={fit_cells}/{fit_pts}"
            + (
                f" | fit2 bins={fit_cells2}/{fit_pts2} | switches={switches}"
                if dual_used
                else ""
            )
        )
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
        if self._sphere_plane_trail_artist is None or self._sphere_plane_head_artist is None:
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
        self._sphere_plane_trail_artist.set_offsets(seg_p if seg_p.size else np.zeros((0, 2), dtype=np.float64))
        self._sphere_plane_head_artist.set_data([p1[idx]], [p2[idx]])
        self._sphere_canvas.draw_idle()

    def _sync_include_checkbox(self) -> None:
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
