from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = SCRIPT_DIR / "venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = SCRIPT_DIR / "venv" / "bin" / "python"
if (
    VENV_PYTHON.exists()
    and Path(sys.executable).resolve() != VENV_PYTHON.resolve()
    and os.environ.get("POLARCAM_SKIP_VENV_REEXEC") != "1"
):
    os.environ["POLARCAM_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if not os.environ.get("MPLCONFIGDIR"):
    config_dir = SCRIPT_DIR / ".mplconfig"
    config_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from nptdms import TdmsFile


DEFAULT_TDMS = Path(
    r"C:\labview software\speaker stage frequency\magnetcoilmagnetically attatched 5k analysis latest attachment inferred on off\bfm220720261652-2147483648.tdms"
)
DEFAULT_MAPPING = (3, 0, 1, 2)  # TDMS channel indices used as I0, I90, I45, I135.
POL_LABELS = ("I0", "I90", "I45", "I135")
BFM_25082026_FULL_INVERSE = np.array(
    [
        [2.56410256, 0.0, 0.0, 0.0],
        [0.0, 2.65305220, 0.0, 0.0],
        [-0.04047189, 0.04187587, 1.87478350, -0.00045528],
        [-0.04045247, 0.04185578, -0.00046721, 1.82373830],
    ],
    dtype=np.float64,
)
THETA_MODELS = {
    "water_1p285": {
        "a": 0.2511031870432621,
        "b": 0.7101695268458184,
        "c": 0.8499508052842775,
        "r_max": 0.8355419189329347,
    },
    "water r_max (0.9208)": {
        # Newer annular-water theta(r) curve from
        # theta_r_curve_parameters/water1p33_glycerol1p47_naout1p3_nain0p39/theta_r_abc_values.csv
        "a": 0.9134121735581628,
        "b": 0.9511965525903582,
        "c": 0.11957056359722706,
        "r_max": 0.9208252165082128,
    },
    "glycerol99p5_1p3": {
        "a": 0.204343183995232,
        "b": 0.592759327477552,
        "c": 0.65681669522379,
        "r_max": 0.902472991000307,
    },
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


def read_tdms_4ch(path: Path) -> tuple[np.ndarray, list[str], float]:
    tdms = TdmsFile.read(path)
    channels = [channel for group in tdms.groups() for channel in group.channels()]
    if len(channels) < 4:
        raise RuntimeError(f"{path} has {len(channels)} channel(s); need at least 4.")
    channels = channels[:4]
    raw = np.column_stack([np.asarray(channel[:], dtype=np.float64) for channel in channels])
    names = [str(channel.name) for channel in channels]
    increment = channels[0].properties.get("wf_increment", None)
    sample_rate = float(1.0 / increment) if increment else float("nan")
    return raw, names, sample_rate


def block_average(values: np.ndarray, block_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    block_size = max(1, int(block_size))
    if block_size == 1 or n <= 1:
        return arr.copy()
    return np.vstack([arr[i : i + block_size].mean(axis=0) for i in range(0, n, block_size)])


def xy_from_intensities(intensities: np.ndarray) -> np.ndarray:
    arr = np.asarray(intensities, dtype=np.float64)
    eps = 1e-12
    x = (arr[:, 0] - arr[:, 1]) / (arr[:, 0] + arr[:, 1] + eps)
    y = (arr[:, 2] - arr[:, 3]) / (arr[:, 2] + arr[:, 3] + eps)
    return np.column_stack((x, y))


def apply_full_inverse_correction(intensities: np.ndarray) -> np.ndarray:
    arr = np.asarray(intensities, dtype=np.float64)
    return arr @ BFM_25082026_FULL_INVERSE.T


def parse_mapping(text: str) -> tuple[int, int, int, int]:
    mapping = tuple(int(part.strip()) for part in text.split(","))
    if sorted(mapping) != [0, 1, 2, 3]:
        raise ValueError("Mapping must be a permutation like 2,0,1,3.")
    return mapping


def theta_from_radius(r: np.ndarray, model: str) -> tuple[np.ndarray, int]:
    params = THETA_MODELS.get(str(model).strip().lower(), THETA_MODELS["hole+fresnel"])
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


def physical_phi_from_xy_series(xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    xy_angle = np.unwrap(np.arctan2(arr[:, 1], arr[:, 0]))
    return np.mod(0.5 * xy_angle, 2.0 * np.pi)


def unit_sphere_from_xy(xy: np.ndarray, theta_model: str) -> tuple[np.ndarray, int]:
    arr = np.asarray(xy, dtype=np.float64)
    r = np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))
    theta, clipped = theta_from_radius(r, theta_model)
    phi = physical_phi_from_xy_series(arr)
    sin_theta = np.sin(theta)
    u = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)))
    return u.astype(np.float64, copy=False), clipped


def fit_origin_arc_from_xy(xy: np.ndarray, theta_model: str, time_s: np.ndarray) -> dict | None:
    u, clipped = unit_sphere_from_xy(xy, theta_model)
    good = np.all(np.isfinite(u), axis=1)
    u = u[good]
    t = np.asarray(time_s, dtype=np.float64)[good]
    if u.shape[0] < 3:
        return None
    norm = np.linalg.norm(u, axis=1)
    keep = norm > 1e-12
    u = u[keep] / norm[keep][:, None]
    t = t[keep]
    if u.shape[0] < 3:
        return None

    moment = (u.T @ u) / float(u.shape[0])
    _vals, vecs = np.linalg.eigh(moment)
    normal = vecs[:, 0]
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
    angle = np.unwrap(np.arctan2(p2, p1))
    angle_ref = float(np.nanpercentile(angle, 99.0))
    angle_p1 = float(np.nanpercentile(angle, 1.0))
    opening_deg = np.degrees(angle_ref - angle)
    arc_opening_99_1_deg = float(np.degrees(angle_ref - angle_p1))
    plane_dist = np.clip(np.abs(u @ normal), 0.0, 1.0)
    residual_deg = np.degrees(np.arcsin(plane_dist))
    sin_90_minus_theta_half = np.sin(np.radians((90.0 - opening_deg) / 2.0))
    return {
        "u": u,
        "time_s": t,
        "opening_deg": opening_deg.astype(np.float64, copy=False),
        "opening_reference_percentile": 99.0,
        "opening_reference_angle_deg": float(np.degrees(angle_ref)),
        "arc_opening_99_1_deg": arc_opening_99_1_deg,
        "sin_90_minus_theta_half": sin_90_minus_theta_half.astype(np.float64, copy=False),
        "residual_deg": residual_deg.astype(np.float64, copy=False),
        "r_clipped_count": int(clipped),
    }


class ManualScalingXYApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TDMS Manual XY")
        self.root.geometry("1220x780")

        self.raw: np.ndarray | None = None
        self.channel_names: list[str] = []
        self.sample_rate_hz = float("nan")
        self.loaded_path: Path | None = None

        self.path_var = tk.StringVar(value=str(DEFAULT_TDMS))
        self.mapping_var = tk.StringVar(value=",".join(str(i) for i in DEFAULT_MAPPING))
        self.average_var = tk.StringVar(value="1")
        self.start_frame_var = tk.StringVar(value="0")
        self.end_frame_var = tk.StringVar(value="20000")
        self.sin_bin_width_var = tk.StringVar(value="0.025")
        self.theta_model_var = tk.StringVar(value="water_1p285")
        self.apply_correction_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Load a TDMS file to begin.")
        self.arc_info_var = tk.StringVar(value="Arc opening rulers: load a TDMS file.")
        self.sin_info_var = tk.StringVar(value="90-theta rulers: load a TDMS file.")
        self._arc_ruler_values: list[float | None] = [None, None]
        self._sin_ruler_values: list[float | None] = [None, None]
        self._dragging_arc_ruler: int | None = None
        self._dragging_sin_ruler: int | None = None
        self._arc_ruler_lines: list[object | None] = [None, None]
        self._sin_ruler_lines: list[object | None] = [None, None]
        self._last_arc_opening = np.asarray([], dtype=np.float64)
        self._last_sin_values = np.asarray([], dtype=np.float64)

        self._build_ui()
        if DEFAULT_TDMS.exists():
            self.load_tdms(DEFAULT_TDMS)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="TDMS").grid(row=0, column=0, sticky="w")
        path_entry = ttk.Entry(top, textvariable=self.path_var)
        path_entry.grid(row=0, column=1, columnspan=7, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="Browse", command=self.browse_tdms).grid(row=0, column=8, padx=(0, 6))
        ttk.Button(top, text="Load", command=self.load_current_path).grid(row=0, column=9)

        ttk.Label(top, text="Mapping [I0,I90,I45,I135]").grid(row=1, column=0, sticky="w", pady=(8, 0))
        mapping_entry = ttk.Entry(top, textvariable=self.mapping_var, width=12)
        mapping_entry.grid(row=1, column=1, sticky="w", padx=(6, 16), pady=(8, 0))
        ttk.Label(top, text="Start frame").grid(row=1, column=2, sticky="w", pady=(8, 0))
        start_entry = ttk.Entry(top, textvariable=self.start_frame_var, width=10)
        start_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(top, text="End frame").grid(row=1, column=4, sticky="w", pady=(8, 0))
        end_entry = ttk.Entry(top, textvariable=self.end_frame_var, width=10)
        end_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(top, text="Averaging number").grid(row=1, column=6, sticky="w", pady=(8, 0))
        avg_entry = ttk.Entry(top, textvariable=self.average_var, width=8)
        avg_entry.grid(row=1, column=7, sticky="w", padx=(6, 18), pady=(8, 0))

        ttk.Button(top, text="Update Plot", command=self.update_plot).grid(row=1, column=9, sticky="e", pady=(8, 0))
        correction_box = ttk.Checkbutton(
            top,
            text="Apply BFM 25082026 A^-1 D^-1 correction",
            variable=self.apply_correction_var,
        )
        correction_box.grid(row=2, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Theta model").grid(row=3, column=0, sticky="w", pady=(8, 0))
        theta_box = ttk.Combobox(
            top,
            textvariable=self.theta_model_var,
            values=tuple(THETA_MODELS.keys()),
            width=22,
            state="readonly",
        )
        theta_box.grid(row=3, column=1, sticky="w", padx=(6, 16), pady=(8, 0))
        ttk.Label(top, text="Sin hist bin").grid(row=3, column=2, sticky="w", pady=(8, 0))
        bin_entry = ttk.Entry(top, textvariable=self.sin_bin_width_var, width=8)
        bin_entry.grid(row=3, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        top.columnconfigure(1, weight=1)

        info = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 0))
        info.pack(side=tk.TOP, fill=tk.X)

        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tab_xy = ttk.Frame(notebook)
        tab_hemi = ttk.Frame(notebook)
        tab_arc = ttk.Frame(notebook)
        tab_sin = ttk.Frame(notebook)
        notebook.add(tab_xy, text="XY / Radius")
        notebook.add(tab_hemi, text="Hemisphere")
        notebook.add(tab_arc, text="Arc Opening")
        notebook.add(tab_sin, text="sin((90-theta)/2)")

        self.figure = Figure(figsize=(11, 6.8), dpi=100)
        self.ax_xy = self.figure.add_subplot(1, 2, 1)
        self.ax_r = self.figure.add_subplot(2, 2, 2)
        self.ax_t = self.figure.add_subplot(2, 2, 4)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=tab_xy)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, tab_xy, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.hemi_figure = Figure(figsize=(11, 6.8), dpi=100)
        self.ax_hemi = self.hemi_figure.add_subplot(1, 1, 1, projection="3d")
        self.hemi_figure.tight_layout()
        self.hemi_canvas = FigureCanvasTkAgg(self.hemi_figure, master=tab_hemi)
        self.hemi_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        hemi_toolbar = NavigationToolbar2Tk(self.hemi_canvas, tab_hemi, pack_toolbar=False)
        hemi_toolbar.update()
        hemi_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(tab_arc, textvariable=self.arc_info_var, anchor="w", padding=(8, 4)).pack(side=tk.TOP, fill=tk.X)
        self.arc_figure = Figure(figsize=(11, 6.8), dpi=100)
        self.ax_arc_time = self.arc_figure.add_subplot(2, 1, 1)
        self.ax_arc_hist = self.arc_figure.add_subplot(2, 1, 2)
        self.arc_figure.tight_layout()
        self.arc_canvas = FigureCanvasTkAgg(self.arc_figure, master=tab_arc)
        self.arc_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        arc_toolbar = NavigationToolbar2Tk(self.arc_canvas, tab_arc, pack_toolbar=False)
        arc_toolbar.update()
        arc_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.arc_canvas.mpl_connect("button_press_event", self._on_arc_ruler_press)
        self.arc_canvas.mpl_connect("motion_notify_event", self._on_arc_ruler_motion)
        self.arc_canvas.mpl_connect("button_release_event", self._on_arc_ruler_release)

        ttk.Label(tab_sin, textvariable=self.sin_info_var, anchor="w", padding=(8, 4)).pack(side=tk.TOP, fill=tk.X)
        self.sin_figure = Figure(figsize=(11, 6.8), dpi=100)
        self.ax_sin_time = self.sin_figure.add_subplot(2, 1, 1)
        self.ax_sin_hist = self.sin_figure.add_subplot(2, 1, 2)
        self.sin_figure.tight_layout()
        self.sin_canvas = FigureCanvasTkAgg(self.sin_figure, master=tab_sin)
        self.sin_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sin_toolbar = NavigationToolbar2Tk(self.sin_canvas, tab_sin, pack_toolbar=False)
        sin_toolbar.update()
        sin_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sin_canvas.mpl_connect("button_press_event", self._on_sin_ruler_press)
        self.sin_canvas.mpl_connect("motion_notify_event", self._on_sin_ruler_motion)
        self.sin_canvas.mpl_connect("button_release_event", self._on_sin_ruler_release)

        for var in [
            self.mapping_var,
            self.average_var,
            self.start_frame_var,
            self.end_frame_var,
            self.sin_bin_width_var,
            self.theta_model_var,
            self.apply_correction_var,
        ]:
            var.trace_add("write", self._schedule_update)

    def _schedule_update(self, *_args: object) -> None:
        if self.raw is None:
            return
        if hasattr(self, "_after_id"):
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(350, self.update_plot)

    def _arc_ruler_transform_value(self, theta_deg: float) -> float:
        return float(np.sin(np.radians((90.0 - float(theta_deg)) / 2.0)))

    def _set_default_or_clamped_arc_rulers(self, opening_deg: np.ndarray) -> None:
        vals = np.asarray(opening_deg, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._arc_ruler_values = [None, None]
            return
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        p1, p99 = np.nanpercentile(vals, [1.0, 99.0])
        defaults = [float(p1), float(p99)]
        for i in range(2):
            current = self._arc_ruler_values[i]
            if current is None or not np.isfinite(float(current)):
                self._arc_ruler_values[i] = defaults[i]
            else:
                self._arc_ruler_values[i] = max(lo, min(hi, float(current)))

    def _update_arc_ruler_artists(self) -> None:
        vals = self._arc_ruler_values
        colors = ("tab:red", "black")
        labels = ("A", "B")
        for i, value in enumerate(vals):
            line = self._arc_ruler_lines[i]
            if value is None:
                continue
            if line is None:
                self._arc_ruler_lines[i] = self.ax_arc_hist.axvline(
                    float(value), color=colors[i], lw=1.8, linestyle="--", label=f"ruler {labels[i]}"
                )
            else:
                line.set_xdata([float(value), float(value)])
        self._update_arc_ruler_info()

    def _update_arc_ruler_info(self) -> None:
        vals = self._arc_ruler_values
        opening = np.asarray(self._last_arc_opening, dtype=np.float64)
        opening = opening[np.isfinite(opening)]
        if opening.size == 0 or vals[0] is None or vals[1] is None:
            self.arc_info_var.set("Arc opening rulers: no fitted arc.")
            return
        a = float(vals[0])
        b = float(vals[1])
        delta = abs(b - a)
        p1, p99 = np.nanpercentile(opening, [1.0, 99.0])
        self.arc_info_var.set(
            " | ".join(
                [
                    f"theta A={a:.4g} deg",
                    f"theta B={b:.4g} deg",
                    f"|B-A|={delta:.4g} deg",
                    f"occupied 99-1={float(p99 - p1):.4g} deg",
                    f"sin((90-A)/2)={self._arc_ruler_transform_value(a):.5f}",
                    f"sin((90-B)/2)={self._arc_ruler_transform_value(b):.5f}",
                ]
            )
        )

    def _on_arc_ruler_press(self, event) -> None:
        if event.inaxes is not self.ax_arc_hist or event.xdata is None:
            return
        opening = np.asarray(self._last_arc_opening, dtype=np.float64)
        opening = opening[np.isfinite(opening)]
        if opening.size == 0:
            return
        self._set_default_or_clamped_arc_rulers(opening)
        vals = [float(v) if v is not None else float(event.xdata) for v in self._arc_ruler_values]
        idx = int(np.argmin([abs(float(event.xdata) - vals[0]), abs(float(event.xdata) - vals[1])]))
        lo = float(np.min(opening))
        hi = float(np.max(opening))
        self._arc_ruler_values[idx] = max(lo, min(hi, float(event.xdata)))
        self._dragging_arc_ruler = idx
        self._update_arc_ruler_artists()
        self.arc_canvas.draw_idle()

    def _on_arc_ruler_motion(self, event) -> None:
        if self._dragging_arc_ruler is None or event.inaxes is not self.ax_arc_hist or event.xdata is None:
            return
        opening = np.asarray(self._last_arc_opening, dtype=np.float64)
        opening = opening[np.isfinite(opening)]
        if opening.size == 0:
            return
        lo = float(np.min(opening))
        hi = float(np.max(opening))
        self._arc_ruler_values[self._dragging_arc_ruler] = max(lo, min(hi, float(event.xdata)))
        self._update_arc_ruler_artists()
        self.arc_canvas.draw_idle()

    def _on_arc_ruler_release(self, _event) -> None:
        self._dragging_arc_ruler = None

    def _set_default_or_clamped_sin_rulers(self, sin_values: np.ndarray) -> None:
        vals = np.asarray(sin_values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._sin_ruler_values = [None, None]
            return
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        p1, p99 = np.nanpercentile(vals, [1.0, 99.0])
        defaults = [float(p1), float(p99)]
        for i in range(2):
            current = self._sin_ruler_values[i]
            if current is None or not np.isfinite(float(current)):
                self._sin_ruler_values[i] = defaults[i]
            else:
                self._sin_ruler_values[i] = max(lo, min(hi, float(current)))

    def _update_sin_ruler_artists(self) -> None:
        vals = self._sin_ruler_values
        colors = ("tab:red", "black")
        labels = ("A", "B")
        for i, value in enumerate(vals):
            line = self._sin_ruler_lines[i]
            if value is None:
                continue
            if line is None:
                self._sin_ruler_lines[i] = self.ax_sin_hist.axvline(
                    float(value), color=colors[i], lw=1.8, linestyle="--", label=f"ruler {labels[i]}"
                )
            else:
                line.set_xdata([float(value), float(value)])
        self._update_sin_ruler_info()

    def _update_sin_ruler_info(self) -> None:
        vals = self._sin_ruler_values
        sin_values = np.asarray(self._last_sin_values, dtype=np.float64)
        sin_values = sin_values[np.isfinite(sin_values)]
        if sin_values.size == 0 or vals[0] is None or vals[1] is None:
            self.sin_info_var.set("90-theta rulers: no fitted transformed distribution.")
            return
        a = float(vals[0])
        b = float(vals[1])
        delta = abs(b - a)
        p1, p99 = np.nanpercentile(sin_values, [1.0, 99.0])
        self.sin_info_var.set(
            " | ".join(
                [
                    f"A={a:.5f}",
                    f"B={b:.5f}",
                    f"|B-A|={delta:.5f}",
                    f"occupied 99-1={float(p99 - p1):.5f}",
                ]
            )
        )

    def _on_sin_ruler_press(self, event) -> None:
        if event.inaxes is not self.ax_sin_hist or event.xdata is None:
            return
        vals_arr = np.asarray(self._last_sin_values, dtype=np.float64)
        vals_arr = vals_arr[np.isfinite(vals_arr)]
        if vals_arr.size == 0:
            return
        self._set_default_or_clamped_sin_rulers(vals_arr)
        vals = [float(v) if v is not None else float(event.xdata) for v in self._sin_ruler_values]
        idx = int(np.argmin([abs(float(event.xdata) - vals[0]), abs(float(event.xdata) - vals[1])]))
        lo = float(np.min(vals_arr))
        hi = float(np.max(vals_arr))
        self._sin_ruler_values[idx] = max(lo, min(hi, float(event.xdata)))
        self._dragging_sin_ruler = idx
        self._update_sin_ruler_artists()
        self.sin_canvas.draw_idle()

    def _on_sin_ruler_motion(self, event) -> None:
        if self._dragging_sin_ruler is None or event.inaxes is not self.ax_sin_hist or event.xdata is None:
            return
        vals_arr = np.asarray(self._last_sin_values, dtype=np.float64)
        vals_arr = vals_arr[np.isfinite(vals_arr)]
        if vals_arr.size == 0:
            return
        lo = float(np.min(vals_arr))
        hi = float(np.max(vals_arr))
        self._sin_ruler_values[self._dragging_sin_ruler] = max(lo, min(hi, float(event.xdata)))
        self._update_sin_ruler_artists()
        self.sin_canvas.draw_idle()

    def _on_sin_ruler_release(self, _event) -> None:
        self._dragging_sin_ruler = None

    def browse_tdms(self) -> None:
        initial_dir = Path(self.path_var.get()).expanduser().parent
        filename = filedialog.askopenfilename(
            title="Choose TDMS file",
            initialdir=str(initial_dir if initial_dir.exists() else Path.home()),
            filetypes=(("TDMS files", "*.tdms"), ("All files", "*.*")),
        )
        if filename:
            self.path_var.set(filename)
            self.load_current_path()

    def load_current_path(self) -> None:
        self.load_tdms(Path(self.path_var.get()).expanduser())

    def load_tdms(self, path: Path) -> None:
        try:
            raw, channel_names, sample_rate_hz = read_tdms_4ch(path)
        except Exception as exc:
            messagebox.showerror("TDMS load failed", str(exc))
            return

        self.raw = raw
        self.channel_names = channel_names
        self.sample_rate_hz = sample_rate_hz
        self.loaded_path = path
        self.path_var.set(str(path))
        self.start_frame_var.set("0")
        self.end_frame_var.set(str(min(20000, int(raw.shape[0]))))
        self.update_plot()

    def _current_inputs(self) -> tuple[tuple[int, int, int, int], int, int, int, float, bool]:
        mapping = parse_mapping(self.mapping_var.get())
        average_n = int(float(self.average_var.get()))
        if average_n <= 0:
            raise ValueError("Averaging number must be positive.")
        total = int(self.raw.shape[0]) if self.raw is not None else 0
        start = int(float(self.start_frame_var.get() or "0"))
        start = max(0, min(start, total))
        end_text = self.end_frame_var.get().strip()
        end = total if end_text == "" else int(float(end_text))
        end = max(0, min(end, total))
        if end <= start:
            raise ValueError("Selected frame range is empty.")
        bin_width = float(self.sin_bin_width_var.get())
        if not np.isfinite(bin_width) or bin_width <= 0.0:
            raise ValueError("Sin histogram bin width must be positive.")
        return mapping, average_n, start, end, bin_width, bool(self.apply_correction_var.get())

    def update_plot(self) -> None:
        if self.raw is None:
            return

        try:
            mapping, average_n, start, end, sin_bin_width, apply_correction = self._current_inputs()
        except Exception as exc:
            self.status_var.set(f"Input error: {exc}")
            return

        selected_raw = self.raw[start:end, :]
        mapped = selected_raw[:, mapping]
        averaged = block_average(mapped, average_n)
        corrected = apply_full_inverse_correction(averaged) if apply_correction else averaged
        xy = xy_from_intensities(corrected)
        r2 = np.sum(xy**2, axis=1)
        selected_frames = int(end - start)
        effective_rate = self.sample_rate_hz * (averaged.shape[0] / max(1, selected_frames))
        t0 = float(start) / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0
        t = t0 + (np.arange(averaged.shape[0], dtype=np.float64) / effective_rate if effective_rate > 0 else np.arange(averaged.shape[0]))

        self.ax_xy.clear()
        color = np.linspace(0.0, 1.0, xy.shape[0])
        self.ax_xy.scatter(xy[:, 0], xy[:, 1], c=color, cmap="viridis", s=2.5, alpha=0.8, linewidths=0)
        self.ax_xy.plot(xy[:, 0], xy[:, 1], color="0.2", lw=0.25, alpha=0.22)
        self.ax_xy.scatter([float(np.mean(xy[:, 0]))], [float(np.mean(xy[:, 1]))], marker="x", s=90, color="red", linewidths=2)
        self.ax_xy.axhline(0.0, color="0.82", lw=1)
        self.ax_xy.axvline(0.0, color="0.82", lw=1)
        self.ax_xy.set_xlim(-1.05, 1.05)
        self.ax_xy.set_ylim(-1.05, 1.05)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlabel("X = (I0 - I90) / (I0 + I90)")
        self.ax_xy.set_ylabel("Y = (I45 - I135) / (I45 + I135)")
        self.ax_xy.set_title("Corrected XY" if apply_correction else "Raw XY")
        self.ax_xy.grid(alpha=0.2)

        self.ax_r.clear()
        self.ax_r.plot(t, r2, color="tab:blue", lw=0.7)
        self.ax_r.axhline(float(np.mean(r2)), color="0.2", ls="--", lw=1)
        self.ax_r.set_ylabel("X^2 + Y^2")
        self.ax_r.set_title("Radius Squared")
        self.ax_r.grid(alpha=0.2)

        self.ax_t.clear()
        self.ax_t.plot(t, xy[:, 0], color="tab:blue", lw=0.7, label="X")
        self.ax_t.plot(t, xy[:, 1], color="tab:orange", lw=0.7, label="Y")
        self.ax_t.axhline(0.0, color="0.82", lw=1)
        self.ax_t.set_xlabel("time (s)")
        self.ax_t.set_ylabel("XY")
        self.ax_t.set_title("X/Y Against Time")
        self.ax_t.grid(alpha=0.2)
        self.ax_t.legend(loc="upper right")

        self.ax_hemi.clear()
        u, hemi_clipped = unit_sphere_from_xy(xy, self.theta_model_var.get())
        good_u = np.all(np.isfinite(u), axis=1)
        u_plot = u[good_u]
        color_u = color[good_u]
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 120)
        theta_grid = np.linspace(0.0, 0.5 * np.pi, 60)
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        xh = np.sin(theta_mesh) * np.cos(phi_mesh)
        yh = np.sin(theta_mesh) * np.sin(phi_mesh)
        zh = np.cos(theta_mesh)
        self.ax_hemi.plot_wireframe(xh, yh, zh, rstride=6, cstride=10, color="0.82", linewidth=0.5)
        if u_plot.size > 0:
            self.ax_hemi.scatter(
                u_plot[:, 0],
                u_plot[:, 1],
                u_plot[:, 2],
                c=color_u,
                cmap="viridis",
                s=3.0,
                alpha=0.85,
                linewidths=0,
            )
            u_mean = np.mean(u_plot, axis=0)
            self.ax_hemi.scatter(
                [float(u_mean[0])],
                [float(u_mean[1])],
                [float(u_mean[2])],
                marker="x",
                s=110,
                color="red",
                linewidths=2,
            )
        self.ax_hemi.set_xlim(-1.0, 1.0)
        self.ax_hemi.set_ylim(-1.0, 1.0)
        self.ax_hemi.set_zlim(0.0, 1.0)
        self.ax_hemi.set_box_aspect((1.0, 1.0, 0.7))
        self.ax_hemi.set_xlabel("Ux")
        self.ax_hemi.set_ylabel("Uy")
        self.ax_hemi.set_zlabel("Uz")
        self.ax_hemi.set_title(f"Hemisphere points | clipped-r={hemi_clipped}")

        arc = fit_origin_arc_from_xy(xy, self.theta_model_var.get(), t)
        self.ax_arc_time.clear()
        self.ax_arc_hist.clear()
        self._arc_ruler_lines = [None, None]
        self.ax_sin_time.clear()
        self.ax_sin_hist.clear()
        self._sin_ruler_lines = [None, None]
        if arc is None:
            self._last_arc_opening = np.asarray([], dtype=np.float64)
            self._last_sin_values = np.asarray([], dtype=np.float64)
            self.ax_arc_time.set_title("Arc angle vs time")
            self.ax_arc_time.text(
                0.5,
                0.5,
                "Need at least 3 valid unit-sphere points",
                ha="center",
                va="center",
                transform=self.ax_arc_time.transAxes,
            )
            self.ax_arc_hist.set_title("Arc angle distribution")
            self.ax_arc_hist.text(0.5, 0.5, "No fitted arc", ha="center", va="center", transform=self.ax_arc_hist.transAxes)
            self.arc_info_var.set("Arc opening rulers: no fitted arc.")
            self.ax_sin_time.set_title("sin((90-theta)/2) vs time")
            self.ax_sin_time.text(0.5, 0.5, "Need at least 3 valid unit-sphere points", ha="center", va="center", transform=self.ax_sin_time.transAxes)
            self.ax_sin_hist.set_title("sin((90-theta)/2) distribution")
            self.ax_sin_hist.text(0.5, 0.5, "No fitted arc", ha="center", va="center", transform=self.ax_sin_hist.transAxes)
            self.sin_info_var.set("90-theta rulers: no fitted transformed distribution.")
            sin_summary = "sin((90-theta)/2)=n/a"
            opening_summary = "arc opening 99-1=n/a"
        else:
            opening = np.asarray(arc["opening_deg"], dtype=np.float64)
            sin_vals = np.asarray(arc["sin_90_minus_theta_half"], dtype=np.float64)
            arc_t = np.asarray(arc["time_s"], dtype=np.float64)
            residual = np.asarray(arc["residual_deg"], dtype=np.float64)
            opening_vals = opening[np.isfinite(opening)]
            self._last_arc_opening = opening_vals

            self.ax_arc_time.plot(arc_t, opening, color="tab:blue", lw=0.8)
            self.ax_arc_time.axhline(0.0, color="0.82", lw=1)
            self.ax_arc_time.set_xlabel("time (s)")
            self.ax_arc_time.set_ylabel("theta (deg)")
            self.ax_arc_time.set_title("Arc angle from p99 point")
            self.ax_arc_time.grid(alpha=0.22)

            if opening_vals.size > 0:
                arc_bw = 1.0
                lo_ang = float(np.floor(float(np.min(opening_vals)) / arc_bw) * arc_bw)
                hi_ang = float(np.ceil(float(np.max(opening_vals)) / arc_bw) * arc_bw)
                if hi_ang <= lo_ang:
                    hi_ang = lo_ang + arc_bw
                angle_bins = np.arange(lo_ang, hi_ang + arc_bw + 1e-12, arc_bw)
                self.ax_arc_hist.hist(opening_vals, bins=angle_bins, color="tab:blue", alpha=0.82)
                self._set_default_or_clamped_arc_rulers(opening_vals)
                self._update_arc_ruler_artists()
            else:
                self.arc_info_var.set("Arc opening rulers: no finite theta values.")
            self.ax_arc_hist.set_xlabel("theta = arc angle from p99 point (deg)")
            self.ax_arc_hist.set_ylabel("count")
            self.ax_arc_hist.set_title("Occupied arc-angle distribution; drag either vertical ruler")
            self.ax_arc_hist.grid(alpha=0.22)

            self.ax_sin_time.plot(arc_t, sin_vals, color="tab:purple", lw=0.8)
            self.ax_sin_time.axhline(0.0, color="0.82", lw=1)
            self.ax_sin_time.set_xlabel("time (s)")
            self.ax_sin_time.set_ylabel("sin((90-theta)/2)")
            self.ax_sin_time.set_title("sin((90-theta)/2); theta = arc angle from p99 point")
            self.ax_sin_time.grid(alpha=0.22)

            vals = sin_vals[np.isfinite(sin_vals)]
            self._last_sin_values = vals
            if vals.size > 0:
                bw = max(1e-9, float(sin_bin_width))
                lo = float(np.floor(float(np.min(vals)) / bw) * bw)
                hi = float(np.ceil(float(np.max(vals)) / bw) * bw)
                if hi <= lo:
                    hi = lo + bw
                bins = np.arange(lo, hi + bw + 1e-12, bw)
                self.ax_sin_hist.hist(vals, bins=bins, color="tab:purple", alpha=0.85)
                self.ax_sin_hist.axvline(float(np.nanmedian(vals)), color="black", linestyle=":", linewidth=1.3)
                self._set_default_or_clamped_sin_rulers(vals)
                self._update_sin_ruler_artists()
                sin_summary = (
                    f"sin((90-theta)/2) mean={np.nanmean(vals):.5f}, std={np.nanstd(vals):.5f}, "
                    f"median={np.nanmedian(vals):.5f}"
                )
            else:
                self.sin_info_var.set("90-theta rulers: no finite transformed values.")
                sin_summary = "sin((90-theta)/2)=n/a"
            opening_summary = f"arc opening 99-1={float(arc['arc_opening_99_1_deg']):.4g} deg"
            self.ax_sin_hist.set_xlabel("sin((90-theta)/2), theta = arc angle from p99 point")
            self.ax_sin_hist.set_ylabel("count")
            self.ax_sin_hist.set_title(
                f"sin((90-theta)/2) distribution | bin={sin_bin_width:g} | "
                f"clipped-r={arc['r_clipped_count']} | residual p95={np.nanpercentile(residual, 95):.3g} deg"
            )
            self.ax_sin_hist.grid(alpha=0.22)

        mapped_names = [self.channel_names[i] for i in mapping]
        correction_label = "A^-1 D^-1 applied" if apply_correction else "no correction"
        self.figure.suptitle(
            f"{self.loaded_path.name if self.loaded_path else ''} | mapping {mapped_names} | {correction_label}",
            fontsize=10,
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.canvas.draw_idle()
        self.hemi_figure.suptitle(
            f"{self.loaded_path.name if self.loaded_path else ''} | theta model {self.theta_model_var.get()}",
            fontsize=10,
        )
        self.hemi_figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.hemi_canvas.draw_idle()
        self.arc_figure.suptitle(
            f"{self.loaded_path.name if self.loaded_path else ''} | theta model {self.theta_model_var.get()} | {opening_summary}",
            fontsize=10,
        )
        self.arc_figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.arc_canvas.draw_idle()
        self.sin_figure.suptitle(
            f"{self.loaded_path.name if self.loaded_path else ''} | theta model {self.theta_model_var.get()} | {opening_summary}",
            fontsize=10,
        )
        self.sin_figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.sin_canvas.draw_idle()

        self.status_var.set(
            " | ".join(
                [
                    f"points={averaged.shape[0]}",
                    f"recording frames={self.raw.shape[0]}",
                    f"selected frames={start}:{end}",
                    f"averaging number={average_n}",
                    f"correction={'on' if apply_correction else 'off'}",
                    f"mean XY=({np.mean(xy[:, 0]):.5f}, {np.mean(xy[:, 1]):.5f})",
                    f"R2 CV={np.std(r2) / max(abs(np.mean(r2)), 1e-12):.5g}",
                    opening_summary,
                    sin_summary,
                ]
            )
        )


def main() -> None:
    root = tk.Tk()
    app = ManualScalingXYApp(root)
    del app
    root.mainloop()


if __name__ == "__main__":
    main()
