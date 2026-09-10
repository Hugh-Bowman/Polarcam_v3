from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyze_40nm_bg_subtracted_sound_precision_new as water_curve


CAMERA_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision_error_model_bg_subtracted_attempt240"
)
APD_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm APD stuck rod data\apd_plot_points.csv"
)
OUT_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision_error_model_bg_subtracted_attempt240\camera_vs_apd_1hz_compare"
)

PHI_CSV = CAMERA_DIR / "0p6ms_phi_fit_rods.csv"
THETA_CSV = CAMERA_DIR / "0p6ms_theta_fit_rods.csv"
BANDPASS_LOW_HZ = 1.0
MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 90.0
FIG_H_MM = 63.0
LOW_THETA_SHADE_DEG = 10.0
PHI_YMAX_DEG = 6.0
EXCLUDED_CAMERA_RODS = {
    "rod_x1473_y870_20260805-184622_1785951982100791600",
}


@dataclass
class CameraPoint:
    rod: str
    path: Path
    theta_deg: float
    sigma_theta_deg: float
    sigma_phi_deg: float
    sample_rate_hz: float
    n_frames: int


def highpass_std_deg(series_deg: np.ndarray, fs: float) -> float:
    values = np.asarray(series_deg, dtype=np.float64)
    ok = np.isfinite(values)
    values = values[ok]
    if values.size < 16:
        return float("nan")
    values = values - float(np.mean(values))
    if not np.isfinite(fs) or fs <= 2.0 * BANDPASS_LOW_HZ:
        return float(np.std(values))
    sos = butter(2, BANDPASS_LOW_HZ, btype="highpass", fs=fs, output="sos")
    return float(np.std(sosfiltfilt(sos, values)))


def load_selected_paths(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_camera_point(rod_path: Path, curve: dict[str, np.ndarray]) -> CameraPoint | None:
    meta_path = rod_path / "capture_maxfps_15x15_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = water_curve.load_xy(meta, meta_path)
    if xy.shape[0] < 16:
        return None
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 16:
        return None
    r = np.hypot(x, y)
    r_mean = float(np.mean(r))
    theta = water_curve.theta_from_r(curve, r_mean)
    if theta is None:
        return None
    theta_deg = float(theta[0])
    r_max = float(water_curve.FOURKAS_WATER["r_max"])
    r_shift = max(0.0, r_mean - r_max)
    r_for_theta = np.clip(r - r_shift, 0.0, r_max)
    theta_series = water_curve.theta_series_from_r(curve, r_for_theta)
    phi_raw = 0.5 * np.unwrap(np.angle(np.exp(2.0j * (0.5 * np.arctan2(y, x)))))
    phi_deg = np.degrees(phi_raw - float(np.mean(phi_raw)))
    fs = float(meta.get("actual", {}).get("fps") or meta.get("requested", {}).get("fps") or np.nan)
    sigma_theta = highpass_std_deg(theta_series, fs)
    sigma_phi = highpass_std_deg(phi_deg, fs)
    return CameraPoint(
        rod=rod_path.name,
        path=rod_path,
        theta_deg=theta_deg,
        sigma_theta_deg=float(sigma_theta),
        sigma_phi_deg=float(sigma_phi),
        sample_rate_hz=fs,
        n_frames=int(x.size),
    )


def load_camera_points(csv_path: Path, curve: dict[str, np.ndarray]) -> list[CameraPoint]:
    rows = load_selected_paths(csv_path)
    out: list[CameraPoint] = []
    for row in rows:
        point = load_camera_point(Path(row["path"]), curve)
        if point is not None:
            if point.rod in EXCLUDED_CAMERA_RODS:
                continue
            out.append(point)
    out.sort(key=lambda p: p.theta_deg)
    return out


def load_apd_points() -> list[dict[str, float | str]]:
    with APD_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        out.append(
            {
                "rod": row["rod_file"],
                "theta_deg": float(row["theta_deg_current_curve"]),
                "sigma_theta_deg": float(row["sigma_theta_deg_bandwidth_1Hz_to_nyquist"]),
                "sigma_phi_deg": float(row["sigma_phi_deg_bandwidth_1Hz_to_nyquist"]),
            }
        )
    return out


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def make_plot(
    camera_points: list[CameraPoint],
    apd_points: list[dict[str, float | str]],
    y_key: str,
    y_label: str,
    panel_label: str,
    out_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    cam_theta = np.asarray([p.theta_deg for p in camera_points], dtype=np.float64)
    cam_y = np.asarray([getattr(p, y_key) for p in camera_points], dtype=np.float64)
    apd_theta = np.asarray([float(p["theta_deg"]) for p in apd_points], dtype=np.float64)
    apd_y = np.asarray([float(p[y_key]) for p in apd_points], dtype=np.float64)

    ax.scatter(
        cam_theta,
        cam_y,
        s=18,
        marker="o",
        color="#1f77b4",
        edgecolors="none",
        alpha=0.9,
        label="40 nm camera, 0.6 ms",
    )
    ax.scatter(
        apd_theta,
        apd_y,
        s=20,
        marker="D",
        color="#2a7f62",
        edgecolors="none",
        alpha=0.9,
        label="40 nm APD",
    )
    if y_key == "sigma_phi_deg":
        ax.axvspan(
            0.0,
            LOW_THETA_SHADE_DEG,
            facecolor="#8f8f8f",
            alpha=0.16,
            hatch="///",
            edgecolor="#8f8f8f",
            linewidth=0.0,
            zorder=0,
        )

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, color="#1f77b4", label="40 nm camera, 0.6 ms"),
        Line2D([], [], linestyle="none", marker="D", markersize=4.5, color="#2a7f62", label="40 nm APD"),
        Line2D([], [], linestyle="none", marker="s", markersize=4.5, color="#9a9a9a", label="25 nm camera pending"),
        Line2D([], [], linestyle="none", marker="^", markersize=4.8, color="#b0b0b0", label="25 nm APD pending"),
    ]
    if y_key == "sigma_phi_deg":
        handles.append(
            Patch(
                facecolor="#8f8f8f",
                edgecolor="#8f8f8f",
                hatch="///",
                alpha=0.16,
                label=r"$\sigma_\phi$ error diverges at low $\theta$",
            )
        )

    y_all = np.concatenate([cam_y[np.isfinite(cam_y)], apd_y[np.isfinite(apd_y)]])
    y_cap = float(np.max(y_all) * 1.08) if y_all.size else 1.0
    ax.set_xlim(0.0, 90.0)
    if y_key == "sigma_phi_deg":
        ax.set_ylim(0.0, PHI_YMAX_DEG)
    else:
        ax.set_ylim(0.0, max(y_cap, 1.0))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    legend_loc = "upper right" if y_key == "sigma_phi_deg" else "upper left"
    ax.legend(handles=handles, frameon=False, loc=legend_loc, handletextpad=0.5, borderpad=0.2, labelspacing=0.35)
    fig.tight_layout(pad=0.8)
    fig.text(0.012, 0.988, panel_label, ha="left", va="top", fontsize=8, fontweight="bold")
    fig.savefig(OUT_DIR / out_name, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style_matplotlib()
    curve = water_curve.build_curve()
    phi_camera = load_camera_points(PHI_CSV, curve)
    theta_camera = load_camera_points(THETA_CSV, curve)
    apd_points = load_apd_points()

    make_plot(
        phi_camera,
        apd_points,
        "sigma_phi_deg",
        r"$\sigma_\phi$ (deg)",
        "a",
        "sigma_phi_vs_theta_40nm_camera_apd_0p6ms_1hz.png",
    )
    make_plot(
        theta_camera,
        apd_points,
        "sigma_theta_deg",
        r"$\sigma_\theta$ (deg)",
        "b",
        "sigma_theta_vs_theta_40nm_camera_apd_0p6ms_1hz.png",
    )

    summary = {
        "camera_phi_csv": str(PHI_CSV),
        "camera_theta_csv": str(THETA_CSV),
        "apd_csv": str(APD_CSV),
        "camera_phi_points": len(phi_camera),
        "camera_theta_points": len(theta_camera),
        "apd_points": len(apd_points),
        "excluded_camera_rods": sorted(EXCLUDED_CAMERA_RODS),
        "bandwidth_method": "std(highpass_1Hz(signal)) with upper bandwidth at Nyquist",
        "theta_curve": water_curve.FOURKAS_WATER,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
