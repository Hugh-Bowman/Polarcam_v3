from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt

import analyze_40nm_bg_subtracted_sound_precision_new as water_curve


DATA_DIR = Path(r"E:\40nm apd precision")
OUT_DIR = DATA_DIR / "analysis"

CHANNEL_MAP = {
    "I90": ("ai0", 1.0000),
    "I45": ("ai1", 2.8859),
    "I135": ("ai2", 2.3267),
    "I0": ("ai3", 1.0732),
}

R_SIGMA_LO_PCT = 16.0
R_SIGMA_HI_PCT = 84.0
R_SIGMA_DIVISOR = 2.0
BANDPASS_LOW_HZ = 1.0


@dataclass
class ApdRod:
    file: Path
    n_samples: int
    sample_rate_hz: float
    duration_s: float
    x_mean: float
    y_mean: float
    r_mean: float
    r_shift_to_rmax: float
    r_p16_raw: float
    r_p84_raw: float
    r_p16: float
    r_p84: float
    sigma_r_p16_p84: float
    theta_deg: float
    theta_p16_deg: float
    theta_p84_deg: float
    sigma_theta_deg: float
    sigma_theta_deg_unfiltered_percentile: float
    phi_mean_deg: float
    sigma_phi_deg: float
    sigma_phi_deg_unfiltered_circular: float
    mean_I0: float
    mean_I45: float
    mean_I90: float
    mean_I135: float


def _channel_short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


def read_scaled_apd_channels(path: Path) -> tuple[dict[str, np.ndarray], float]:
    tdms = TdmsFile.read(path)
    group = tdms.groups()[0]
    channels_by_short = {_channel_short_name(ch.name): ch for ch in group.channels()}
    out: dict[str, np.ndarray] = {}
    fs = float("nan")
    min_len = None
    for logical, (short_name, scale) in CHANNEL_MAP.items():
        if short_name not in channels_by_short:
            raise ValueError(f"{path.name} missing channel {short_name}")
        ch = channels_by_short[short_name]
        vals = np.asarray(ch[:], dtype=np.float64) * float(scale)
        out[logical] = vals
        min_len = vals.size if min_len is None else min(min_len, vals.size)
        dt = ch.properties.get("wf_increment")
        if dt:
            fs = 1.0 / float(dt)
    if min_len is None or min_len < 5:
        raise ValueError(f"{path.name} has too few samples")
    out = {k: v[:min_len] for k, v in out.items()}
    return out, fs


def theta_series_from_r(curve: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    return water_curve.theta_series_from_r(curve, r)


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
    filtered = sosfiltfilt(sos, values)
    return float(np.std(filtered))


def circular_phi_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    phi = 0.5 * np.arctan2(y, x)
    z = np.exp(2.0j * phi)
    z_mean = np.mean(z)
    mean_phi = 0.5 * float(np.angle(z_mean))
    resultant = float(np.clip(np.abs(z_mean), 1e-12, 1.0))
    sigma_phi = 0.5 * math.sqrt(max(0.0, -2.0 * math.log(resultant)))
    return float(np.degrees(mean_phi)), float(np.degrees(sigma_phi))


def analyse_file(path: Path, curve: dict[str, np.ndarray]) -> ApdRod:
    ch, fs = read_scaled_apd_channels(path)
    eps = 1e-15
    i0 = ch["I0"]
    i90 = ch["I90"]
    i45 = ch["I45"]
    i135 = ch["I135"]
    x = (i0 - i90) / np.maximum(i0 + i90, eps)
    y = (i45 - i135) / np.maximum(i45 + i135, eps)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 5:
        raise ValueError(f"{path.name} has too few finite XY samples")
    r = np.hypot(x, y)
    r_mean = float(np.mean(r))
    r_max = float(water_curve.FOURKAS_WATER["r_max"])
    theta = water_curve.theta_from_r(curve, r_mean)
    if theta is None:
        raise ValueError(f"{path.name} cannot be mapped to theta, r={r_mean}")
    theta_deg, _dtheta, _clipped = theta
    r_p16_raw = float(np.percentile(r, R_SIGMA_LO_PCT))
    r_p84_raw = float(np.percentile(r, R_SIGMA_HI_PCT))
    r_shift = max(0.0, r_mean - r_max)
    r_for_theta = np.clip(r - r_shift, 0.0, r_max)
    r_p16 = float(np.percentile(r_for_theta, R_SIGMA_LO_PCT))
    r_p84 = float(np.percentile(r_for_theta, R_SIGMA_HI_PCT))
    theta_p16 = water_curve.theta_value_from_r(curve, r_p16)
    theta_p84 = water_curve.theta_value_from_r(curve, r_p84)
    sigma_theta_unfiltered = float(abs(theta_p84 - theta_p16) / R_SIGMA_DIVISOR)
    theta_series = theta_series_from_r(curve, r_for_theta)
    sigma_theta = highpass_std_deg(theta_series, fs)
    phi_raw = 0.5 * np.unwrap(np.angle(np.exp(2.0j * (0.5 * np.arctan2(y, x)))))
    phi_deg = np.degrees(phi_raw - float(np.mean(phi_raw)))
    phi_mean, sigma_phi_unfiltered = circular_phi_stats(x, y)
    sigma_phi = highpass_std_deg(phi_deg, fs)
    n = int(x.size)
    duration = float(n / fs) if np.isfinite(fs) and fs > 0 else float("nan")
    return ApdRod(
        file=path,
        n_samples=n,
        sample_rate_hz=float(fs),
        duration_s=duration,
        x_mean=float(np.mean(x)),
        y_mean=float(np.mean(y)),
        r_mean=r_mean,
        r_shift_to_rmax=float(r_shift),
        r_p16_raw=r_p16_raw,
        r_p84_raw=r_p84_raw,
        r_p16=r_p16,
        r_p84=r_p84,
        sigma_r_p16_p84=float((r_p84 - r_p16) / R_SIGMA_DIVISOR),
        theta_deg=float(theta_deg),
        theta_p16_deg=float(theta_p16),
        theta_p84_deg=float(theta_p84),
        sigma_theta_deg=sigma_theta,
        sigma_theta_deg_unfiltered_percentile=sigma_theta_unfiltered,
        phi_mean_deg=phi_mean,
        sigma_phi_deg=float(sigma_phi),
        sigma_phi_deg_unfiltered_circular=float(sigma_phi_unfiltered),
        mean_I0=float(np.mean(i0[ok])),
        mean_I45=float(np.mean(i45[ok])),
        mean_I90=float(np.mean(i90[ok])),
        mean_I135=float(np.mean(i135[ok])),
    )


def write_csv(rows: list[ApdRod], path: Path) -> None:
    fields = [
        "file",
        "n_samples",
        "sample_rate_hz",
        "duration_s",
        "x_mean",
        "y_mean",
        "r_mean",
        "r_shift_to_rmax",
        "r_p16_raw",
        "r_p84_raw",
        "r_p16",
        "r_p84",
        "sigma_r_p16_p84",
        "theta_deg",
        "theta_p16_deg",
        "theta_p84_deg",
        "sigma_theta_deg",
        "sigma_theta_deg_unfiltered_percentile",
        "phi_mean_deg",
        "sigma_phi_deg",
        "sigma_phi_deg_unfiltered_circular",
        "mean_I0",
        "mean_I45",
        "mean_I90",
        "mean_I135",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            d = r.__dict__.copy()
            d["file"] = str(r.file)
            writer.writerow(d)


def add_low_theta_region(ax: plt.Axes) -> None:
    ax.axvspan(
        0.0,
        10.0,
        facecolor="#777777",
        alpha=0.08,
        hatch="///",
        edgecolor="#777777",
        linewidth=0.0,
        label=r"error diverges at low $\theta$",
        zorder=0,
    )


def plot_vs_theta(
    rows: list[ApdRod],
    y_attr: str,
    ylabel: str,
    title: str,
    filename: str,
    exclude_zero_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    add_low_theta_region(ax)
    plot_rows = [r for r in rows if (not exclude_zero_y) or float(getattr(r, y_attr)) > 0.0]
    theta = np.asarray([r.theta_deg for r in plot_rows], dtype=np.float64)
    y = np.asarray([getattr(r, y_attr) for r in plot_rows], dtype=np.float64)
    ax.scatter(theta, y, s=50, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45, alpha=0.86, label="APD rods")
    y_ok = y[np.isfinite(y)]
    y_cap = float(np.max(y_ok) * 1.12) if y_ok.size else 1.0
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(y_cap, 1.0))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}, {BANDPASS_LOW_HZ:g} Hz-Nyquist bandwidth")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220)
    plt.close(fig)


def plot_phi_vs_phi(rows: list[ApdRod]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    phi = np.asarray([r.phi_mean_deg for r in rows], dtype=np.float64)
    y = np.asarray([r.sigma_phi_deg for r in rows], dtype=np.float64)
    ax.scatter(phi, y, s=50, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45, alpha=0.86)
    y_ok = y[np.isfinite(y)]
    y_cap = float(np.max(y_ok) * 1.12) if y_ok.size else 1.0
    ax.set_ylim(0.0, max(y_cap, 1.0))
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\sigma_\phi$ (deg)")
    ax.set_title(rf"40 nm APD rods: $\sigma_\phi$ vs $\phi$, {BANDPASS_LOW_HZ:g} Hz-Nyquist bandwidth")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "apd_sigma_phi_vs_phi.png", dpi=220)
    plt.close(fig)


def plot_xy(rows: list[ApdRod]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    x = [r.x_mean for r in rows]
    y = [r.y_mean for r in rows]
    ax.scatter(x, y, s=52, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45)
    for r in rows:
        ax.annotate(r.file.stem, (r.x_mean, r.y_mean), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X = (I0 - I90) / (I0 + I90)")
    ax.set_ylabel("Y = (I45 - I135) / (I45 + I135)")
    ax.set_title("40 nm APD rods: mean XY")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "apd_mean_xy.png", dpi=220)
    plt.close(fig)


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = water_curve.build_curve()
    tdms_files = sorted(DATA_DIR.glob("*.tdms"), key=natural_sort_key)
    rows = []
    for path in tdms_files:
        try:
            rows.append(analyse_file(path, curve))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    if not rows:
        raise SystemExit(f"No TDMS files analysed in {DATA_DIR}")
    write_csv(rows, OUT_DIR / "apd_precision_points.csv")
    plot_vs_theta(rows, "sigma_theta_deg", r"$\sigma_\theta$ (deg)", r"40 nm APD rods: $\sigma_\theta$ vs $\theta$", "apd_sigma_theta_vs_theta.png")
    plot_vs_theta(rows, "sigma_phi_deg", r"$\sigma_\phi$ (deg)", r"40 nm APD rods: $\sigma_\phi$ vs $\theta$", "apd_sigma_phi_vs_theta.png")
    plot_phi_vs_phi(rows)
    plot_xy(rows)
    summary = {
        "source_dir": str(DATA_DIR),
        "n_tdms_files": len(tdms_files),
        "n_analysed": len(rows),
        "channel_scalings": CHANNEL_MAP,
        "xy_definition": "X=(I0-I90)/(I0+I90), Y=(I45-I135)/(I45+I135)",
        "theta_curve": water_curve.FOURKAS_WATER,
        "bandwidth_method": f"reported plotted sigma values are standard deviations after a {BANDPASS_LOW_HZ:g} Hz high-pass filter; upper bandwidth is Nyquist from each TDMS sample rate",
        "sigma_theta_method": "theta(t) is reconstructed from r(t). If r_mean>r_max, the full r(t) distribution is shifted by r_mean-r_max before theta reconstruction, while the point is plotted at theta=90 deg. sigma_theta_deg is std(highpass_1Hz(theta(t))).",
        "theta_plot_filter": "none",
        "sigma_phi_method": "phi(t)=0.5*unwrap(atan2(Y,X)); sigma_phi_deg is std(highpass_1Hz(phi(t))).",
        "unfiltered_columns": "sigma_theta_deg_unfiltered_percentile and sigma_phi_deg_unfiltered_circular preserve the previous unfiltered metrics",
        "output_csv": str(OUT_DIR / "apd_precision_points.csv"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {OUT_DIR}")
    print(f"Analysed {len(rows)} TDMS files")


if __name__ == "__main__":
    main()
