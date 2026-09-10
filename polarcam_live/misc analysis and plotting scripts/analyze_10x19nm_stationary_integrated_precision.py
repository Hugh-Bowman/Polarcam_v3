from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(r"E:\10x19nm stationary")
OUT_DIR = DATA_DIR / "plots"
CURVE_CSV = (
    Path(__file__).resolve().parents[1]
    / "theta_r_curve_parameters"
    / "water1p33_glycerol1p47_naout1p3_nain0p39"
    / "theta_r_abc_values.csv"
)

MM_TO_IN = 1.0 / 25.4
FONT_PT = 7
FIG_W_MM = 90.0
FIG_H_MM = 62.0
LOW_FREQ_HZ = 1.0
THETA_BIN_WIDTH_DEG = 10.0
MAX_PER_THETA_BIN = 2
PHI_PLOT_THETA_MIN_DEG = 15.0


@dataclass
class ThetaModel:
    label: str
    a: float
    b: float
    c: float
    r_max: float


@dataclass
class RodPoint:
    group: str
    rod: str
    meta_path: Path
    n_frames: int
    fps: float
    duration_s: float
    r_mean: float
    theta_deg: float
    phi_mean_deg: float
    sigma_phi_deg: float
    sigma_theta_deg: float
    theta_bin_deg: int


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "svg.fonttype": "none",
        }
    )


def load_water_theta_model(path: Path) -> ThetaModel:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    water_rows = [row for row in rows if row.get("model", "").strip().lower() == "water"]
    if not water_rows:
        raise ValueError(f"No water row found in {path}")
    row = water_rows[0]
    return ThetaModel(
        label=f"water finite-NA r_max={float(row['r_max']):.6f}",
        a=float(row["A"]),
        b=float(row["B"]),
        c=float(row["C"]),
        r_max=float(row["r_max"]),
    )


def theta_from_r(model: ThetaModel, r: np.ndarray) -> np.ndarray:
    rr = np.asarray(r, dtype=np.float64)
    theta = np.full(rr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(rr) & (rr >= 0.0)
    clipped = finite & (rr >= model.r_max)
    valid = finite & (rr < model.r_max)
    theta[clipped] = 90.0
    denom = model.b - (model.c * rr[valid])
    value = (model.a * rr[valid]) / np.maximum(denom, 1e-15)
    theta[valid] = np.degrees(np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0))))
    return theta


def band_limited_std_deg(values_deg: np.ndarray, fps: float, low_hz: float = LOW_FREQ_HZ) -> float:
    values = np.asarray(values_deg, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4 or not np.isfinite(fps) or fps <= 0.0:
        return float("nan")
    values = values - float(np.mean(values))
    coeff = np.fft.rfft(values)
    power = (np.abs(coeff) ** 2) / float(values.size * values.size)
    if power.size > 2:
        power[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(values.size, d=1.0 / float(fps))
    band = (freqs >= low_hz) & (freqs <= (float(fps) / 2.0))
    return float(np.sqrt(np.sum(power[band])))


def load_xy(meta: dict, meta_path: Path) -> np.ndarray:
    xy = np.asarray(meta.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Bad or missing xy_series in {meta_path}")
    actual_frames = meta.get("actual", {}).get("frames")
    if actual_frames is not None:
        xy = xy[: int(actual_frames)]
    xy = xy[:, :2]
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    return xy[ok]


def analyse_meta(meta_path: Path, model: ThetaModel, group: str) -> RodPoint:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = load_xy(meta, meta_path)
    if xy.shape[0] < 4:
        raise ValueError(f"Too few XY samples in {meta_path}")
    fps = float(meta.get("actual", {}).get("fps", float("nan")))
    duration = float(meta.get("actual", {}).get("duration_s", xy.shape[0] / fps if fps > 0 else float("nan")))
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.hypot(x, y)
    r_mean = float(np.mean(r))
    theta_deg = float(theta_from_r(model, np.asarray([r_mean], dtype=np.float64))[0])
    theta_series = theta_from_r(model, np.clip(r, 0.0, model.r_max))

    phi_wrapped = 0.5 * np.arctan2(y, x)
    phi_unwrapped = 0.5 * np.unwrap(np.angle(np.exp(2.0j * phi_wrapped)))
    phi_mean_deg = float(np.degrees(0.5 * np.angle(np.mean(np.exp(2.0j * phi_wrapped)))))
    phi_series_deg = np.degrees(phi_unwrapped - float(np.mean(phi_unwrapped)))

    sigma_phi = band_limited_std_deg(phi_series_deg, fps)
    sigma_theta = band_limited_std_deg(theta_series, fps)
    theta_bin = int(math.floor(theta_deg / THETA_BIN_WIDTH_DEG) * THETA_BIN_WIDTH_DEG)
    theta_bin = max(0, min(80, theta_bin))

    return RodPoint(
        group=group,
        rod=meta_path.parent.name,
        meta_path=meta_path,
        n_frames=int(xy.shape[0]),
        fps=fps,
        duration_s=duration,
        r_mean=r_mean,
        theta_deg=theta_deg,
        phi_mean_deg=phi_mean_deg,
        sigma_phi_deg=sigma_phi,
        sigma_theta_deg=sigma_theta,
        theta_bin_deg=theta_bin,
    )


def find_meta_files(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for group in ("good", "pending", "bad"):
        group_dir = root / group
        if not group_dir.exists():
            continue
        for meta_path in sorted(group_dir.rglob("capture_maxfps_15x15_meta.json")):
            out.append((group, meta_path))
    return out


def select_best_per_theta_bin(points: list[RodPoint], attr: str) -> list[RodPoint]:
    selected: list[RodPoint] = []
    bins = sorted({point.theta_bin_deg for point in points})
    for theta_bin in bins:
        subset = [
            point
            for point in points
            if point.theta_bin_deg == theta_bin and np.isfinite(float(getattr(point, attr)))
        ]
        subset.sort(key=lambda point: float(getattr(point, attr)))
        selected.extend(subset[:MAX_PER_THETA_BIN])
    return sorted(selected, key=lambda point: (point.theta_deg, float(getattr(point, attr)), point.rod))


def write_points_csv(path: Path, points: list[RodPoint]) -> None:
    fields = list(RodPoint.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = point.__dict__.copy()
            row["meta_path"] = str(point.meta_path)
            writer.writerow(row)


def add_low_theta_region(ax: plt.Axes) -> None:
    ax.axvspan(
        0.0,
        10.0,
        facecolor="#777777",
        alpha=0.08,
        hatch="///",
        edgecolor="#777777",
        linewidth=0.0,
        label=r"low $\theta$",
        zorder=0,
    )


def plot_vs_theta(
    all_points: list[RodPoint],
    selected_points: list[RodPoint],
    attr: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    x_all = np.asarray([point.theta_deg for point in all_points], dtype=np.float64)
    y_all = np.asarray([getattr(point, attr) for point in all_points], dtype=np.float64)
    x_sel = np.asarray([point.theta_deg for point in selected_points], dtype=np.float64)
    y_sel = np.asarray([getattr(point, attr) for point in selected_points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    add_low_theta_region(ax)
    ax.scatter(
        x_all,
        y_all,
        s=17,
        marker="o",
        color="#9aa0a6",
        edgecolors="none",
        alpha=0.34,
        label="all rods",
    )
    ax.scatter(
        x_sel,
        y_sel,
        s=28,
        marker="D",
        color="#2a7f62",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.92,
        label="best two per 10 deg",
    )
    y_ok = y_sel[np.isfinite(y_sel)]
    if y_ok.size == 0:
        y_ok = y_all[np.isfinite(y_all)]
    y_max = float(np.max(y_ok) * 1.16) if y_ok.size else 1.0
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(y_max, 0.05))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    style_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_water_theta_model(CURVE_CSV)
    points: list[RodPoint] = []
    skipped: list[dict[str, str]] = []
    for group, meta_path in find_meta_files(DATA_DIR):
        try:
            points.append(analyse_meta(meta_path, model, group))
        except Exception as exc:
            skipped.append({"meta_path": str(meta_path), "error": str(exc)})
    if not points:
        raise SystemExit(f"No usable capture metadata found in {DATA_DIR}")

    phi_points = [point for point in points if point.theta_deg >= PHI_PLOT_THETA_MIN_DEG]
    selected_phi = select_best_per_theta_bin(phi_points, "sigma_phi_deg")
    selected_theta = select_best_per_theta_bin(points, "sigma_theta_deg")

    write_points_csv(OUT_DIR / "integrated_precision_all_points.csv", points)
    write_points_csv(OUT_DIR / "integrated_precision_selected_phi_points.csv", selected_phi)
    write_points_csv(OUT_DIR / "integrated_precision_selected_theta_points.csv", selected_theta)

    plot_vs_theta(
        phi_points,
        selected_phi,
        "sigma_phi_deg",
        r"$\sigma_\phi$ (deg)",
        rf"10x19 nm stationary rods: $\sigma_\phi$ vs $\theta$, {LOW_FREQ_HZ:g} Hz-Nyquist",
        OUT_DIR / "integrated_sigma_phi_vs_theta_1hz_to_nyquist.svg",
    )
    plot_vs_theta(
        points,
        selected_theta,
        "sigma_theta_deg",
        r"$\sigma_\theta$ (deg)",
        rf"10x19 nm stationary rods: $\sigma_\theta$ vs $\theta$, {LOW_FREQ_HZ:g} Hz-Nyquist",
        OUT_DIR / "integrated_sigma_theta_vs_theta_1hz_to_nyquist.svg",
    )

    theta_values = np.asarray([point.theta_deg for point in points], dtype=np.float64)
    summary = {
        "source_dir": str(DATA_DIR),
        "output_dir": str(OUT_DIR),
        "theta_curve_csv": str(CURVE_CSV),
        "theta_model": model.__dict__,
        "n_points_all": int(len(points)),
        "n_points_phi_plot": int(len(phi_points)),
        "n_points_selected_phi": int(len(selected_phi)),
        "n_points_selected_theta": int(len(selected_theta)),
        "groups": {group: sum(1 for point in points if point.group == group) for group in sorted({p.group for p in points})},
        "theta_range_deg": [float(np.min(theta_values)), float(np.max(theta_values))],
        "selection_rule": f"for each plot, keep up to {MAX_PER_THETA_BIN} rods with the lowest plotted sigma in each {THETA_BIN_WIDTH_DEG:g} degree theta bin",
        "phi_plot_theta_filter": f"phi plot excludes rods with theta < {PHI_PLOT_THETA_MIN_DEG:g} deg",
        "bandwidth_method": "sigma is sqrt(sum one-sided FFT variance components from 1 Hz through Nyquist after subtracting the mean)",
        "sigma_phi_method": "phi(t)=0.5*unwrap(angle(exp(2j*0.5*atan2(Y,X))))",
        "sigma_theta_method": "theta(t) reconstructed from r(t)=sqrt(X^2+Y^2) using the latest water theta-r curve",
        "skipped": skipped,
    }
    (OUT_DIR / "integrated_precision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {OUT_DIR}")
    print(f"Analysed {len(points)} rods; selected {len(selected_phi)} for phi and {len(selected_theta)} for theta.")


if __name__ == "__main__":
    main()
