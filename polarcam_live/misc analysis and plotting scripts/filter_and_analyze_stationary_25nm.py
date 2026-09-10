from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_DIR = Path("stationary rods 25nm 02072026") / "pending"
FILTERED_ROOT = Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400"
FILTERED_GOOD_DIR = FILTERED_ROOT / "good"
OUTPUT_DIR = FILTERED_ROOT / "plots" / "theta_phi_uncertainty_from_empirical_curve"
CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
INTENSITY_MIN = 50.0
INTENSITY_MAX = 400.0


@dataclass
class RodPoint:
    rod: str
    intensity_p98: float
    max_raw_value: float
    max_saved_value: float
    n_frames: int
    r_mean: float
    sigma_x: float
    sigma_y: float
    sigma_xy: float
    theta_deg: float
    sigma_phi_deg: float
    sigma_theta_xy_only_deg: float
    sigma_theta_curve_deg: float
    sigma_theta_total_deg: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = _load_json(meta_path)
    xy_series = payload.get("xy_series")
    if not isinstance(xy_series, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape in {meta_path}: {arr.shape}")
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _load_curve_with_uncertainty(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_std_deg = np.asarray([float(row["theta_deg_std"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {
        "r": r,
        "theta_deg": theta_deg,
        "theta_std_deg": theta_std_deg,
        "dtheta_dr_rad": dtheta_dr_rad,
    }


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], r_max_dataset: float) -> tuple[dict[str, np.ndarray], float]:
    r_max_empirical = float(np.max(curve["r"]))
    scale = float(r_max_dataset / max(r_max_empirical, 1e-12))
    return (
        {
            "r": curve["r"] * scale,
            "theta_deg": curve["theta_deg"].copy(),
            "theta_std_deg": curve["theta_std_deg"].copy(),
            "dtheta_dr_rad": curve["dtheta_dr_rad"] / max(scale, 1e-12),
        },
        scale,
    )


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _iter_source_rods() -> list[tuple[Path, dict, dict]]:
    out: list[tuple[Path, dict, dict]] = []
    for rod_dir in sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir()):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        if not meta_path.exists():
            continue
        payload = _load_json(meta_path)
        actual = dict(payload.get("actual", {}) or {})
        out.append((rod_dir, payload, actual))
    return out


def _copy_filtered_rods() -> tuple[list[Path], dict]:
    FILTERED_GOOD_DIR.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    rows: list[dict] = []
    source_dirs = {p.name: p for p, _payload, _actual in _iter_source_rods()}
    selected_names: set[str] = set()

    for rod_dir, _payload, actual in _iter_source_rods():
        intensity = float(actual.get("intensity_p98", float("nan")))
        if not np.isfinite(intensity):
            continue
        if intensity < INTENSITY_MIN or intensity > INTENSITY_MAX:
            continue
        selected.append(rod_dir)
        selected_names.add(rod_dir.name)
        rows.append(
            {
                "rod": rod_dir.name,
                "intensity_p98": intensity,
                "max_raw_value": actual.get("max_raw_value"),
                "max_saved_value": actual.get("max_saved_value"),
            }
        )

    for dst in list(FILTERED_GOOD_DIR.iterdir()) if FILTERED_GOOD_DIR.exists() else []:
        if not dst.is_dir():
            continue
        if dst.name not in selected_names:
            shutil.rmtree(dst)

    for rod_dir in selected:
        dst = FILTERED_GOOD_DIR / rod_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(rod_dir, dst)

    manifest_path = FILTERED_ROOT / "filtered_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rod", "intensity_p98", "max_raw_value", "max_saved_value"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_dir": str(SOURCE_DIR.resolve()),
        "filtered_good_dir": str(FILTERED_GOOD_DIR.resolve()),
        "intensity_p98_min": INTENSITY_MIN,
        "intensity_p98_max": INTENSITY_MAX,
        "n_selected": len(selected),
        "n_source_total": len(source_dirs),
    }
    return selected, summary


def _load_filtered_points(curve: dict[str, np.ndarray]) -> list[RodPoint]:
    rod_dirs = sorted(p for p in FILTERED_GOOD_DIR.iterdir() if p.is_dir())
    temp: list[dict] = []
    r_means: list[float] = []
    for rod_dir in rod_dirs:
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        arr = _load_xy_series(meta_path)
        if arr.size == 0:
            continue
        x = arr[:, 0]
        y = arr[:, 1]
        r = np.sqrt((x * x) + (y * y))
        r_mean = float(np.mean(r))
        r_means.append(r_mean)
        payload = _load_json(meta_path)
        actual = dict(payload.get("actual", {}) or {})
        temp.append(
            {
                "rod": rod_dir.name,
                "arr": arr,
                "r_mean": r_mean,
                "intensity_p98": float(actual.get("intensity_p98", float("nan"))),
                "max_raw_value": float(actual.get("max_raw_value", float("nan"))),
                "max_saved_value": float(actual.get("max_saved_value", float("nan"))),
            }
        )

    if not temp:
        return []

    stretched_curve, stretch_scale = _stretch_curve_for_dataset(curve, float(max(r_means)))
    points: list[RodPoint] = []
    for row in temp:
        arr = row["arr"]
        x = arr[:, 0]
        y = arr[:, 1]
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
        sigma_xy = float(np.hypot(sigma_x, sigma_y))
        r_mean = float(row["r_mean"])
        theta_deg = _interp(stretched_curve["r"], stretched_curve["theta_deg"], r_mean)
        theta_std_deg = _interp(stretched_curve["r"], stretched_curve["theta_std_deg"], r_mean)
        dtheta_dr_rad = _interp(stretched_curve["r"], stretched_curve["dtheta_dr_rad"], r_mean)
        sigma_phi_deg = float(np.degrees(sigma_xy / max(2.0 * r_mean, 1e-12)))
        sigma_theta_xy_only_deg = float(np.degrees(abs(dtheta_dr_rad) * sigma_xy))
        sigma_theta_total_deg = float(np.hypot(sigma_theta_xy_only_deg, theta_std_deg))
        points.append(
            RodPoint(
                rod=row["rod"],
                intensity_p98=float(row["intensity_p98"]),
                max_raw_value=float(row["max_raw_value"]),
                max_saved_value=float(row["max_saved_value"]),
                n_frames=int(arr.shape[0]),
                r_mean=r_mean,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                sigma_xy=sigma_xy,
                theta_deg=theta_deg,
                sigma_phi_deg=sigma_phi_deg,
                sigma_theta_xy_only_deg=sigma_theta_xy_only_deg,
                sigma_theta_curve_deg=theta_std_deg,
                sigma_theta_total_deg=sigma_theta_total_deg,
            )
        )

    points.sort(key=lambda p: p.theta_deg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "curve_csv": str(CURVE_CSV.resolve()),
        "stretch_scale_r": float(stretch_scale),
        "r_max_filtered_dataset": float(max(r_means)),
        "r_max_empirical_curve": float(np.max(curve["r"])),
    }
    (OUTPUT_DIR / "curve_stretch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return points


def _write_points_csv(points: list[RodPoint]) -> None:
    rows = [
        {
            "rod": p.rod,
            "intensity_p98": p.intensity_p98,
            "max_raw_value": p.max_raw_value,
            "max_saved_value": p.max_saved_value,
            "n_frames": p.n_frames,
            "r_mean": p.r_mean,
            "sigma_x": p.sigma_x,
            "sigma_y": p.sigma_y,
            "sigma_xy": p.sigma_xy,
            "theta_deg": p.theta_deg,
            "sigma_phi_deg": p.sigma_phi_deg,
            "sigma_theta_xy_only_deg": p.sigma_theta_xy_only_deg,
            "sigma_theta_curve_deg": p.sigma_theta_curve_deg,
            "sigma_theta_total_deg": p.sigma_theta_total_deg,
        }
        for p in points
    ]
    with (OUTPUT_DIR / "theta_phi_uncertainty_points.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scatter(x: np.ndarray, y: np.ndarray, out_path: Path, xlabel: str, ylabel: str, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=34, alpha=0.88, color=color, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _make_plots(points: list[RodPoint]) -> None:
    th = np.asarray([p.theta_deg for p in points], dtype=np.float64)
    sphi = np.asarray([p.sigma_phi_deg for p in points], dtype=np.float64)
    sth_xy = np.asarray([p.sigma_theta_xy_only_deg for p in points], dtype=np.float64)
    sth_total = np.asarray([p.sigma_theta_total_deg for p in points], dtype=np.float64)
    inten = np.asarray([p.intensity_p98 for p in points], dtype=np.float64)
    r_mean = np.asarray([p.r_mean for p in points], dtype=np.float64)

    _scatter(
        th,
        sphi,
        OUTPUT_DIR / "std_phi_vs_theta.png",
        xlabel="theta (deg)",
        ylabel="std dev phi (deg)",
        title="25nm rods: standard deviation in phi vs theta",
        color="#1f77b4",
    )
    _scatter(
        th,
        sth_xy,
        OUTPUT_DIR / "std_theta_vs_theta_xy_only.png",
        xlabel="theta (deg)",
        ylabel="std dev theta (deg)",
        title="25nm rods: standard deviation in theta vs theta from XY propagation only",
        color="#2ca02c",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sth_total, s=34, alpha=0.88, color="#d62728", label="XY + theta(r) curve uncertainty", edgecolors="none")
    ax.scatter(th, sth_xy, s=24, alpha=0.42, color="#2ca02c", label="XY only", edgecolors="none")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("25nm rods: standard deviation in theta vs theta including theta(r) uncertainty")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "std_theta_vs_theta_with_curve_uncertainty.png", dpi=220)
    plt.close(fig)

    _scatter(
        r_mean,
        inten,
        OUTPUT_DIR / "intensity_p98_vs_mean_r_filtered.png",
        xlabel="Mean r",
        ylabel="Intensity p98 after background subtraction",
        title="25nm rods filtered: intensity vs mean r",
        color="#9467bd",
    )


def main() -> None:
    curve = _load_curve_with_uncertainty(Path.cwd() / CURVE_CSV)
    selected, filter_summary = _copy_filtered_rods()
    if not selected:
        raise SystemExit("No rods matched the requested intensity_p98 filter.")

    points = _load_filtered_points(curve)
    if not points:
        raise SystemExit("No filtered rods could be analyzed.")

    _write_points_csv(points)
    _make_plots(points)

    th = np.asarray([p.theta_deg for p in points], dtype=np.float64)
    sphi = np.asarray([p.sigma_phi_deg for p in points], dtype=np.float64)
    sth_xy = np.asarray([p.sigma_theta_xy_only_deg for p in points], dtype=np.float64)
    sth_total = np.asarray([p.sigma_theta_total_deg for p in points], dtype=np.float64)
    summary = {
        **filter_summary,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "n_analyzed": len(points),
        "theta_range_deg": [float(np.min(th)), float(np.max(th))],
        "sigma_phi_deg_range": [float(np.min(sphi)), float(np.max(sphi))],
        "sigma_theta_xy_only_deg_range": [float(np.min(sth_xy)), float(np.max(sth_xy))],
        "sigma_theta_total_deg_range": [float(np.min(sth_total)), float(np.max(sth_total))],
        "method_theta_curve": "Empirical glycerol theta(r) curve stretched along r to match the filtered 25nm dataset r_max",
        "method_phi": "sigma_phi = sigma_xy / (2 r), then convert to degrees",
        "method_theta_xy_only": "sigma_theta = |dtheta/dr| * sigma_xy using the stretched empirical theta(r) center curve",
        "method_theta_with_curve": "sigma_theta_total = sqrt(sigma_theta_xy_only^2 + sigma_theta_curve^2)",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
