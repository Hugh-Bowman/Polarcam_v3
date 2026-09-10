from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")


DATASET_DIR = Path("various exposure recording of 25nm rods 03072026")
OUTPUT_DIR = DATASET_DIR / "plots" / "precision_vs_exposure"
EMPIRICAL_CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
SATURATION_EXPOSURE_MS = 0.8


@dataclass
class RodPoint:
    rod: str
    exposure_ms: float
    gain_analog: float
    max_pixel_value: float
    fraction_saturated: float
    intensity_p98: float
    r_mean: float
    theta_deg: float
    sigma_xy: float
    sigma_phi_deg: float
    sigma_theta_deg: float


def _load_empirical_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "r": r,
        "dtheta_dr_rad": dtheta_dr_rad,
    }


def _stretch_curve_for_dataset(
    curve: dict[str, np.ndarray], r_max_dataset: float
) -> tuple[dict[str, np.ndarray], float]:
    r_max_empirical = float(np.max(curve["r"]))
    scale = float(r_max_dataset / max(r_max_empirical, 1e-12))
    stretched_r = curve["r"] * scale
    stretched_dtheta_dr_rad = curve["dtheta_dr_rad"] / max(scale, 1e-12)
    return {
        "theta_deg": curve["theta_deg"].copy(),
        "theta_rad": curve["theta_rad"].copy(),
        "r": stretched_r,
        "dtheta_dr_rad": stretched_dtheta_dr_rad,
    }, scale


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _load_xy_series(meta: dict) -> np.ndarray:
    xy_series = meta.get("xy_series")
    if not isinstance(xy_series, list):
        raise ValueError("No xy_series in metadata")
    arr = np.asarray(xy_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape {arr.shape}")
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _collect_points(dataset_dir: Path) -> tuple[list[RodPoint], float, float, float, str]:
    raw_points: list[dict] = []
    r_means: list[float] = []
    for rod_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        name_lc = rod_dir.name.lower()
        if name_lc.startswith("superseded_") or name_lc.startswith("excluded_"):
            continue
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        req = dict(meta.get("requested") or {})
        act = dict(meta.get("actual") or {})
        xy = _load_xy_series(meta)
        if xy.size == 0:
            continue
        x = xy[:, 0]
        y = xy[:, 1]
        r = np.sqrt((x * x) + (y * y))
        r_mean = float(np.mean(r))
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
        sigma_xy = float(np.hypot(sigma_x, sigma_y))
        raw_points.append(
            {
                "rod": rod_dir.name,
                "req": req,
                "act": act,
                "r_mean": r_mean,
                "sigma_xy": sigma_xy,
            }
        )
        r_means.append(r_mean)

    if not raw_points:
        return [], float("nan"), float("nan"), float("nan"), ""

    empirical_curve = _load_empirical_curve(Path.cwd() / EMPIRICAL_CURVE_CSV)
    stretched_curve, stretch_scale = _stretch_curve_for_dataset(empirical_curve, float(max(r_means)))

    reference_row = min(raw_points, key=lambda row: (abs(float(row["req"].get("exp_ms", float("inf"))) - 0.02), str(row["rod"])))
    reference_r_mean = float(reference_row["r_mean"])
    reference_dtheta_dr_rad = _interp(
        np.asarray(stretched_curve["r"], dtype=np.float64),
        np.asarray(stretched_curve["dtheta_dr_rad"], dtype=np.float64),
        reference_r_mean,
    )
    reference_rod = str(reference_row["rod"])

    out: list[RodPoint] = []
    for row in raw_points:
        req = row["req"]
        act = row["act"]
        r_mean = float(row["r_mean"])
        sigma_xy = float(row["sigma_xy"])
        theta_deg = _interp(
            np.asarray(stretched_curve["r"], dtype=np.float64),
            np.asarray(stretched_curve["theta_deg"], dtype=np.float64),
            r_mean,
        )
        sigma_phi_deg = float(np.degrees(sigma_xy / max(2.0 * r_mean, 1e-12)))
        sigma_theta_deg = float(np.degrees(abs(reference_dtheta_dr_rad) * sigma_xy))
        out.append(
            RodPoint(
                rod=str(row["rod"]),
                exposure_ms=float(req.get("exp_ms")),
                gain_analog=float(req.get("gain_analog")),
                max_pixel_value=float(act.get("max_pixel_value", float("nan"))),
                fraction_saturated=float(act.get("fraction_saturated", float("nan"))),
                intensity_p98=float(act.get("intensity_p98", float("nan"))),
                r_mean=r_mean,
                theta_deg=theta_deg,
                sigma_xy=sigma_xy,
                sigma_phi_deg=sigma_phi_deg,
                sigma_theta_deg=sigma_theta_deg,
            )
        )
    return out, float(stretch_scale), reference_r_mean, float(reference_dtheta_dr_rad), reference_rod


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_saturation_region(ax: plt.Axes) -> None:
    x0, x1 = ax.get_xlim()
    right = max(x1, SATURATION_EXPOSURE_MS + 0.04)
    ax.set_xlim(x0, right)
    ax.axvspan(
        SATURATION_EXPOSURE_MS,
        right,
        facecolor="#d9d9d9",
        alpha=0.25,
        hatch="///",
        edgecolor="#999999",
    )
    y0, y1 = ax.get_ylim()
    ax.text(
        SATURATION_EXPOSURE_MS + 0.5 * (right - SATURATION_EXPOSURE_MS),
        y0 + 0.92 * (y1 - y0),
        "At and beyond 0.8 ms rods saturate",
        ha="center",
        va="top",
        color="#555555",
    )


def _plot_metric(
    exposures: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    order = np.argsort(exposures)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(exposures, values, s=42, alpha=0.9, color="#1f77b4")
    ax.plot(exposures[order], values[order], color="#1f77b4", lw=1.6, alpha=0.7)
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    _add_saturation_region(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    dataset_dir = Path.cwd() / DATASET_DIR
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    points, stretch_scale, reference_r_mean, reference_dtheta_dr_rad, reference_rod = _collect_points(dataset_dir)
    if not points:
        raise SystemExit("No usable rods found.")

    rows = []
    for p in points:
        rows.append(
            {
                "rod": p.rod,
                "exposure_ms": p.exposure_ms,
                "gain_analog": p.gain_analog,
                "max_pixel_value": p.max_pixel_value,
                "fraction_saturated": p.fraction_saturated,
                "intensity_p98": p.intensity_p98,
                "r_mean": p.r_mean,
                "theta_deg": p.theta_deg,
                "sigma_xy": p.sigma_xy,
                "sigma_phi_deg": p.sigma_phi_deg,
                "sigma_theta_deg": p.sigma_theta_deg,
            }
        )
    _write_csv(
        out_dir / "precision_vs_exposure_points.csv",
        rows,
        [
            "rod",
            "exposure_ms",
            "gain_analog",
            "max_pixel_value",
            "fraction_saturated",
            "intensity_p98",
            "r_mean",
            "theta_deg",
            "sigma_xy",
            "sigma_phi_deg",
            "sigma_theta_deg",
        ],
    )

    exposures = np.asarray([p.exposure_ms for p in points], dtype=np.float64)
    sigma_phi = np.asarray([p.sigma_phi_deg for p in points], dtype=np.float64)
    sigma_theta = np.asarray([p.sigma_theta_deg for p in points], dtype=np.float64)
    max_px = np.asarray([p.max_pixel_value for p in points], dtype=np.float64)
    frac_sat = np.asarray([p.fraction_saturated for p in points], dtype=np.float64)
    theta_deg = np.asarray([p.theta_deg for p in points], dtype=np.float64)

    _plot_metric(
        exposures,
        sigma_phi,
        ylabel="std dev phi (deg)",
        title="25nm rods precision: phi uncertainty vs exposure",
        out_path=out_dir / "std_phi_vs_exposure.png",
    )
    _plot_metric(
        exposures,
        sigma_theta,
        ylabel="std dev theta (deg)",
        title="25nm rods precision: theta uncertainty vs exposure",
        out_path=out_dir / "std_theta_vs_exposure.png",
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "n_rods": int(len(points)),
        "theta_curve_source": str(Path.cwd() / EMPIRICAL_CURVE_CSV),
        "theta_curve_method": "empirical glycerol theta(r) curve stretched along r to match this dataset r_max",
        "theta_curve_uncertainty_used": False,
        "selection_rule": "directories starting with superseded_ or excluded_ are ignored",
        "exposure_ms_sorted": [float(v) for v in sorted(exposures.tolist())],
        "theta_deg_range": [float(np.min(theta_deg)), float(np.max(theta_deg))],
        "sigma_phi_deg_range": [float(np.min(sigma_phi)), float(np.max(sigma_phi))],
        "sigma_theta_deg_range": [float(np.min(sigma_theta)), float(np.max(sigma_theta))],
        "max_pixel_value_range": [float(np.min(max_px)), float(np.max(max_px))],
        "fraction_saturated_range": [float(np.min(frac_sat)), float(np.max(frac_sat))],
        "stretch_scale_r": float(stretch_scale),
        "saturation_note": "Plots mark exposure >= 0.80 ms as saturation-onset region.",
        "method_phi": "sigma_phi = sigma_xy / (2 r), converted to degrees",
        "method_theta": "sigma_theta = |dtheta/dr|_ref * sigma_xy using the stretched empirical theta(r) center curve only, with fixed reference slope from the 0.02 ms recording",
        "theta_reference_exposure_ms": 0.02,
        "theta_reference_rod": reference_rod,
        "theta_reference_r_mean": float(reference_r_mean),
        "theta_reference_dtheta_dr_rad": float(reference_dtheta_dr_rad),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
