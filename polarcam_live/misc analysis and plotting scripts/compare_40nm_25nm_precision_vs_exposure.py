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


EMPIRICAL_CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)

DATASETS = {
    "40nm": {
        "dir": Path("40nm precision exposure relation final") / "pending",
        "label": "40x65nm",
    },
    "25nm": {
        "dir": Path("various exposure recording of 25nm rods 03072026"),
        "label": "25x65nm",
    },
}

OUTPUT_DIR = Path("various exposure recording of 25nm rods 03072026")
TARGET_EXPOSURE_MS = 0.02


@dataclass
class RodPoint:
    rod: str
    exposure_ms: float
    r_mean: float
    sigma_xy: float


@dataclass
class DatasetSeries:
    key: str
    label: str
    points: list[RodPoint]
    stretch_scale_r: float
    ref_rod: str
    ref_exposure_ms: float
    ref_r_mean: float
    ref_dtheta_dr_rad: float


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


def _load_dataset_series(key: str, cfg: dict, curve: dict[str, np.ndarray]) -> DatasetSeries:
    dataset_dir = Path.cwd() / cfg["dir"]
    raw_points: list[RodPoint] = []
    r_means: list[float] = []
    for rod_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        req = dict(meta.get("requested") or {})
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
            RodPoint(
                rod=rod_dir.name,
                exposure_ms=float(req.get("exp_ms")),
                r_mean=r_mean,
                sigma_xy=sigma_xy,
            )
        )
        r_means.append(r_mean)
    if not raw_points:
        raise RuntimeError(f"No usable rods found in {dataset_dir}")

    stretched_curve, stretch_scale = _stretch_curve_for_dataset(curve, float(max(r_means)))
    ref_point = min(raw_points, key=lambda p: (abs(float(p.exposure_ms) - TARGET_EXPOSURE_MS), p.rod))
    ref_dtheta_dr_rad = _interp(
        np.asarray(stretched_curve["r"], dtype=np.float64),
        np.asarray(stretched_curve["dtheta_dr_rad"], dtype=np.float64),
        float(ref_point.r_mean),
    )
    raw_points.sort(key=lambda p: (p.exposure_ms, p.rod))
    return DatasetSeries(
        key=key,
        label=str(cfg["label"]),
        points=raw_points,
        stretch_scale_r=float(stretch_scale),
        ref_rod=str(ref_point.rod),
        ref_exposure_ms=float(ref_point.exposure_ms),
        ref_r_mean=float(ref_point.r_mean),
        ref_dtheta_dr_rad=float(ref_dtheta_dr_rad),
    )


def _phi_values(series: DatasetSeries, fixed_r: bool) -> np.ndarray:
    ref_r = float(series.ref_r_mean)
    out = []
    for p in series.points:
        r_use = ref_r if fixed_r else float(p.r_mean)
        out.append(float(np.degrees(float(p.sigma_xy) / max(2.0 * r_use, 1e-12))))
    return np.asarray(out, dtype=np.float64)


def _theta_values(series: DatasetSeries, fixed_r: bool, curve: dict[str, np.ndarray]) -> np.ndarray:
    stretched_curve, _scale = _stretch_curve_for_dataset(curve, float(max(p.r_mean for p in series.points)))
    out = []
    for p in series.points:
        if fixed_r:
            dtheta_dr = float(series.ref_dtheta_dr_rad)
        else:
            dtheta_dr = _interp(
                np.asarray(stretched_curve["r"], dtype=np.float64),
                np.asarray(stretched_curve["dtheta_dr_rad"], dtype=np.float64),
                float(p.r_mean),
            )
        out.append(float(np.degrees(abs(dtheta_dr) * float(p.sigma_xy))))
    return np.asarray(out, dtype=np.float64)


def _plot_two_dataset(
    exposures_a: np.ndarray,
    values_a: np.ndarray,
    label_a: str,
    exposures_b: np.ndarray,
    values_b: np.ndarray,
    label_b: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    order_a = np.argsort(exposures_a)
    order_b = np.argsort(exposures_b)
    ax.scatter(exposures_a, values_a, s=42, alpha=0.9, color="#1f77b4", label=label_a)
    ax.plot(exposures_a[order_a], values_a[order_a], color="#1f77b4", lw=1.6, alpha=0.75)
    ax.scatter(exposures_b, values_b, s=42, alpha=0.9, color="#d62728", label=label_b)
    ax.plot(exposures_b[order_b], values_b[order_b], color="#d62728", lw=1.6, alpha=0.75)
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _copy_single_dataset_plots() -> None:
    src_dir = OUTPUT_DIR / "plots" / "precision_vs_exposure"
    for name in ("std_phi_vs_exposure.png", "std_theta_vs_exposure.png"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / name)


def main() -> None:
    curve = _load_empirical_curve(Path.cwd() / EMPIRICAL_CURVE_CSV)
    s40 = _load_dataset_series("40nm", DATASETS["40nm"], curve)
    s25 = _load_dataset_series("25nm", DATASETS["25nm"], curve)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _copy_single_dataset_plots()

    exp40 = np.asarray([p.exposure_ms for p in s40.points], dtype=np.float64)
    exp25 = np.asarray([p.exposure_ms for p in s25.points], dtype=np.float64)

    phi40_var = _phi_values(s40, fixed_r=False)
    theta40_var = _theta_values(s40, fixed_r=False, curve=curve)
    phi40_fix = _phi_values(s40, fixed_r=True)
    theta40_fix = _theta_values(s40, fixed_r=True, curve=curve)

    phi25_fix = _phi_values(s25, fixed_r=True)
    theta25_fix = _theta_values(s25, fixed_r=True, curve=curve)

    _plot_two_dataset(
        exp40,
        phi40_var,
        "40x65nm",
        exp25,
        phi25_fix,
        "25x65nm",
        ylabel="std dev phi (deg)",
        title="Phi uncertainty vs exposure",
        out_path=OUTPUT_DIR / "compare_std_phi_vs_exposure_25fixed_40variable.png",
    )
    _plot_two_dataset(
        exp40,
        theta40_var,
        "40x65nm",
        exp25,
        theta25_fix,
        "25x65nm",
        ylabel="std dev theta (deg)",
        title="Theta uncertainty vs exposure",
        out_path=OUTPUT_DIR / "compare_std_theta_vs_exposure_25fixed_40variable.png",
    )
    _plot_two_dataset(
        exp40,
        phi40_fix,
        "40x65nm",
        exp25,
        phi25_fix,
        "25x65nm",
        ylabel="std dev phi (deg)",
        title="Phi uncertainty vs exposure",
        out_path=OUTPUT_DIR / "compare_std_phi_vs_exposure_both_fixed.png",
    )
    _plot_two_dataset(
        exp40,
        theta40_fix,
        "40x65nm",
        exp25,
        theta25_fix,
        "25x65nm",
        ylabel="std dev theta (deg)",
        title="Theta uncertainty vs exposure: both datasets fixed r",
        out_path=OUTPUT_DIR / "compare_std_theta_vs_exposure_both_fixed.png",
    )

    summary = {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "curve_csv": str((Path.cwd() / EMPIRICAL_CURVE_CSV).resolve()),
        "mode_1": {
            "description": "25nm fixed at first-capture r and dtheta/dr; 40nm varying local r and dtheta/dr",
            "files": [
                "compare_std_phi_vs_exposure_25fixed_40variable.png",
                "compare_std_theta_vs_exposure_25fixed_40variable.png",
            ],
        },
        "mode_2": {
            "description": "Both 25nm and 40nm fixed at their own first-capture r and dtheta/dr",
            "files": [
                "compare_std_phi_vs_exposure_both_fixed.png",
                "compare_std_theta_vs_exposure_both_fixed.png",
            ],
        },
        "datasets": {
            "40nm": {
                "dataset_dir": str((Path.cwd() / DATASETS["40nm"]["dir"]).resolve()),
                "n_points": len(s40.points),
                "stretch_scale_r": float(s40.stretch_scale_r),
                "reference_rod": s40.ref_rod,
                "reference_exposure_ms": float(s40.ref_exposure_ms),
                "reference_r_mean": float(s40.ref_r_mean),
                "reference_dtheta_dr_rad": float(s40.ref_dtheta_dr_rad),
            },
            "25nm": {
                "dataset_dir": str((Path.cwd() / DATASETS["25nm"]["dir"]).resolve()),
                "n_points": len(s25.points),
                "stretch_scale_r": float(s25.stretch_scale_r),
                "reference_rod": s25.ref_rod,
                "reference_exposure_ms": float(s25.ref_exposure_ms),
                "reference_r_mean": float(s25.ref_r_mean),
                "reference_dtheta_dr_rad": float(s25.ref_dtheta_dr_rad),
            },
        },
    }
    (OUTPUT_DIR / "compare_40nm_25nm_precision_vs_exposure_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
