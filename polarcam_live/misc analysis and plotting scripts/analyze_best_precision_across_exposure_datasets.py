from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")


DATASET_DIRS = [
    Path("40nm precision vs exposure") / "pending",
    Path("precision vs exposure 02072026") / "good",
    Path("precision vs exposure 02072026") / "pending",
]
OUTPUT_DIR = Path("40nm precision exposure relation final") / "plots" / "precision_vs_exposure"
EMPIRICAL_CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
MAX_EXPOSURE_MS = 0.30


@dataclass
class RodPoint:
    source_dir: str
    rod: str
    exposure_ms: float
    gain_analog: float
    max_pixel_value: float
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
    return {"r": r, "theta_deg": theta_deg, "dtheta_dr_rad": dtheta_dr_rad}


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], r_max_dataset: float) -> tuple[dict[str, np.ndarray], float]:
    r_max_empirical = float(np.max(curve["r"]))
    scale = float(r_max_dataset / max(r_max_empirical, 1e-12))
    return {
        "r": curve["r"] * scale,
        "theta_deg": curve["theta_deg"].copy(),
        "dtheta_dr_rad": curve["dtheta_dr_rad"] / max(scale, 1e-12),
    }, scale


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _load_xy_series(meta: dict) -> np.ndarray:
    arr = np.asarray(meta.get("xy_series"), dtype=np.float64)
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _collect_raw_rows(dataset_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for dataset_dir in dataset_dirs:
        full_dir = Path.cwd() / dataset_dir
        if not full_dir.exists():
            continue
        for rod_dir in sorted([p for p in full_dir.iterdir() if p.is_dir()]):
            name_lc = rod_dir.name.lower()
            if name_lc.startswith("superseded_") or name_lc.startswith("excluded_"):
                continue
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            req = dict(meta.get("requested") or {})
            act = dict(meta.get("actual") or {})
            exp_ms = float(req.get("exp_ms", float("nan")))
            if not np.isfinite(exp_ms) or exp_ms > MAX_EXPOSURE_MS:
                continue
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
            rows.append(
                {
                    "source_dir": str(dataset_dir),
                    "rod": rod_dir.name,
                    "exposure_ms": exp_ms,
                    "gain_analog": float(req.get("gain_analog", float("nan"))),
                    "max_pixel_value": float(act.get("max_pixel_value", float("nan"))),
                    "intensity_p98": float(act.get("intensity_p98", float("nan"))),
                    "r_mean": r_mean,
                    "sigma_xy": sigma_xy,
                }
            )
    return rows


def _finalize_points(raw_rows: list[dict]) -> tuple[list[RodPoint], float]:
    if not raw_rows:
        return [], float("nan")
    empirical_curve = _load_empirical_curve(Path.cwd() / EMPIRICAL_CURVE_CSV)
    stretched_curve, scale = _stretch_curve_for_dataset(
        empirical_curve,
        max(float(r["r_mean"]) for r in raw_rows),
    )
    points: list[RodPoint] = []
    for row in raw_rows:
        r_mean = float(row["r_mean"])
        sigma_xy = float(row["sigma_xy"])
        theta_deg = _interp(stretched_curve["r"], stretched_curve["theta_deg"], r_mean)
        dtheta_dr_rad = _interp(stretched_curve["r"], stretched_curve["dtheta_dr_rad"], r_mean)
        sigma_phi_deg = float(np.degrees(sigma_xy / max(2.0 * r_mean, 1e-12)))
        sigma_theta_deg = float(np.degrees(abs(dtheta_dr_rad) * sigma_xy))
        points.append(
            RodPoint(
                source_dir=str(row["source_dir"]),
                rod=str(row["rod"]),
                exposure_ms=float(row["exposure_ms"]),
                gain_analog=float(row["gain_analog"]),
                max_pixel_value=float(row["max_pixel_value"]),
                intensity_p98=float(row["intensity_p98"]),
                r_mean=r_mean,
                theta_deg=theta_deg,
                sigma_xy=sigma_xy,
                sigma_phi_deg=sigma_phi_deg,
                sigma_theta_deg=sigma_theta_deg,
            )
        )
    return points, scale


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_by_exposure(points: list[RodPoint], attr: str) -> list[RodPoint]:
    out: list[RodPoint] = []
    exposures = sorted({float(p.exposure_ms) for p in points})
    for exp in exposures:
        subset = [p for p in points if abs(float(p.exposure_ms) - exp) < 1e-12]
        out.append(min(subset, key=lambda p: float(getattr(p, attr))))
    return out


def _plot_best(
    all_points: list[RodPoint],
    best_points: list[RodPoint],
    y_attr: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    x_all = np.asarray([p.exposure_ms for p in all_points], dtype=np.float64)
    y_all = np.asarray([getattr(p, y_attr) for p in all_points], dtype=np.float64)
    x_best = np.asarray([p.exposure_ms for p in best_points], dtype=np.float64)
    y_best = np.asarray([getattr(p, y_attr) for p in best_points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(x_all, y_all, s=28, alpha=0.35, color="#9aa0a6", label="All observed")
    ax.scatter(x_best, y_best, s=52, alpha=0.95, color="#1f77b4", label="Best observed")
    ax.plot(x_best, y_best, lw=1.8, color="#1f77b4")
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _collect_raw_rows(DATASET_DIRS)
    points, stretch_scale = _finalize_points(raw_rows)
    if not points:
        raise SystemExit("No usable points found.")

    all_rows = []
    for p in points:
        all_rows.append(
            {
                "source_dir": p.source_dir,
                "rod": p.rod,
                "exposure_ms": p.exposure_ms,
                "gain_analog": p.gain_analog,
                "max_pixel_value": p.max_pixel_value,
                "intensity_p98": p.intensity_p98,
                "r_mean": p.r_mean,
                "theta_deg": p.theta_deg,
                "sigma_xy": p.sigma_xy,
                "sigma_phi_deg": p.sigma_phi_deg,
                "sigma_theta_deg": p.sigma_theta_deg,
            }
        )
    _write_csv(
        out_dir / "all_points_up_to_0p30ms.csv",
        all_rows,
        ["source_dir", "rod", "exposure_ms", "gain_analog", "max_pixel_value", "intensity_p98", "r_mean", "theta_deg", "sigma_xy", "sigma_phi_deg", "sigma_theta_deg"],
    )

    best_phi = _best_by_exposure(points, "sigma_phi_deg")
    best_theta = _best_by_exposure(points, "sigma_theta_deg")

    _write_csv(
        out_dir / "best_sigma_phi_by_exposure.csv",
        [
            {
                "exposure_ms": p.exposure_ms,
                "source_dir": p.source_dir,
                "rod": p.rod,
                "sigma_phi_deg": p.sigma_phi_deg,
            }
            for p in best_phi
        ],
        ["exposure_ms", "source_dir", "rod", "sigma_phi_deg"],
    )
    _write_csv(
        out_dir / "best_sigma_theta_by_exposure.csv",
        [
            {
                "exposure_ms": p.exposure_ms,
                "source_dir": p.source_dir,
                "rod": p.rod,
                "sigma_theta_deg": p.sigma_theta_deg,
            }
            for p in best_theta
        ],
        ["exposure_ms", "source_dir", "rod", "sigma_theta_deg"],
    )

    _plot_best(
        points,
        best_phi,
        "sigma_phi_deg",
        "std dev phi (deg)",
        "Best observed phi precision vs exposure up to 0.30 ms",
        out_dir / "best_std_phi_vs_exposure.png",
    )
    _plot_best(
        points,
        best_theta,
        "sigma_theta_deg",
        "std dev theta (deg)",
        "Best observed theta precision vs exposure up to 0.30 ms",
        out_dir / "best_std_theta_vs_exposure.png",
    )

    summary = {
        "dataset_dirs": [str(Path.cwd() / p) for p in DATASET_DIRS],
        "output_dir": str(out_dir),
        "n_points_total": int(len(points)),
        "max_exposure_ms": float(MAX_EXPOSURE_MS),
        "theta_curve_source": str(Path.cwd() / EMPIRICAL_CURVE_CSV),
        "theta_curve_method": "empirical glycerol theta(r) curve stretched along r to match the combined dataset r_max",
        "stretch_scale": float(stretch_scale),
        "best_exposures": [float(p.exposure_ms) for p in best_phi],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
