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


GOOD_DIR = Path("stationary rod data 01072026") / "good"
CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
OUTPUT_DIR = Path("stationary rod data 01072026") / "plots" / "theta_phi_uncertainty_from_empirical_curve"


@dataclass
class RodUncertainty:
    rod: str
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


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
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


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(
        x_new,
        arr_x,
        arr_y,
        left=float(arr_y[0]),
        right=float(arr_y[-1]),
    )


def _load_rod_uncertainties(good_dir: Path, curve: dict[str, np.ndarray]) -> list[RodUncertainty]:
    rods: list[RodUncertainty] = []
    for rod_dir in sorted([p for p in good_dir.iterdir() if p.is_dir()]):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        if not meta_path.exists():
            continue
        arr = _load_xy_series(meta_path)
        if arr.size == 0:
            continue
        x = arr[:, 0]
        y = arr[:, 1]
        r = np.sqrt((x * x) + (y * y))
        r_mean = float(np.mean(r))
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
        sigma_xy = float(np.sqrt((sigma_x * sigma_x) + (sigma_y * sigma_y)))

        theta_deg = float(_interp(curve["r"], curve["theta_deg"], np.asarray([r_mean], dtype=np.float64))[0])
        theta_std_deg = float(_interp(curve["r"], curve["theta_std_deg"], np.asarray([r_mean], dtype=np.float64))[0])
        dtheta_dr_rad = float(_interp(curve["r"], curve["dtheta_dr_rad"], np.asarray([r_mean], dtype=np.float64))[0])

        sigma_phi_rad = sigma_xy / max(2.0 * r_mean, 1e-12)
        sigma_theta_xy_rad = abs(dtheta_dr_rad) * sigma_xy

        sigma_phi_deg = float(np.degrees(sigma_phi_rad))
        sigma_theta_xy_only_deg = float(np.degrees(sigma_theta_xy_rad))
        sigma_theta_total_deg = float(np.sqrt((sigma_theta_xy_only_deg ** 2) + (theta_std_deg ** 2)))

        rods.append(
            RodUncertainty(
                rod=rod_dir.name,
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
    return rods


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    good_dir = Path.cwd() / GOOD_DIR
    curve_csv = Path.cwd() / CURVE_CSV
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_curve_with_uncertainty(curve_csv)
    rods = _load_rod_uncertainties(good_dir, curve)
    if not rods:
        raise SystemExit("No rod uncertainties could be computed from the good folder.")

    rows = []
    for rod in rods:
        rows.append(
            {
                "rod": rod.rod,
                "n_frames": rod.n_frames,
                "r_mean": rod.r_mean,
                "sigma_x": rod.sigma_x,
                "sigma_y": rod.sigma_y,
                "sigma_xy": rod.sigma_xy,
                "theta_deg": rod.theta_deg,
                "sigma_phi_deg": rod.sigma_phi_deg,
                "sigma_theta_xy_only_deg": rod.sigma_theta_xy_only_deg,
                "sigma_theta_curve_deg": rod.sigma_theta_curve_deg,
                "sigma_theta_total_deg": rod.sigma_theta_total_deg,
            }
        )
    _write_csv(
        out_dir / "theta_phi_uncertainty_points.csv",
        rows,
        [
            "rod",
            "n_frames",
            "r_mean",
            "sigma_x",
            "sigma_y",
            "sigma_xy",
            "theta_deg",
            "sigma_phi_deg",
            "sigma_theta_xy_only_deg",
            "sigma_theta_curve_deg",
            "sigma_theta_total_deg",
        ],
    )

    th = np.asarray([r.theta_deg for r in rods], dtype=np.float64)
    sphi = np.asarray([r.sigma_phi_deg for r in rods], dtype=np.float64)
    sth_xy = np.asarray([r.sigma_theta_xy_only_deg for r in rods], dtype=np.float64)
    sth_total = np.asarray([r.sigma_theta_total_deg for r in rods], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sphi, s=40, alpha=0.88, color="#1f77b4")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("Standard deviation in phi vs theta")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "std_phi_vs_theta.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sth_xy, s=40, alpha=0.88, color="#2ca02c")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("Standard deviation in theta vs theta from XY propagation only")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "std_theta_vs_theta_xy_only.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sth_total, s=40, alpha=0.88, color="#d62728", label="XY + theta(r) curve uncertainty")
    ax.scatter(th, sth_xy, s=28, alpha=0.45, color="#2ca02c", label="XY only")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("Standard deviation in theta vs theta including theta(r) uncertainty")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "std_theta_vs_theta_with_curve_uncertainty.png", dpi=220)
    plt.close(fig)

    summary = {
        "good_dir": str(good_dir),
        "curve_csv": str(curve_csv),
        "output_dir": str(out_dir),
        "n_rods": int(len(rods)),
        "theta_range_deg": [float(np.min(th)), float(np.max(th))],
        "sigma_phi_deg_range": [float(np.min(sphi)), float(np.max(sphi))],
        "sigma_theta_xy_only_deg_range": [float(np.min(sth_xy)), float(np.max(sth_xy))],
        "sigma_theta_total_deg_range": [float(np.min(sth_total)), float(np.max(sth_total))],
        "method_phi": "sigma_phi = sigma_xy / (2 r), then convert to degrees",
        "method_theta_xy_only": "sigma_theta = |dtheta/dr| * sigma_xy using empirical theta(r) center curve",
        "method_theta_with_curve": "sigma_theta_total = sqrt(sigma_theta_xy_only^2 + sigma_theta_curve^2)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
