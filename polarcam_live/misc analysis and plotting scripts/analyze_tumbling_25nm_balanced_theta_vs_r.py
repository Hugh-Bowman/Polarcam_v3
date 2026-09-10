from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


POINTS_CSV = (
    Path("datasets") / "tumbling 25nm glycerol"
    / "plots"
    / "balanced_phi_xy_points"
    / "balanced_sampled_xy_points.csv"
)
OUTPUT_DIR = Path("datasets") / "tumbling 25nm glycerol" / "plots" / "balanced_phi_xy_points"


def _gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    sigma = float(max(0.0, sigma_bins))
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(4.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _gaussian_smooth_hist(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_bins)
    if kernel.size == 1:
        return counts.copy()
    return np.convolve(counts, kernel, mode="same")


def _fit_uniform_costheta_curve(
    r_values: np.ndarray,
    bins: int = 160,
    sigma_bins: float = 3.0,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r_values, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("No finite r values to fit.")
    lo = float(np.percentile(r, lo_pct))
    hi = float(np.percentile(r, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid r fit range.")
    counts, edges = np.histogram(r, bins=int(bins), range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    smooth_density = _gaussian_smooth_hist(counts, sigma_bins=sigma_bins)
    smooth_density = np.maximum(smooth_density, 0.0)
    area = np.sum(smooth_density * widths)
    if area > 0.0:
        smooth_density = smooth_density / area
    cdf = np.cumsum(smooth_density * widths)
    cdf = np.clip(cdf, 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    theta_deg = np.maximum.accumulate(theta_deg)
    return centers, smooth_density, cdf, theta_deg


def _interp_monotonic_theta(r_centers: np.ndarray, theta_deg: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    if PchipInterpolator is not None:
        fn = PchipInterpolator(r_centers, theta_deg, extrapolate=True)
        out = fn(r_grid)
    else:
        out = np.interp(r_grid, r_centers, theta_deg, left=theta_deg[0], right=theta_deg[-1])
    out = np.maximum.accumulate(np.asarray(out, dtype=np.float64))
    return np.clip(out, 0.0, 90.0)


def _theta_curve_density(r_grid: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    theta_rad = np.radians(theta_deg)
    dtheta_dr = np.gradient(theta_rad, r_grid)
    density = np.sin(theta_rad) * np.maximum(dtheta_dr, 0.0)
    density = np.where(np.isfinite(density), density, 0.0)
    area = np.trapezoid(density, r_grid)
    if area > 0.0:
        density = density / area
    return density


def _load_r_values(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("No rows in balanced sampled points CSV.")
    return np.asarray([float(row["r"]) for row in rows], dtype=np.float64)


def main() -> None:
    points_csv = Path.cwd() / POINTS_CSV
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    r = _load_r_values(points_csv)
    r_centers, smooth_density, cdf, theta_deg = _fit_uniform_costheta_curve(r)
    r_grid = np.linspace(float(r_centers[0]), float(r_centers[-1]), 600)
    theta_grid_deg = _interp_monotonic_theta(r_centers, theta_deg, r_grid)
    density_from_curve = _theta_curve_density(r_grid, theta_grid_deg)

    rows = []
    smooth_density_grid = np.interp(
        r_grid,
        r_centers,
        smooth_density,
        left=float(smooth_density[0]),
        right=float(smooth_density[-1]),
    )
    cdf_grid = np.interp(
        r_grid,
        r_centers,
        cdf,
        left=float(cdf[0]),
        right=float(cdf[-1]),
    )
    for rv, thv, cdfv, sdv, dfv in zip(r_grid, theta_grid_deg, cdf_grid, smooth_density_grid, density_from_curve):
        rows.append(
            {
                "r": float(rv),
                "theta_deg": float(thv),
                "cos_theta": float(np.cos(np.radians(thv))),
                "cdf": float(cdfv),
                "smoothed_r_density": float(sdv),
                "density_from_theta_curve": float(dfv),
            }
        )
    with (out_dir / "theta_vs_r_curve_balanced_sampled_points.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["r", "theta_deg", "cos_theta", "cdf", "smoothed_r_density", "density_from_theta_curve"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(r_grid, theta_grid_deg, color="#2ca02c", lw=2.2)
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) from balanced tumbling 25nm xy points")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_balanced_sampled_points.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(r_centers, smooth_density, color="#1f77b4", lw=2.0, label="Smoothed measured r density")
    ax.plot(r_grid, density_from_curve, color="#d62728", lw=1.8, ls="--", label="Density implied by theta(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
    ax.set_title("r density and fitted theta(r) consistency")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "r_density_vs_theta_curve_balanced_sampled_points.png", dpi=220)
    plt.close(fig)

    summary = {
        "points_csv": str(points_csv.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_points": int(r.size),
        "r_fit_range": [float(np.min(r_grid)), float(np.max(r_grid))],
        "r_data_range": [float(np.min(r)), float(np.max(r))],
        "method": "Fit theta(r) so that the sampled balanced-point r distribution corresponds to uniform cos(theta).",
    }
    (out_dir / "theta_vs_r_balanced_sampled_points_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
