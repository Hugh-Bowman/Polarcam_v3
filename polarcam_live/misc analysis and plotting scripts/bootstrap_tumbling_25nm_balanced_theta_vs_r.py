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


DATASET_ROOT = Path("datasets") / "tumbling 25nm glycerol"
SOURCE_CSV = DATASET_ROOT / "plots" / "balanced_phi_xy_points" / "balanced_sampled_xy_points.csv"
OUTPUT_DIR = DATASET_ROOT / "plots" / "balanced_phi_xy_points"
N_BOOT = 250
BOOT_SEED = 24680


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
) -> tuple[np.ndarray, np.ndarray]:
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
    return centers, theta_deg


def _interp_monotonic_theta(r_centers: np.ndarray, theta_deg: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    if PchipInterpolator is not None:
        fn = PchipInterpolator(r_centers, theta_deg, extrapolate=True)
        out = fn(r_grid)
    else:
        out = np.interp(r_grid, r_centers, theta_deg, left=theta_deg[0], right=theta_deg[-1])
    out = np.maximum.accumulate(np.asarray(out, dtype=np.float64))
    return np.clip(out, 0.0, 90.0)


def _load_r(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return np.asarray([float(row["r"]) for row in rows], dtype=np.float64)


def main() -> None:
    source_csv = Path.cwd() / SOURCE_CSV
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    r = _load_r(source_csv)
    if r.size == 0:
        raise SystemExit("No r values found in balanced sampled points CSV.")

    center_r, center_theta = _fit_uniform_costheta_curve(r)
    r_grid = np.linspace(float(center_r[0]), float(center_r[-1]), 600)
    theta_center_grid = _interp_monotonic_theta(center_r, center_theta, r_grid)

    rng = np.random.default_rng(int(BOOT_SEED))
    curves: list[np.ndarray] = []
    n = int(r.size)
    for _ in range(int(N_BOOT)):
        idx = rng.integers(0, n, size=n)
        r_boot = r[idx]
        try:
            rb, tb = _fit_uniform_costheta_curve(r_boot)
            curve = _interp_monotonic_theta(rb, tb, r_grid)
        except Exception:
            continue
        curves.append(curve)

    if not curves:
        raise SystemExit("Bootstrap failed to produce any theta(r) curves.")

    curves_arr = np.asarray(curves, dtype=np.float64)
    theta_lo = np.percentile(curves_arr, 16.0, axis=0)
    theta_hi = np.percentile(curves_arr, 84.0, axis=0)
    theta_std = np.std(curves_arr, axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros_like(theta_center_grid)

    rows = []
    for rv, cv, lv, hv, sv in zip(r_grid, theta_center_grid, theta_lo, theta_hi, theta_std):
        rows.append(
            {
                "r": float(rv),
                "theta_deg_center": float(cv),
                "theta_deg_lo_1sigma": float(lv),
                "theta_deg_hi_1sigma": float(hv),
                "theta_deg_std": float(sv),
            }
        )
    with (out_dir / "theta_vs_r_balanced_sampled_points_uncertainty_band.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["r", "theta_deg_center", "theta_deg_lo_1sigma", "theta_deg_hi_1sigma", "theta_deg_std"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(
        r_grid,
        theta_lo,
        theta_hi,
        color="#2ca02c",
        alpha=0.22,
        label="Bootstrap 1 sigma band",
    )
    ax.plot(r_grid, theta_center_grid, color="#1b7f3a", lw=2.3, label="Theta(r) empirical")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) from balanced tumbling 25nm xy points")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_balanced_sampled_points_with_uncertainty.png", dpi=220)
    plt.close(fig)

    summary = {
        "source_csv": str(source_csv.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_points": int(r.size),
        "n_boot": int(len(curves)),
        "seed": int(BOOT_SEED),
        "method": "Bootstrap resample the balanced sampled r-values with replacement, refit theta(r) for each resample, then use the 16th/84th percentiles across curves as the 1 sigma band.",
    }
    (out_dir / "theta_vs_r_balanced_sampled_points_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
