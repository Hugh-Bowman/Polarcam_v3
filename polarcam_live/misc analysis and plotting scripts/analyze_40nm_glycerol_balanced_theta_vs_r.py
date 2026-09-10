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


DATASET_ROOT = Path("glycerol suspended rods 17062026")
SUBDIRS = ("pending", "good", "bad")
OUTPUT_DIR = DATASET_ROOT / "plots" / "balanced_phi_xy_points_all_recordings"
PHI_BIN_DEG = 5.0
RANGE_THRESHOLD = 1.0
RNG_SEED = 12345
N_BOOT = 250
BOOT_SEED = 24680


def _phi_deg_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi = 0.5 * np.arctan2(y, x)
    return np.degrees(np.mod(phi, np.pi))


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


def _load_filtered_points(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    counts = {src: 0 for src in SUBDIRS}
    kept_recordings = 0
    for source in SUBDIRS:
        src_dir = root / source
        if not src_dir.exists():
            continue
        for rod_dir in sorted([p for p in src_dir.iterdir() if p.is_dir()]):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("xy_metrics") or {})
            range_x = metrics.get("range_x")
            range_y = metrics.get("range_y")
            if range_x is None or range_y is None:
                continue
            if not (float(range_x) > RANGE_THRESHOLD and float(range_y) > RANGE_THRESHOLD):
                continue
            xy_series = payload.get("xy_series")
            if not isinstance(xy_series, list) or not xy_series:
                continue
            xy = np.asarray(xy_series, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            xy = xy[:, :2]
            valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[valid]
            if xy.size == 0:
                continue
            xs.append(np.asarray(xy[:, 0], dtype=np.float64))
            ys.append(np.asarray(xy[:, 1], dtype=np.float64))
            counts[source] += 1
            kept_recordings += 1
    if not xs:
        raise SystemExit("No recordings passed the range filter.")
    return np.concatenate(xs), np.concatenate(ys), counts, kept_recordings


def _sample_balanced_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    phi_deg = _phi_deg_from_xy(x, y)
    edges = np.arange(0.0, 180.0 + PHI_BIN_DEG, PHI_BIN_DEG, dtype=np.float64)
    bin_ids = np.digitize(phi_deg, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    idx_by_bin = [np.flatnonzero(bin_ids == i) for i in range(len(edges) - 1)]
    counts = [int(idx.size) for idx in idx_by_bin]
    min_count = int(min(counts))
    n_per_bin = max(1, min_count // 2)
    rng = np.random.default_rng(int(RNG_SEED))
    sampled_idx = []
    rows = []
    for i, idx in enumerate(idx_by_bin):
        choose = rng.choice(idx, size=n_per_bin, replace=False)
        sampled_idx.append(np.asarray(choose, dtype=np.int64))
        rows.append(
            {
                "phi_bin_start_deg": float(edges[i]),
                "phi_bin_end_deg": float(edges[i + 1]),
                "available_points": int(idx.size),
                "sampled_points": int(n_per_bin),
            }
        )
    sampled_idx_arr = np.concatenate(sampled_idx)
    return x[sampled_idx_arr], y[sampled_idx_arr], {
        "phi_bin_edges_deg": edges.tolist(),
        "least_populated_bin_count": int(min_count),
        "sampled_per_bin": int(n_per_bin),
        "bin_rows": rows,
    }


def _write_points_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    phi_deg = _phi_deg_from_xy(x, y)
    r = np.sqrt((x * x) + (y * y))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["x", "y", "r", "phi_deg"])
        writer.writeheader()
        for xv, yv, rv, pv in zip(x, y, r, phi_deg):
            writer.writerow({"x": float(xv), "y": float(yv), "r": float(rv), "phi_deg": float(pv)})


def _plot_hist(vals: np.ndarray, bins: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path, color: str, density: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(vals, bins=bins, color=color, alpha=0.88, edgecolor="white", density=density)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _bootstrap_theta_curve(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_r, center_theta = _fit_uniform_costheta_curve(r)
    r_grid = np.linspace(float(center_r[0]), float(center_r[-1]), 600)
    center_curve = _interp_monotonic_theta(center_r, center_theta, r_grid)
    rng = np.random.default_rng(int(BOOT_SEED))
    curves = []
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
    theta_std = np.std(curves_arr, axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros_like(center_curve)
    return r_grid, center_curve, theta_lo, theta_hi, theta_std


def main() -> None:
    root = Path.cwd() / DATASET_ROOT
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, recording_counts, n_recordings = _load_filtered_points(root)
    x_sel, y_sel, sample_info = _sample_balanced_points(x_all, y_all)
    r_sel = np.sqrt((x_sel * x_sel) + (y_sel * y_sel))
    phi_sel = _phi_deg_from_xy(x_sel, y_sel)

    _write_points_csv(out_dir / "balanced_sampled_xy_points.csv", x_sel, y_sel)
    with (out_dir / "phi_bin_sampling_counts.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phi_bin_start_deg", "phi_bin_end_deg", "available_points", "sampled_points"])
        writer.writeheader()
        writer.writerows(list(sample_info["bin_rows"]))

    _plot_hist(
        phi_sel,
        np.arange(0.0, 185.0, 5.0, dtype=np.float64),
        "Phi (deg)",
        "Point count",
        "Balanced phi distribution from pooled 40nm glycerol xy points",
        out_dir / "phi_distribution_balanced_sampled_points.png",
        "#1f77b4",
        density=False,
    )
    _plot_hist(
        r_sel,
        np.linspace(float(np.min(r_sel)), float(np.max(r_sel)), 50),
        "r",
        "Density",
        "r density from balanced phi-bin sampled 40nm glycerol points",
        out_dir / "r_density_balanced_sampled_points.png",
        "#2ca02c",
        density=True,
    )

    r_grid, theta_center, theta_lo, theta_hi, theta_std = _bootstrap_theta_curve(r_sel)
    rows = []
    for rv, cv, lv, hv, sv in zip(r_grid, theta_center, theta_lo, theta_hi, theta_std):
        rows.append(
            {
                "r": float(rv),
                "theta_deg_center": float(cv),
                "theta_deg_lo_1sigma": float(lv),
                "theta_deg_hi_1sigma": float(hv),
                "theta_deg_std": float(sv),
            }
        )
    with (out_dir / "theta_vs_r_balanced_sampled_points_uncertainty_band.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["r", "theta_deg_center", "theta_deg_lo_1sigma", "theta_deg_hi_1sigma", "theta_deg_std"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(r_grid, theta_lo, theta_hi, color="#2ca02c", alpha=0.22, label="Bootstrap 1 sigma band")
    ax.plot(r_grid, theta_center, color="#1b7f3a", lw=2.3, label="Theta(r) empirical")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) from balanced 40nm glycerol xy points")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_balanced_sampled_points_with_uncertainty.png", dpi=220)
    plt.close(fig)

    summary = {
        "dataset_root": str(root.resolve()),
        "output_dir": str(out_dir.resolve()),
        "range_threshold": float(RANGE_THRESHOLD),
        "phi_bin_deg": float(PHI_BIN_DEG),
        "rng_seed": int(RNG_SEED),
        "n_recordings_passing_filter": int(n_recordings),
        "recordings_by_source": recording_counts,
        "n_pooled_points": int(x_all.size),
        "least_populated_bin_count": int(sample_info["least_populated_bin_count"]),
        "sampled_per_bin": int(sample_info["sampled_per_bin"]),
        "n_selected_points": int(x_sel.size),
        "r_range_selected": [float(np.min(r_sel)), float(np.max(r_sel))],
        "n_boot": int(N_BOOT),
        "boot_seed": int(BOOT_SEED),
        "method": "Apply the same 25nm balanced-point method to all 40nm glycerol recordings: filter recordings by range_x > 1 and range_y > 1, pool all xy points, bin by phi in 5 degree bins, sample half of the least populated bin from each phi bin, fit theta(r), then bootstrap by resampling the selected r-values with replacement.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
