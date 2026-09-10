from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


RANGE_THRESHOLD = 1.0
PHI_BIN_DEG = 5.0
RNG_SEED_BALANCE = 12345
N_BOOT = 250
BOOT_SEED = 86420

DATASETS = [
    {
        "name": "25nm_full",
        "root": Path("datasets") / "tumbling 25nm glycerol",
        "output_dir": Path("datasets") / "tumbling 25nm glycerol" / "plots" / "balanced_phi_xy_points",
        "title": "Theta(r) from balanced tumbling 25nm xy points",
        "color_fill": "#2ca02c",
        "color_line": "#1b7f3a",
    },
    {
        "name": "40nm_full",
        "root": Path("glycerol suspended rods 17062026"),
        "output_dir": Path("glycerol suspended rods 17062026") / "plots" / "balanced_phi_xy_points_all_recordings",
        "title": "Theta(r) from balanced 40nm glycerol xy points",
        "color_fill": "#d62728",
        "color_line": "#a51c30",
    },
]


@dataclass
class Recording:
    source: str
    rod: str
    x: np.ndarray
    y: np.ndarray


def _phi_deg_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.degrees(np.mod(0.5 * np.arctan2(y, x), np.pi))


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


def _load_recordings(root: Path) -> list[Recording]:
    out: list[Recording] = []
    for source in ("pending", "good", "bad"):
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
            out.append(
                Recording(
                    source=source,
                    rod=rod_dir.name,
                    x=np.asarray(xy[:, 0], dtype=np.float64),
                    y=np.asarray(xy[:, 1], dtype=np.float64),
                )
            )
    return out


def _sample_balanced_r_from_recordings(recordings: list[Recording], rng_seed: int) -> tuple[np.ndarray, dict]:
    x = np.concatenate([rec.x for rec in recordings])
    y = np.concatenate([rec.y for rec in recordings])
    phi_deg = _phi_deg_from_xy(x, y)
    r = np.sqrt((x * x) + (y * y))
    edges = np.arange(0.0, 180.0 + PHI_BIN_DEG, PHI_BIN_DEG, dtype=np.float64)
    bin_ids = np.digitize(phi_deg, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    idx_by_bin = [np.flatnonzero(bin_ids == i) for i in range(len(edges) - 1)]
    counts = [int(idx.size) for idx in idx_by_bin]
    min_count = int(min(counts))
    n_per_bin = max(1, min_count // 2)
    rng = np.random.default_rng(int(rng_seed))
    sampled_idx = [np.asarray(rng.choice(idx, size=n_per_bin, replace=False), dtype=np.int64) for idx in idx_by_bin]
    sampled_idx_arr = np.concatenate(sampled_idx)
    return r[sampled_idx_arr], {
        "n_pooled_points": int(r.size),
        "least_populated_bin_count": int(min_count),
        "sampled_per_bin": int(n_per_bin),
        "n_selected_points": int(sampled_idx_arr.size),
    }


def _recording_bootstrap_curve(recordings: list[Recording]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    r_center_sample, sample_meta = _sample_balanced_r_from_recordings(recordings, rng_seed=RNG_SEED_BALANCE)
    center_r, center_theta = _fit_uniform_costheta_curve(r_center_sample)
    r_grid = np.linspace(float(center_r[0]), float(center_r[-1]), 600)
    theta_center = _interp_monotonic_theta(center_r, center_theta, r_grid)

    rng = np.random.default_rng(int(BOOT_SEED))
    curves: list[np.ndarray] = []
    n_rec = len(recordings)
    for boot_i in range(int(N_BOOT)):
        idx = rng.integers(0, n_rec, size=n_rec)
        sampled_recs = [recordings[int(i)] for i in idx]
        try:
            r_boot, _meta = _sample_balanced_r_from_recordings(sampled_recs, rng_seed=RNG_SEED_BALANCE + boot_i + 1)
            rb, tb = _fit_uniform_costheta_curve(r_boot)
            curve = _interp_monotonic_theta(rb, tb, r_grid)
        except Exception:
            continue
        curves.append(curve)
    if not curves:
        raise SystemExit("Recording bootstrap failed.")
    curves_arr = np.asarray(curves, dtype=np.float64)
    theta_lo = np.percentile(curves_arr, 16.0, axis=0)
    theta_hi = np.percentile(curves_arr, 84.0, axis=0)
    theta_std = np.std(curves_arr, axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros_like(theta_center)
    meta = {
        **sample_meta,
        "n_boot_completed": int(curves_arr.shape[0]),
    }
    return r_grid, theta_center, theta_lo, theta_hi, theta_std, meta


def _write_curve_csv(path: Path, r_grid: np.ndarray, center: np.ndarray, lo: np.ndarray, hi: np.ndarray, std: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["r", "theta_deg_center", "theta_deg_lo_1sigma", "theta_deg_hi_1sigma", "theta_deg_std"],
        )
        writer.writeheader()
        for rv, cv, lv, hv, sv in zip(r_grid, center, lo, hi, std):
            writer.writerow(
                {
                    "r": float(rv),
                    "theta_deg_center": float(cv),
                    "theta_deg_lo_1sigma": float(lv),
                    "theta_deg_hi_1sigma": float(hv),
                    "theta_deg_std": float(sv),
                }
            )


def _plot_curve(path: Path, title: str, fill_color: str, line_color: str, r_grid: np.ndarray, center: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(r_grid, lo, hi, color=fill_color, alpha=0.22, label="Recording-bootstrap 1 sigma band")
    ax.plot(r_grid, center, color=line_color, lw=2.3, label="Theta(r) empirical")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    all_summary: dict[str, dict] = {}
    for cfg in DATASETS:
        root = Path.cwd() / cfg["root"]
        out_dir = Path.cwd() / cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        recordings = _load_recordings(root)
        if not recordings:
            raise SystemExit(f"No recordings loaded for {cfg['name']}")
        r_grid, center, lo, hi, std, meta = _recording_bootstrap_curve(recordings)
        _write_curve_csv(out_dir / "theta_vs_r_balanced_sampled_points_recording_bootstrap_uncertainty_band.csv", r_grid, center, lo, hi, std)
        _plot_curve(
            out_dir / "theta_vs_r_balanced_sampled_points_recording_bootstrap_with_uncertainty.png",
            cfg["title"],
            cfg["color_fill"],
            cfg["color_line"],
            r_grid,
            center,
            lo,
            hi,
        )
        summary = {
            "dataset_root": str(root.resolve()),
            "output_dir": str(out_dir.resolve()),
            "n_recordings_passing_filter": int(len(recordings)),
            "recordings_by_source": {
                src: int(sum(1 for r in recordings if r.source == src)) for src in ("pending", "good", "bad")
            },
            "n_pooled_points": int(meta["n_pooled_points"]),
            "least_populated_bin_count": int(meta["least_populated_bin_count"]),
            "sampled_per_bin": int(meta["sampled_per_bin"]),
            "n_selected_points": int(meta["n_selected_points"]),
            "n_boot_completed": int(meta["n_boot_completed"]),
            "balance_seed": int(RNG_SEED_BALANCE),
            "boot_seed": int(BOOT_SEED),
            "method": "Recording-level bootstrap: resample recordings with replacement, repool points, apply the same 5 degree phi-bin balancing and half-of-least-populated-bin sampling, refit theta(r), and take the 16th/84th percentiles as the 1 sigma band.",
        }
        (out_dir / "theta_vs_r_balanced_sampled_points_recording_bootstrap_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        all_summary[cfg["name"]] = summary

    (Path.cwd() / "tumbling 25nm glycerol" / "plots" / "balanced_phi_xy_points" / "recording_bootstrap_all_summary.json").write_text(
        json.dumps(all_summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
