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

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:
    PchipInterpolator = None


THETA_RECON_LUT_STEP_DEG = 0.1
THETA_RECON_GLYCEROL50 = {
    "label": "50% glycerol finite-NA",
    "J1": 0.7051533822554514,
    "J2": 0.049316635956229066,
    "J3": 0.11661874844458937,
    "r_max": 0.869268135868078,
}
PREVIOUS_USED_RADIUS_MAX = 0.15
PHI_R_MIN = 0.25
THETA_BOOTSTRAP_RECORDINGS = 250
THETA_BOOTSTRAP_SEED = 12345


@dataclass
class RecordingData:
    rod: str
    source: str
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    phi_deg_axial: np.ndarray

    @property
    def mean_x(self) -> float:
        return float(np.mean(self.x))

    @property
    def mean_y(self) -> float:
        return float(np.mean(self.y))

    @property
    def mean_radius(self) -> float:
        return float(np.hypot(self.mean_x, self.mean_y))


def _theta_recon_lut() -> tuple[np.ndarray, np.ndarray]:
    j1 = float(THETA_RECON_GLYCEROL50["J1"])
    j2 = float(THETA_RECON_GLYCEROL50["J2"])
    j3 = float(THETA_RECON_GLYCEROL50["J3"])
    r_max = float(THETA_RECON_GLYCEROL50["r_max"])
    a = j1 - j2
    b = j1 + j2
    theta_deg = np.arange(0.0, 90.0, THETA_RECON_LUT_STEP_DEG, dtype=np.float64)
    theta_deg = np.append(theta_deg, 90.0)
    theta_rad = np.radians(theta_deg)
    tan2 = np.tan(theta_rad) ** 2
    r_vals = np.full(theta_rad.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(tan2)
    den = (2.0 * j3) + (b * tan2[finite])
    ok = np.isfinite(den) & (den > 0.0)
    r_tmp = np.full(den.shape, np.nan, dtype=np.float64)
    r_tmp[ok] = (a * tan2[finite][ok]) / den[ok]
    r_vals[np.where(finite)[0]] = r_tmp
    if r_vals.size:
        r_vals[0] = 0.0
        r_vals[-1] = r_max
    return r_vals, theta_deg


def _theta_fourkas_deg_from_r(r: np.ndarray) -> np.ndarray:
    lut_r, lut_theta_deg = _theta_recon_lut()
    r_arr = np.asarray(r, dtype=np.float64)
    r_clip = np.clip(r_arr, 0.0, float(THETA_RECON_GLYCEROL50["r_max"]))
    return np.interp(r_clip, lut_r, lut_theta_deg, left=0.0, right=90.0)


def _fourkas_r_density(r_grid: np.ndarray) -> np.ndarray:
    theta_deg = _theta_fourkas_deg_from_r(r_grid)
    theta_rad = np.radians(theta_deg)
    dtheta_dr = np.gradient(theta_rad, r_grid)
    density = np.sin(theta_rad) * np.maximum(dtheta_dr, 0.0)
    density = np.where(np.isfinite(density), density, 0.0)
    area = np.trapezoid(density, r_grid)
    if area > 0.0:
        density = density / area
    return density


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


def _gaussian_smooth_2d(arr: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_bins)
    if kernel.size == 1:
        return arr.copy()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=tmp)
    return out


def _axial_phi_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi = np.degrees(0.5 * np.arctan2(y, x))
    return np.mod(phi, 180.0)


def _load_xy_series(meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    series = payload.get("xy_series")
    if not isinstance(series, list):
        raise ValueError(f"No xy_series list in {meta_path}")
    pts = np.asarray(series, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"xy_series in {meta_path} has unexpected shape {pts.shape}")
    x = pts[:, 0]
    y = pts[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _load_recording_from_dir(rod_dir: Path, source: str) -> RecordingData | None:
    meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
    if not meta_path.exists():
        return None
    x, y = _load_xy_series(meta_path)
    if x.size == 0:
        return None
    r = np.sqrt((x ** 2) + (y ** 2))
    phi_deg_axial = _axial_phi_deg(x, y)
    return RecordingData(
        rod=rod_dir.name,
        source=source,
        x=x,
        y=y,
        r=r,
        phi_deg_axial=phi_deg_axial,
    )


def _load_good_recordings(good_dir: Path) -> list[RecordingData]:
    out: list[RecordingData] = []
    for rod_dir in sorted([p for p in good_dir.iterdir() if p.is_dir()]):
        rec = _load_recording_from_dir(rod_dir, source="good")
        if rec is not None:
            out.append(rec)
    return out


def _load_previous_used_rod_names(points_csv: Path) -> list[str]:
    with points_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh)
        return sorted({str(row["rod"]).strip() for row in rows if str(row.get("rod", "")).strip()})


def _load_previous_used_filtered(good_dir: Path, points_csv: Path, radius_max: float) -> list[RecordingData]:
    out: list[RecordingData] = []
    for rod_name in _load_previous_used_rod_names(points_csv):
        rod_dir = good_dir / rod_name
        if not rod_dir.is_dir():
            continue
        rec = _load_recording_from_dir(rod_dir, source="previous_used")
        if rec is None:
            continue
        if rec.mean_radius <= float(radius_max):
            out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        raise ValueError("No finite r values to fit")
    lo = float(np.percentile(r, lo_pct))
    hi = float(np.percentile(r, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid r fit range")
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


def _fit_theta_curve_from_recordings(recordings: list[RecordingData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_r = np.concatenate([rec.r for rec in recordings])
    r_centers, _smooth_density, _cdf, fitted_theta_deg = _fit_uniform_costheta_curve(all_r)
    r_fit_grid = np.linspace(float(r_centers[0]), float(r_centers[-1]), 600)
    theta_fit_grid_deg = _interp_monotonic_theta(r_centers, fitted_theta_deg, r_fit_grid)
    return all_r, r_fit_grid, theta_fit_grid_deg


def _bootstrap_theta_curve_recording_level(
    recordings: list[RecordingData],
    n_boot: int = THETA_BOOTSTRAP_RECORDINGS,
    seed: int = THETA_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not recordings:
        raise ValueError("No recordings available for bootstrap.")
    all_r, center_r_grid, center_theta_deg = _fit_theta_curve_from_recordings(recordings)
    rng = np.random.default_rng(int(seed))
    curves: list[np.ndarray] = []
    n_rec = len(recordings)
    for _ in range(int(n_boot)):
        sample_idxs = rng.integers(0, n_rec, size=n_rec)
        sampled = [recordings[int(i)] for i in sample_idxs]
        try:
            _r_boot, r_grid_boot, theta_boot_deg = _fit_theta_curve_from_recordings(sampled)
        except Exception:
            continue
        theta_interp = np.interp(
            center_r_grid,
            r_grid_boot,
            theta_boot_deg,
            left=float(theta_boot_deg[0]),
            right=float(theta_boot_deg[-1]),
        )
        curves.append(theta_interp)
    if not curves:
        raise RuntimeError("Bootstrap failed to produce any theta(r) curves.")
    curves_arr = np.asarray(curves, dtype=np.float64)
    theta_lo = np.percentile(curves_arr, 16.0, axis=0)
    theta_hi = np.percentile(curves_arr, 84.0, axis=0)
    theta_std = np.std(curves_arr, axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros(center_theta_deg.shape)
    return center_r_grid, center_theta_deg, theta_lo, theta_hi, theta_std


def _phi_uniform_score(recordings: list[RecordingData], bins: int = 36) -> tuple[float, dict]:
    if not recordings:
        return float("inf"), {"n_points": 0}
    all_r = np.concatenate([rec.r for rec in recordings])
    all_phi = np.concatenate([rec.phi_deg_axial for rec in recordings])
    mask = all_r >= float(PHI_R_MIN)
    phi = all_phi[mask]
    if phi.size == 0:
        return float("inf"), {"n_points": 0}
    edges = np.linspace(0.0, 180.0, int(bins) + 1)
    counts, _ = np.histogram(phi, bins=edges)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    target = np.full(probs.shape, 1.0 / float(probs.size), dtype=np.float64)
    sse = float(np.sum((probs - target) ** 2))
    max_abs = float(np.max(np.abs(probs - target)))
    return sse, {
        "n_points": int(phi.size),
        "phi_hist_counts": counts.tolist(),
        "phi_hist_probs": probs.tolist(),
        "uniform_prob": float(target[0]),
        "max_abs_bin_dev": max_abs,
        "bins": int(bins),
    }


def _refine_removed_set_by_swaps(
    all_recordings: list[RecordingData],
    removed_idxs: set[int],
    bins: int = 36,
) -> set[int]:
    improved = True
    while improved:
        improved = False
        current_keep = [rec for i, rec in enumerate(all_recordings) if i not in removed_idxs]
        current_score, _ = _phi_uniform_score(current_keep, bins=bins)
        kept_idxs = [i for i in range(len(all_recordings)) if i not in removed_idxs]
        for rem_idx in list(removed_idxs):
            best_swap = None
            best_score = current_score
            for keep_idx in kept_idxs:
                trial_removed = set(removed_idxs)
                trial_removed.remove(rem_idx)
                trial_removed.add(keep_idx)
                trial_keep = [rec for i, rec in enumerate(all_recordings) if i not in trial_removed]
                trial_score, _ = _phi_uniform_score(trial_keep, bins=bins)
                if trial_score < best_score - 1e-15:
                    best_score = trial_score
                    best_swap = keep_idx
            if best_swap is not None:
                removed_idxs.remove(rem_idx)
                removed_idxs.add(best_swap)
                improved = True
                break
    return removed_idxs


def _select_recordings_to_remove_for_uniform_phi(
    all_recordings: list[RecordingData],
    min_remove: int = 5,
    max_remove: int = 10,
    bins: int = 36,
) -> dict:
    if len(all_recordings) <= min_remove:
        raise ValueError("Not enough recordings to optimize removals.")
    best_result: dict | None = None
    n_total = len(all_recordings)
    for n_remove in range(int(min_remove), int(max_remove) + 1):
        removed_idxs: set[int] = set()
        for _ in range(n_remove):
            best_candidate_idx = None
            best_candidate_score = float("inf")
            for idx in range(n_total):
                if idx in removed_idxs:
                    continue
                trial_removed = set(removed_idxs)
                trial_removed.add(idx)
                trial_keep = [rec for i, rec in enumerate(all_recordings) if i not in trial_removed]
                trial_score, _ = _phi_uniform_score(trial_keep, bins=bins)
                if trial_score < best_candidate_score - 1e-15:
                    best_candidate_score = trial_score
                    best_candidate_idx = idx
            if best_candidate_idx is None:
                break
            removed_idxs.add(best_candidate_idx)
        removed_idxs = _refine_removed_set_by_swaps(all_recordings, removed_idxs, bins=bins)
        kept = [rec for i, rec in enumerate(all_recordings) if i not in removed_idxs]
        score, meta = _phi_uniform_score(kept, bins=bins)
        result = {
            "n_remove": int(n_remove),
            "removed_idxs": sorted(int(i) for i in removed_idxs),
            "score_sse": float(score),
            "score_meta": meta,
        }
        if best_result is None or result["score_sse"] < best_result["score_sse"]:
            best_result = result
    if best_result is None:
        raise RuntimeError("Failed to optimize recording removals.")
    return best_result


def _build_case_outputs(case_name: str, out_dir: Path, recordings: list[RecordingData]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    display_name = case_name.replace("optimized for uniform phi", "").replace("  ", " ").strip(" :")
    frame_rows: list[dict] = []
    summary_rows: list[dict] = []
    for rec in recordings:
        for idx, (xv, yv, rv, phiv) in enumerate(zip(rec.x, rec.y, rec.r, rec.phi_deg_axial)):
            frame_rows.append(
                {
                    "rod": rec.rod,
                    "source": rec.source,
                    "frame": idx,
                    "x": float(xv),
                    "y": float(yv),
                    "r": float(rv),
                    "phi_deg_axial": float(phiv),
                }
            )
        summary_rows.append(
            {
                "rod": rec.rod,
                "source": rec.source,
                "frames_used": int(rec.x.size),
                "mean_x": rec.mean_x,
                "mean_y": rec.mean_y,
                "mean_xy_radius": rec.mean_radius,
                "mean_r": float(np.mean(rec.r)),
                "mean_phi_deg_axial": float(np.mean(rec.phi_deg_axial)),
            }
        )

    _write_csv(
        out_dir / "combined_per_frame_points.csv",
        frame_rows,
        ["rod", "source", "frame", "x", "y", "r", "phi_deg_axial"],
    )
    _write_csv(
        out_dir / "combined_per_recording_summary.csv",
        summary_rows,
        ["rod", "source", "frames_used", "mean_x", "mean_y", "mean_xy_radius", "mean_r", "mean_phi_deg_axial"],
    )

    all_x = np.concatenate([rec.x for rec in recordings])
    all_y = np.concatenate([rec.y for rec in recordings])
    all_r = np.concatenate([rec.r for rec in recordings])
    all_phi = np.concatenate([rec.phi_deg_axial for rec in recordings])

    # Smooth XY heatmap
    xy_range = [[-1.0, 1.0], [-1.0, 1.0]]
    counts_xy, xedges, yedges = np.histogram2d(all_x, all_y, bins=220, range=xy_range)
    smooth_xy = _gaussian_smooth_2d(counts_xy.T, sigma_bins=1.4)
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    im = ax.imshow(
        smooth_xy,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="hot",
        interpolation="bilinear",
        aspect="equal",
    )
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Smoothed XY heatmap: {display_name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Smoothed counts")
    fig.tight_layout()
    fig.savefig(out_dir / "xy_heatmap_smooth.png", dpi=220)
    plt.close(fig)

    # Phi distribution with r cut
    phi_mask = all_r >= float(PHI_R_MIN)
    phi_used = all_phi[phi_mask]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins_phi = np.linspace(0.0, 180.0, 73)
    ax.hist(phi_used, bins=bins_phi, density=True, histtype="stepfilled", alpha=0.28, color="#4c78a8")
    ax.hist(phi_used, bins=bins_phi, density=True, histtype="step", lw=1.8, color="#1f4e79")
    ax.set_xlabel("Axial phi (deg)")
    ax.set_ylabel("Density")
    ax.set_title(f"Phi distribution with r >= {PHI_R_MIN:.2f}: {display_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "phi_distribution_r_ge_0p25.png", dpi=220)
    plt.close(fig)

    # Linear r-density with truncated Fourkas curve
    r_grid = np.linspace(0.0, float(THETA_RECON_GLYCEROL50["r_max"]), 600)
    fourkas_density = _fourkas_r_density(r_grid)
    fourkas_cutoff = min(
        float(np.percentile(all_r, 95.0)),
        float(THETA_RECON_GLYCEROL50["r_max"]) - 1e-6,
    )
    fourkas_mask = r_grid <= fourkas_cutoff
    bins_r = np.linspace(0.0, float(THETA_RECON_GLYCEROL50["r_max"]), 70)
    counts_r, edges_r = np.histogram(all_r, bins=bins_r, density=True)
    centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(centers_r, counts_r, color="#4c78a8", lw=1.8, drawstyle="steps-mid", label="Measured r density")
    ax.plot(
        r_grid[fourkas_mask],
        fourkas_density[fourkas_mask],
        color="#f58518",
        lw=2.0,
        label=f"Fourkas theta(r), cut at r={fourkas_cutoff:.3f}",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
    ax.set_title(f"Measured r density vs truncated Fourkas: {display_name}")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "r_density_vs_fourkas_cutoff.png", dpi=220)
    plt.close(fig)

    # Theta(r) fit and comparison
    r_centers, smooth_density, _cdf, fitted_theta_deg = _fit_uniform_costheta_curve(all_r)
    r_fit_grid = np.linspace(float(r_centers[0]), float(r_centers[-1]), 600)
    theta_fit_grid_deg = _interp_monotonic_theta(r_centers, fitted_theta_deg, r_fit_grid)
    fit_density = _theta_curve_density(r_fit_grid, theta_fit_grid_deg)
    fourkas_theta_deg = _theta_fourkas_deg_from_r(r_fit_grid)

    curve_rows = []
    smooth_density_grid = np.interp(r_fit_grid, r_centers, smooth_density, left=smooth_density[0], right=smooth_density[-1])
    for rv, thv, fdv, sdv, thf in zip(r_fit_grid, theta_fit_grid_deg, fit_density, smooth_density_grid, fourkas_theta_deg):
        curve_rows.append(
            {
                "r": float(rv),
                "theta_deg": float(thv),
                "cos_theta": float(np.cos(np.radians(thv))),
                "smoothed_r_density": float(sdv),
                "density_from_theta_curve": float(fdv),
                "theta_deg_fourkas": float(thf),
            }
        )
    _write_csv(
        out_dir / "theta_vs_r_curve.csv",
        curve_rows,
        ["r", "theta_deg", "cos_theta", "smoothed_r_density", "density_from_theta_curve", "theta_deg_fourkas"],
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(r_fit_grid, theta_fit_grid_deg, color="#2ca02c", lw=2.2, label="Theta(r) empirical")
    ax.plot(r_fit_grid, fourkas_theta_deg, color="#f58518", lw=2.0, ls="--", label="Fourkas glycerol theta(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"Theta(r) empirical ({len(recordings)} recordings) vs Fourkas")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_uniform_costheta_vs_fourkas.png", dpi=220)
    plt.close(fig)

    theta_boot_r = None
    theta_boot_center = None
    theta_boot_lo = None
    theta_boot_hi = None
    theta_boot_std = None
    theta_boot_n = 0
    try:
        theta_boot_r, theta_boot_center, theta_boot_lo, theta_boot_hi, theta_boot_std = _bootstrap_theta_curve_recording_level(
            recordings=recordings,
            n_boot=THETA_BOOTSTRAP_RECORDINGS,
            seed=THETA_BOOTSTRAP_SEED,
        )
        theta_boot_n = int(THETA_BOOTSTRAP_RECORDINGS)
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.fill_between(
            theta_boot_r,
            theta_boot_lo,
            theta_boot_hi,
            color="#2ca02c",
            alpha=0.22,
            label="Recording-bootstrap 1 sigma band",
        )
        ax.plot(theta_boot_r, theta_boot_center, color="#1b7f3a", lw=2.4, label="Theta(r) empirical")
        ax.plot(theta_boot_r, _theta_fourkas_deg_from_r(theta_boot_r), color="#f58518", lw=2.0, ls="--", label="Fourkas glycerol theta(r)")
        ax.set_xlabel("r")
        ax.set_ylabel("theta (deg)")
        ax.set_title(f"Theta(r) empirical ({len(recordings)} recordings) vs Fourkas")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "theta_vs_r_with_uncertainty.png", dpi=220)
        plt.close(fig)

        band_rows = []
        for rv, ctv, lov, hiv, sdv in zip(theta_boot_r, theta_boot_center, theta_boot_lo, theta_boot_hi, theta_boot_std):
            band_rows.append(
                {
                    "r": float(rv),
                    "theta_deg_center": float(ctv),
                    "theta_deg_lo_1sigma": float(lov),
                    "theta_deg_hi_1sigma": float(hiv),
                    "theta_deg_std": float(sdv),
                    "theta_deg_fourkas": float(_theta_fourkas_deg_from_r(np.asarray([rv], dtype=np.float64))[0]),
                }
            )
        _write_csv(
            out_dir / "theta_vs_r_uncertainty_band.csv",
            band_rows,
            [
                "r",
                "theta_deg_center",
                "theta_deg_lo_1sigma",
                "theta_deg_hi_1sigma",
                "theta_deg_std",
                "theta_deg_fourkas",
            ],
        )
    except Exception:
        theta_boot_n = 0

    summary = {
        "case_name": case_name,
        "output_dir": str(out_dir),
        "combined_recordings": len(recordings),
        "combined_frames": int(all_r.size),
        "phi_r_min": float(PHI_R_MIN),
        "phi_points_used": int(phi_used.size),
        "r_mean": float(np.mean(all_r)),
        "r_median": float(np.median(all_r)),
        "r_p05": float(np.percentile(all_r, 5.0)),
        "r_p95": float(np.percentile(all_r, 95.0)),
        "r_p95_fourkas_cutoff": fourkas_cutoff,
        "phi_mean_axial_deg_all": float(np.mean(all_phi)),
        "phi_mean_axial_deg_r_ge_min": float(np.mean(phi_used)) if phi_used.size else None,
        "theta_fit_r_range": [float(r_centers[0]), float(r_centers[-1])],
        "fourkas_r_max": float(THETA_RECON_GLYCEROL50["r_max"]),
        "theta_bootstrap_recording_level_replicates": theta_boot_n,
        "theta_bootstrap_seed": int(THETA_BOOTSTRAP_SEED),
        "theta_bootstrap_method": "recording-level bootstrap with replacement; 16th-84th percentile band",
        "theta_bootstrap_midband_mean_width_deg": (
            float(np.mean(theta_boot_hi - theta_boot_lo))
            if theta_boot_lo is not None and theta_boot_hi is not None
            else None
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    dataset_root = Path.cwd() / "glycerol suspended rods 17062026"
    good_dir = dataset_root / "good"
    previous_points_csv = (
        dataset_root
        / "plots"
        / "good_y_negative_phi_90_135_r_phi_distributions"
        / "good_y_negative_phi_90_135_points.csv"
    )
    root_out_dir = dataset_root / "plots" / "good_vs_good_plus_previous_used_center015"
    root_out_dir.mkdir(parents=True, exist_ok=True)

    good_recordings = _load_good_recordings(good_dir)
    if not good_recordings:
        raise SystemExit("No recording data found in the good folder.")
    previous_used_filtered = _load_previous_used_filtered(
        good_dir=good_dir,
        points_csv=previous_points_csv,
        radius_max=PREVIOUS_USED_RADIUS_MAX,
    )

    _build_case_outputs(
        case_name="good only",
        out_dir=root_out_dir / "good_only",
        recordings=good_recordings,
    )
    _build_case_outputs(
        case_name="good + prev-used center<=0.15",
        out_dir=root_out_dir / "good_plus_previous_used_center015",
        recordings=[*good_recordings, *previous_used_filtered],
    )

    combined_recordings = [*good_recordings, *previous_used_filtered]
    opt = _select_recordings_to_remove_for_uniform_phi(
        combined_recordings,
        min_remove=5,
        max_remove=10,
        bins=36,
    )
    removed_set = set(opt["removed_idxs"])
    optimized_recordings = [rec for i, rec in enumerate(combined_recordings) if i not in removed_set]
    _build_case_outputs(
        case_name="combined optimized for uniform phi",
        out_dir=root_out_dir / "combined_phi_uniform_optimized",
        recordings=optimized_recordings,
    )

    root_summary = {
        "dataset_root": str(dataset_root),
        "good_dir": str(good_dir),
        "previous_points_csv": str(previous_points_csv),
        "root_output_dir": str(root_out_dir),
        "good_recordings": len(good_recordings),
        "previous_used_filtered_recordings": len(previous_used_filtered),
        "previous_used_filter_radius": float(PREVIOUS_USED_RADIUS_MAX),
        "previous_used_filtered_rods": [
            {
                "rod": rec.rod,
                "mean_x": rec.mean_x,
                "mean_y": rec.mean_y,
                "mean_xy_radius": rec.mean_radius,
            }
            for rec in previous_used_filtered
        ],
        "phi_uniform_optimization": {
            "source_case": "good_plus_previous_used_center015",
            "n_start_recordings": len(combined_recordings),
            "n_removed": int(opt["n_remove"]),
            "score_sse": float(opt["score_sse"]),
            "score_meta": opt["score_meta"],
            "removed_recordings": [
                {
                    "index": int(i),
                    "rod": combined_recordings[i].rod,
                    "source": combined_recordings[i].source,
                    "mean_x": combined_recordings[i].mean_x,
                    "mean_y": combined_recordings[i].mean_y,
                    "mean_xy_radius": combined_recordings[i].mean_radius,
                }
                for i in opt["removed_idxs"]
            ],
        },
    }
    (root_out_dir / "root_summary.json").write_text(json.dumps(root_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
