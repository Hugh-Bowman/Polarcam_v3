#!/usr/bin/env python
"""
Background characterisation pipeline for polarisation camera recordings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from skimage.filters import sobel

try:
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


EPS = 1e-12


@dataclass
class PCAFit:
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    coefficients: np.ndarray


def fit_pca(x: np.ndarray, n_components: int) -> PCAFit:
    """
    Fit PCA on rows of x (samples x features).
    Uses sklearn when available, otherwise NumPy SVD fallback.
    """
    n_components = min(n_components, x.shape[0], x.shape[1])
    if SKLEARN_AVAILABLE:
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
        coeff = pca.fit_transform(x)
        return PCAFit(
            components=pca.components_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            coefficients=coeff,
        )

    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = n_components
    components = vt[:k]
    coeff = u[:, :k] * s[:k]
    s2 = s * s
    evr = s2[:k] / (np.sum(s2) + EPS)
    return PCAFit(components=components, explained_variance_ratio=evr, coefficients=coeff)


@dataclass
class RecordingStats:
    name: str
    n_frames: int
    mean_img: np.ndarray
    var_img: np.ndarray
    std_img: np.ndarray
    cv_img: np.ndarray
    frame_mean: np.ndarray
    frame_std: np.ndarray
    frame_median: np.ndarray
    frame_mad: np.ndarray
    frame_p10: np.ndarray
    frame_p90: np.ndarray


@dataclass
class ResidualResult:
    method: str
    residual_mean_img: np.ndarray
    residual_var_img: np.ndarray
    residual_std_img: np.ndarray
    frame_mean: np.ndarray
    frame_std: np.ndarray
    frame_median: np.ndarray
    frame_mad: np.ndarray
    avg_residual_variance: float
    median_residual_variance: float
    residual_rms: float
    residual_mad: float
    variance_reduction_fraction: float
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("-", "_")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def percentile_limits(arr: np.ndarray, low: float = 0.0, high: float = 98.0) -> Tuple[float, float]:
    vmin = float(np.nanpercentile(arr, low))
    vmax = float(np.nanpercentile(arr, high))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def robust_frame_stats(
    stack: np.ndarray, sample_pixels: int = 200_000, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, h, w = stack.shape
    total_pixels = h * w
    rng = np.random.default_rng(seed)
    n = min(sample_pixels, total_pixels)
    idx = rng.choice(total_pixels, size=n, replace=False)

    med = np.zeros(t, dtype=np.float64)
    mad = np.zeros(t, dtype=np.float64)
    p10 = np.zeros(t, dtype=np.float64)
    p90 = np.zeros(t, dtype=np.float64)

    for i in range(t):
        frame = stack[i].reshape(-1)[idx].astype(np.float64)
        med[i] = float(np.median(frame))
        mad[i] = float(median_abs_deviation(frame, scale=1.0, nan_policy="omit"))
        p10[i] = float(np.percentile(frame, 10))
        p90[i] = float(np.percentile(frame, 90))
    return med, mad, p10, p90


def compute_recording_stats(name: str, stack: np.ndarray, cv_floor: float = 1.0) -> RecordingStats:
    t = stack.shape[0]
    mean_img = stack.mean(axis=0, dtype=np.float64)
    var_img = stack.var(axis=0, dtype=np.float64)
    std_img = np.sqrt(var_img)

    cv_img = np.full_like(mean_img, np.nan, dtype=np.float64)
    valid = mean_img > cv_floor
    cv_img[valid] = std_img[valid] / mean_img[valid]

    frame_mean = stack.reshape(t, -1).mean(axis=1, dtype=np.float64)
    frame_std = stack.reshape(t, -1).std(axis=1, dtype=np.float64)
    frame_median, frame_mad, frame_p10, frame_p90 = robust_frame_stats(stack)

    return RecordingStats(
        name=name,
        n_frames=t,
        mean_img=mean_img,
        var_img=var_img,
        std_img=std_img,
        cv_img=cv_img,
        frame_mean=frame_mean,
        frame_std=frame_std,
        frame_median=frame_median,
        frame_mad=frame_mad,
        frame_p10=frame_p10,
        frame_p90=frame_p90,
    )


def plot_basic_stats(stats: RecordingStats, out_dir: Path, tag: str) -> None:
    mean_vmin, mean_vmax = percentile_limits(stats.mean_img, 0, 98)
    std_vmin, std_vmax = percentile_limits(stats.std_img, 0, 98)
    cv_vmin, cv_vmax = percentile_limits(np.nan_to_num(stats.cv_img, nan=0.0), 0, 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    im0 = axes[0].imshow(stats.mean_img, cmap="viridis", vmin=mean_vmin, vmax=mean_vmax)
    axes[0].set_title(f"{stats.name}: Mean image (0-98th percentile scale)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(stats.std_img, cmap="magma", vmin=std_vmin, vmax=std_vmax)
    axes[1].set_title(f"{stats.name}: Std image (0-98th percentile scale)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.nan_to_num(stats.cv_img, nan=0.0), cmap="cividis", vmin=cv_vmin, vmax=cv_vmax)
    axes[2].set_title(f"{stats.name}: Coefficient of variation (0-98th percentile scale)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    save_figure(fig, out_dir / f"{tag}_basic_maps.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes2[0].hist(stats.mean_img.ravel(), bins=200, color="tab:blue", alpha=0.8)
    axes2[0].set_title(f"{stats.name}: Histogram of per-pixel mean")
    axes2[0].set_xlabel("Mean intensity")
    axes2[0].set_ylabel("Count")

    axes2[1].hist(stats.var_img.ravel(), bins=200, color="tab:orange", alpha=0.8)
    axes2[1].set_title(f"{stats.name}: Histogram of per-pixel variance")
    axes2[1].set_xlabel("Variance")
    axes2[1].set_ylabel("Count")

    save_figure(fig2, out_dir / f"{tag}_pixel_histograms.png")

    x = np.arange(stats.n_frames)
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 7.5), constrained_layout=True)
    axes3[0].plot(x, stats.frame_mean, label="Global mean", color="tab:blue")
    axes3[0].set_title(f"{stats.name}: Frame-wise global mean")
    axes3[0].set_xlabel("Frame")
    axes3[0].set_ylabel("Intensity")
    axes3[0].grid(alpha=0.25)

    axes3[1].plot(x, stats.frame_std, label="Global std", color="tab:red")
    axes3[1].plot(x, stats.frame_median, label="Robust median", color="tab:green")
    axes3[1].plot(x, stats.frame_mad, label="Robust MAD", color="tab:purple")
    axes3[1].fill_between(x, stats.frame_p10, stats.frame_p90, alpha=0.2, color="gray", label="p10-p90")
    axes3[1].set_title(f"{stats.name}: Frame-wise std/median/MAD")
    axes3[1].set_xlabel("Frame")
    axes3[1].set_ylabel("Intensity")
    axes3[1].grid(alpha=0.25)
    axes3[1].legend()

    save_figure(fig3, out_dir / f"{tag}_global_timeseries.png")


def compute_per_pixel_median(stack: np.ndarray, fallback_stride: int = 3) -> np.ndarray:
    try:
        return np.median(np.asarray(stack), axis=0).astype(np.float64)
    except MemoryError:
        subset = np.asarray(stack[::fallback_stride])
        return np.median(subset, axis=0).astype(np.float64)


def power_spectrum_image(img: np.ndarray) -> np.ndarray:
    centered = img - np.mean(img)
    f = np.fft.fft2(centered)
    pwr = np.abs(np.fft.fftshift(f)) ** 2
    return np.log1p(pwr)


def plot_background_candidates(
    b_no: np.ndarray, b_cov: np.ndarray, b_cov_minus_no: np.ndarray, out_dir: Path, tag: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    vmin0, vmax0 = percentile_limits(b_no, 0, 98)
    vmin1, vmax1 = percentile_limits(b_cov, 0, 98)
    vmax2 = float(np.percentile(np.abs(b_cov_minus_no), 98))
    vmax2 = max(vmax2, 1e-6)

    im0 = axes[0].imshow(b_no, cmap="viridis", vmin=vmin0, vmax=vmax0)
    axes[0].set_title("B_no = mean(no_coverslip)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b_cov, cmap="viridis", vmin=vmin1, vmax=vmax1)
    axes[1].set_title("B_cov = mean(coverslip_static)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(b_cov_minus_no, cmap="coolwarm", vmin=-vmax2, vmax=vmax2)
    axes[2].set_title("B_cov_minus_no")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    save_figure(fig, out_dir / f"{tag}_background_candidates.png")

    h, w = b_cov.shape
    cy, cx = h // 2, w // 2
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes2[0].plot(b_no[cy, :], label="B_no")
    axes2[0].plot(b_cov[cy, :], label="B_cov")
    axes2[0].plot(b_cov_minus_no[cy, :], label="B_cov_minus_no")
    axes2[0].set_title(f"Central row profile (y={cy})")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("Intensity")
    axes2[0].grid(alpha=0.25)
    axes2[0].legend()

    axes2[1].plot(b_no[:, cx], label="B_no")
    axes2[1].plot(b_cov[:, cx], label="B_cov")
    axes2[1].plot(b_cov_minus_no[:, cx], label="B_cov_minus_no")
    axes2[1].set_title(f"Central column profile (x={cx})")
    axes2[1].set_xlabel("y")
    axes2[1].set_ylabel("Intensity")
    axes2[1].grid(alpha=0.25)
    axes2[1].legend()

    save_figure(fig2, out_dir / f"{tag}_background_line_profiles.png")

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes3[0].hist(b_cov_minus_no.ravel(), bins=200, color="tab:blue", alpha=0.8)
    axes3[0].set_title("Histogram of B_cov_minus_no")
    axes3[0].set_xlabel("Intensity difference")
    axes3[0].set_ylabel("Count")

    pwr = power_spectrum_image(b_cov_minus_no)
    im3 = axes3[1].imshow(pwr, cmap="magma")
    axes3[1].set_title("Log power spectrum of B_cov_minus_no")
    axes3[1].set_xlabel("kx")
    axes3[1].set_ylabel("ky")
    fig3.colorbar(im3, ax=axes3[1], fraction=0.046, pad=0.04)

    save_figure(fig3, out_dir / f"{tag}_background_hist_power.png")


def compute_significance_z_map(
    mu_cov: np.ndarray, var_cov: np.ndarray, n_cov: int, mu_no: np.ndarray, var_no: np.ndarray, n_no: int
) -> np.ndarray:
    denom = np.sqrt(var_cov / max(n_cov, 1) + var_no / max(n_no, 1) + EPS)
    return (mu_cov - mu_no) / denom


def plot_z_map(z_map: np.ndarray, out_dir: Path, tag: str) -> None:
    vmax = float(np.percentile(np.abs(z_map), 98))
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    im = axes[0].imshow(z_map, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Z-map for coverslip shift significance")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(z_map.ravel(), bins=250, color="tab:gray", alpha=0.9)
    axes[1].set_title("Histogram of Z values")
    axes[1].set_xlabel("Z")
    axes[1].set_ylabel("Count")

    save_figure(fig, out_dir / f"{tag}_zmap_hist.png")


def fit_alpha_beta_ols(frame: np.ndarray, bg: np.ndarray) -> Tuple[float, float]:
    x = bg.ravel().astype(np.float64)
    y = frame.ravel().astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    denom = float(np.dot(xv, xv)) + EPS
    beta = float(np.dot(xv, yv) / denom)
    alpha = float(ym - beta * xm)
    return alpha, beta


def compute_residuals(
    stack: np.ndarray,
    raw_stats: RecordingStats,
    background: np.ndarray,
    method_name: str,
    fit_per_frame_gain_offset: bool,
    example_frame_indices: Sequence[int] = (0, 150, 299),
) -> Tuple[ResidualResult, Dict[int, np.ndarray]]:
    t, h, w = stack.shape
    if not fit_per_frame_gain_offset:
        residual_mean_img = raw_stats.mean_img - background
        residual_var_img = raw_stats.var_img.copy()
        residual_std_img = np.sqrt(residual_var_img)

        frame_mean = np.zeros(t, dtype=np.float64)
        frame_std = np.zeros(t, dtype=np.float64)
        frame_median = np.zeros(t, dtype=np.float64)
        frame_mad = np.zeros(t, dtype=np.float64)
        examples: Dict[int, np.ndarray] = {}
        for i in range(t):
            resid = stack[i].astype(np.float64) - background
            frame_mean[i] = float(np.mean(resid))
            frame_std[i] = float(np.std(resid))
            flat = resid.ravel()
            frame_median[i] = float(np.median(flat))
            frame_mad[i] = float(median_abs_deviation(flat, scale=1.0, nan_policy="omit"))
            if i in example_frame_indices:
                examples[i] = resid
        alpha = None
        beta = None
    else:
        mean_acc = np.zeros((h, w), dtype=np.float64)
        m2_acc = np.zeros((h, w), dtype=np.float64)
        frame_mean = np.zeros(t, dtype=np.float64)
        frame_std = np.zeros(t, dtype=np.float64)
        frame_median = np.zeros(t, dtype=np.float64)
        frame_mad = np.zeros(t, dtype=np.float64)
        alpha = np.zeros(t, dtype=np.float64)
        beta = np.zeros(t, dtype=np.float64)
        examples: Dict[int, np.ndarray] = {}
        for i in range(t):
            frame = stack[i].astype(np.float64)
            a_i, b_i = fit_alpha_beta_ols(frame, background)
            alpha[i] = a_i
            beta[i] = b_i
            resid = frame - (a_i + b_i * background)
            delta = resid - mean_acc
            mean_acc += delta / (i + 1)
            delta2 = resid - mean_acc
            m2_acc += delta * delta2
            frame_mean[i] = float(np.mean(resid))
            frame_std[i] = float(np.std(resid))
            flat = resid.ravel()
            frame_median[i] = float(np.median(flat))
            frame_mad[i] = float(median_abs_deviation(flat, scale=1.0, nan_policy="omit"))
            if i in example_frame_indices:
                examples[i] = resid
        residual_mean_img = mean_acc
        residual_var_img = m2_acc / max(t, 1)
        residual_std_img = np.sqrt(residual_var_img)

    avg_residual_variance = float(np.mean(residual_var_img))
    median_residual_variance = float(np.median(residual_var_img))
    residual_rms = float(np.sqrt(np.mean(residual_mean_img**2 + residual_var_img)))
    residual_mad = float(np.median(frame_mad))
    raw_mean_var = float(np.mean(raw_stats.var_img))
    variance_reduction_fraction = (raw_mean_var - avg_residual_variance) / (raw_mean_var + EPS)

    result = ResidualResult(
        method=method_name,
        residual_mean_img=residual_mean_img,
        residual_var_img=residual_var_img,
        residual_std_img=residual_std_img,
        frame_mean=frame_mean,
        frame_std=frame_std,
        frame_median=frame_median,
        frame_mad=frame_mad,
        avg_residual_variance=avg_residual_variance,
        median_residual_variance=median_residual_variance,
        residual_rms=residual_rms,
        residual_mad=residual_mad,
        variance_reduction_fraction=float(variance_reduction_fraction),
        alpha=alpha,
        beta=beta,
    )
    return result, examples


def plot_residual_result(recording_name: str, res: ResidualResult, examples: Mapping[int, np.ndarray], out_dir: Path, tag: str) -> None:
    vmin_m, vmax_m = percentile_limits(res.residual_mean_img, 1, 99)
    vmin_s, vmax_s = percentile_limits(res.residual_std_img, 0, 98)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    im0 = axes[0].imshow(res.residual_mean_img, cmap="coolwarm", vmin=vmin_m, vmax=vmax_m)
    axes[0].set_title(f"{recording_name} [{res.method}] residual mean")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(res.residual_var_img, cmap="magma", vmin=0, vmax=float(np.percentile(res.residual_var_img, 98)))
    axes[1].set_title(f"{recording_name} [{res.method}] residual variance")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(res.residual_std_img, cmap="magma", vmin=vmin_s, vmax=vmax_s)
    axes[2].set_title(f"{recording_name} [{res.method}] residual std")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    save_figure(fig, out_dir / f"{tag}_residual_maps.png")

    x = np.arange(len(res.frame_mean))
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7.2), constrained_layout=True)
    axes2[0].plot(x, res.frame_mean, color="tab:blue")
    axes2[0].set_title(f"{recording_name} [{res.method}] residual global mean vs frame")
    axes2[0].set_xlabel("Frame")
    axes2[0].set_ylabel("Mean")
    axes2[0].grid(alpha=0.25)
    axes2[1].plot(x, res.frame_std, color="tab:red", label="Residual std")
    axes2[1].plot(x, res.frame_mad, color="tab:purple", label="Residual MAD")
    axes2[1].plot(x, res.frame_median, color="tab:green", label="Residual median")
    axes2[1].set_title(f"{recording_name} [{res.method}] residual std/median/MAD vs frame")
    axes2[1].set_xlabel("Frame")
    axes2[1].set_ylabel("Intensity")
    axes2[1].grid(alpha=0.25)
    axes2[1].legend()
    save_figure(fig2, out_dir / f"{tag}_residual_timeseries.png")

    if examples:
        idxs = sorted(examples.keys())
        fig3, axes3 = plt.subplots(1, len(idxs), figsize=(5 * len(idxs), 4.8), constrained_layout=True)
        axes = [axes3] if len(idxs) == 1 else axes3
        for ax, idx in zip(axes, idxs):
            img = examples[idx]
            vmax = float(np.percentile(np.abs(img), 98))
            vmax = max(vmax, 1e-6)
            im = ax.imshow(img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Residual frame {idx}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_figure(fig3, out_dir / f"{tag}_residual_examples.png")


def quad_model(i: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * i + c * i * i


def binned_quantiles(
    x: np.ndarray, y: np.ndarray, n_bins: int = 80, q_list: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> Dict[str, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("Insufficient dynamic range for binning")
    centers = []
    counts = []
    qs = {q: [] for q in q_list}
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        m = (x >= lo) & (x < hi)
        if np.count_nonzero(m) < 20:
            continue
        yy = y[m]
        centers.append(0.5 * (lo + hi))
        counts.append(np.count_nonzero(m))
        for q in q_list:
            qs[q].append(float(np.quantile(yy, q)))
    out = {"centers": np.asarray(centers, dtype=np.float64), "counts": np.asarray(counts, dtype=np.int64)}
    for q in q_list:
        out[f"q{int(round(q*100))}"] = np.asarray(qs[q], dtype=np.float64)
    return out


def variance_vs_intensity_analysis(
    intensity_img: np.ndarray,
    var_img: np.ndarray,
    out_dir: Path,
    tag: str,
    title_prefix: str,
    scatter_points: int = 200_000,
) -> Dict[str, float]:
    x = intensity_img.ravel().astype(np.float64)
    y = var_img.ravel().astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    rng = np.random.default_rng(0)
    if x.size > scatter_points:
        idx = rng.choice(x.size, size=scatter_points, replace=False)
        xs = x[idx]
        ys = y[idx]
    else:
        xs, ys = x, y

    try:
        bq = binned_quantiles(x, y)
        centers = bq["centers"]
        q10 = bq["q10"]
        q25 = bq["q25"]
        q50 = bq["q50"]
        q75 = bq["q75"]
        q90 = bq["q90"]
        popt, pcov = curve_fit(quad_model, centers, q50, maxfev=10000)
        fit = quad_model(centers, *popt)
        ss_res = float(np.sum((q50 - fit) ** 2))
        ss_tot = float(np.sum((q50 - np.mean(q50)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + EPS)
        a, b, c = map(float, popt)
        ua, ub, uc = np.sqrt(np.clip(np.diag(pcov), 0, np.inf)).tolist()
        residual_cloud = y - quad_model(x, a, b, c)
        residual_cloud_var = float(np.var(residual_cloud))
    except ValueError:
        # Low-dynamic-range intensity (e.g., after near-perfect background subtraction):
        # fall back to constant model.
        centers = np.asarray([float(np.mean(x))], dtype=np.float64)
        medv = float(np.median(y))
        q10 = np.asarray([float(np.quantile(y, 0.1))], dtype=np.float64)
        q25 = np.asarray([float(np.quantile(y, 0.25))], dtype=np.float64)
        q50 = np.asarray([medv], dtype=np.float64)
        q75 = np.asarray([float(np.quantile(y, 0.75))], dtype=np.float64)
        q90 = np.asarray([float(np.quantile(y, 0.9))], dtype=np.float64)
        fit = np.asarray([medv], dtype=np.float64)
        a, b, c = medv, 0.0, 0.0
        ua, ub, uc = 0.0, 0.0, 0.0
        r2 = 0.0
        residual_cloud_var = float(np.var(y - medv))

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.scatter(xs, ys, s=2, alpha=0.05, color="tab:blue", label="Pixels")
    ax.fill_between(centers, q25, q75, alpha=0.25, color="tab:orange", label="IQR")
    ax.fill_between(centers, q10, q90, alpha=0.15, color="tab:green", label="10-90%")
    ax.plot(centers, q50, color="tab:orange", linewidth=2, label="Binned median")
    ax.plot(centers, fit, color="tab:red", linewidth=2, label="Fit: a+bI+cI^2")
    ax.set_title(f"{title_prefix}: Var vs I")
    ax.set_xlabel("Per-pixel mean intensity I")
    ax.set_ylabel("Per-pixel variance")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir / f"{tag}_var_vs_intensity.png")

    width_p90_p10 = q90 - q10
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax2.plot(centers, width_p90_p10, color="tab:purple")
    ax2.set_title(f"{title_prefix}: Cloud width (q90-q10) vs I")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Variance spread")
    ax2.grid(alpha=0.25)
    save_figure(fig2, out_dir / f"{tag}_var_cloud_width.png")

    return {
        "a": a,
        "b": b,
        "c": c,
        "ua": float(ua),
        "ub": float(ub),
        "uc": float(uc),
        "r2_binned_median": r2,
        "residual_cloud_variance": residual_cloud_var,
        "median_cloud_width_q90_q10": float(np.median(width_p90_p10)),
    }


def split_polarisation_channels(stack: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "90deg": stack[:, 0::2, 0::2],
        "45deg": stack[:, 0::2, 1::2],
        "135deg": stack[:, 1::2, 0::2],
        "0deg": stack[:, 1::2, 1::2],
    }


def plot_compare_static_tapped(
    static_stats: RecordingStats,
    tapped_stats: RecordingStats,
    b_cov_minus_no: np.ndarray,
    grad_mag: np.ndarray,
    out_dir: Path,
    tag: str,
) -> Dict[str, float]:
    mean_diff = tapped_stats.mean_img - static_stats.mean_img
    var_ratio = tapped_stats.var_img / (static_stats.var_img + EPS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(tapped_stats.mean_img, cmap="viridis", vmin=0, vmax=float(np.percentile(tapped_stats.mean_img, 98)))
    axes[0, 0].set_title("Tapped mean image (0-98th percentile)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im1 = axes[0, 1].imshow(tapped_stats.var_img, cmap="magma", vmin=0, vmax=float(np.percentile(tapped_stats.var_img, 98)))
    axes[0, 1].set_title("Tapped variance image (0-98th percentile)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    vmax_md = float(np.percentile(np.abs(mean_diff), 98))
    im2 = axes[1, 0].imshow(mean_diff, cmap="coolwarm", vmin=-vmax_md, vmax=vmax_md)
    axes[1, 0].set_title("Mean difference: tapped - static")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    vmax_vr = float(np.percentile(var_ratio, 98))
    im3 = axes[1, 1].imshow(var_ratio, cmap="plasma", vmin=0, vmax=vmax_vr)
    axes[1, 1].set_title("Variance ratio: tapped/static")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    save_figure(fig, out_dir / f"{tag}_tapped_static_maps.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes2[0].hist(var_ratio.ravel(), bins=250, color="tab:orange", alpha=0.85)
    axes2[0].set_title("Histogram of variance ratio (tapped/static)")
    axes2[0].set_xlabel("Variance ratio")
    axes2[0].set_ylabel("Count")
    x = np.arange(static_stats.n_frames)
    axes2[1].plot(x, static_stats.frame_mean, label="Static mean", color="tab:blue")
    axes2[1].plot(x, tapped_stats.frame_mean, label="Tapped mean", color="tab:cyan")
    axes2[1].plot(x, static_stats.frame_std, label="Static std", color="tab:red")
    axes2[1].plot(x, tapped_stats.frame_std, label="Tapped std", color="tab:purple")
    axes2[1].set_title("Frame-wise global mean/std: static vs tapped")
    axes2[1].set_xlabel("Frame")
    axes2[1].set_ylabel("Intensity")
    axes2[1].grid(alpha=0.25)
    axes2[1].legend()
    save_figure(fig2, out_dir / f"{tag}_tapped_static_hist_timeseries.png")

    tapped_resid_var = tapped_stats.var_img
    corr_b = float(np.corrcoef(b_cov_minus_no.ravel(), tapped_resid_var.ravel())[0, 1])
    corr_g = float(np.corrcoef(grad_mag.ravel(), tapped_resid_var.ravel())[0, 1])

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    im_g = axes3[0].imshow(grad_mag, cmap="magma", vmin=0, vmax=float(np.percentile(grad_mag, 98)))
    axes3[0].set_title("Gradient magnitude of B_cov")
    axes3[0].set_xlabel("x")
    axes3[0].set_ylabel("y")
    fig3.colorbar(im_g, ax=axes3[0], fraction=0.046, pad=0.04)
    idx = np.random.default_rng(0).choice(grad_mag.size, size=min(200_000, grad_mag.size), replace=False)
    gx = grad_mag.ravel()[idx]
    gy = tapped_resid_var.ravel()[idx]
    axes3[1].scatter(gx, gy, s=2, alpha=0.05, color="tab:blue")
    bq = binned_quantiles(gx, gy, n_bins=80)
    axes3[1].plot(bq["centers"], bq["q50"], color="tab:red", linewidth=2, label="Median")
    axes3[1].fill_between(bq["centers"], bq["q10"], bq["q90"], color="tab:red", alpha=0.2, label="10-90%")
    axes3[1].set_title("Tapped residual variance vs |grad(B_cov)|")
    axes3[1].set_xlabel("Gradient magnitude")
    axes3[1].set_ylabel("Tapped variance")
    axes3[1].grid(alpha=0.25)
    axes3[1].legend()
    save_figure(fig3, out_dir / f"{tag}_gradient_vs_tapped_variance.png")

    med_ratio = float(np.median(var_ratio))
    p90_ratio = float(np.percentile(var_ratio, 90))
    return {
        "median_variance_ratio_tapped_over_static": med_ratio,
        "p90_variance_ratio_tapped_over_static": p90_ratio,
        "frac_ratio_gt_2": float(np.mean(var_ratio > 2.0)),
        "frac_ratio_gt_5": float(np.mean(var_ratio > 5.0)),
        "frac_ratio_gt_10": float(np.mean(var_ratio > 10.0)),
        "corr_tapped_var_with_B_cov_minus_no": corr_b,
        "corr_tapped_var_with_grad_mag": corr_g,
    }


def run_pca_analysis(
    stacks: Mapping[str, np.ndarray],
    out_dir: Path,
    tag: str,
    downsample: int = 8,
    n_components: int = 10,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for rec_name, stack in stacks.items():
        ds = stack[:, ::downsample, ::downsample].astype(np.float32)
        t, h, w = ds.shape
        mean_img = ds.mean(axis=0, dtype=np.float64)
        x = ds.reshape(t, -1).astype(np.float32)
        x -= mean_img.ravel()[None, :].astype(np.float32)
        k = min(n_components, t)
        pca_fit = fit_pca(x, k)
        coeff = pca_fit.coefficients
        modes = pca_fit.components.reshape(k, h, w)
        evr = pca_fit.explained_variance_ratio

        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        ax.plot(np.arange(1, len(evr) + 1), evr, marker="o")
        ax.set_title(f"{rec_name}: PCA explained variance ratio")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained variance ratio")
        ax.grid(alpha=0.25)
        save_figure(fig, out_dir / f"{tag}_{sanitize(rec_name)}_pca_explained_variance.png")

        n_show = min(5, k)
        fig2, axes2 = plt.subplots(1, n_show, figsize=(3.8 * n_show, 3.8), constrained_layout=True)
        axes = [axes2] if n_show == 1 else axes2
        for i, axm in enumerate(axes):
            m = modes[i]
            vmax = float(np.percentile(np.abs(m), 98))
            vmax = max(vmax, 1e-9)
            im = axm.imshow(m, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axm.set_title(f"Mode {i+1}")
            axm.set_xlabel("x")
            axm.set_ylabel("y")
            fig2.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
        save_figure(fig2, out_dir / f"{tag}_{sanitize(rec_name)}_pca_modes.png")

        fig3, axes3 = plt.subplots(n_show, 1, figsize=(10, 2.0 * n_show), constrained_layout=True)
        axes = [axes3] if n_show == 1 else axes3
        for i, axc in enumerate(axes):
            axc.plot(coeff[:, i], color="tab:blue")
            axc.set_title(f"{rec_name}: PC{i+1} coefficient vs frame")
            axc.set_xlabel("Frame")
            axc.set_ylabel("Coeff")
            axc.grid(alpha=0.25)
        save_figure(fig3, out_dir / f"{tag}_{sanitize(rec_name)}_pca_coefficients.png")

        summary[rec_name] = {
            "explained_variance_ratio_pc1": float(evr[0]),
            "explained_variance_ratio_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
            "explained_variance_ratio_pc5": float(evr[4]) if len(evr) > 4 else np.nan,
        }
    return summary


def pca_subtraction_eval(
    reference_stack: np.ndarray,
    target_stacks: Mapping[str, np.ndarray],
    out_dir: Path,
    tag: str,
    downsample: int = 8,
    ks: Sequence[int] = (0, 1, 2, 5, 10),
) -> Dict[str, Dict[str, float]]:
    ref = reference_stack[:, ::downsample, ::downsample].astype(np.float32)
    t_ref = ref.shape[0]
    ref_mean = ref.mean(axis=0, dtype=np.float64)
    x_ref = ref.reshape(t_ref, -1).astype(np.float32)
    x_ref -= ref_mean.ravel()[None, :].astype(np.float32)
    max_k = min(max(ks), t_ref)
    pca_fit = fit_pca(x_ref, max_k)
    all_components = pca_fit.components

    out: Dict[str, Dict[str, float]] = {}
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for name, stack in target_stacks.items():
        ds = stack[:, ::downsample, ::downsample].astype(np.float32)
        t = ds.shape[0]
        x = ds.reshape(t, -1).astype(np.float32)
        x -= ref_mean.ravel()[None, :].astype(np.float32)
        vals = []
        for k in ks:
            kk = min(k, max_k)
            if kk == 0:
                resid = x
            else:
                comps = all_components[:kk]
                coef = x @ comps.T
                recon = coef @ comps
                resid = x - recon
            vals.append(float(np.var(resid)))
        ax.plot(list(ks), vals, marker="o", label=name)
        out[name] = {f"residual_variance_k{k}": float(v) for k, v in zip(ks, vals)}
    ax.set_title("Residual variance vs PCA modes removed")
    ax.set_xlabel("Number of PCA modes removed")
    ax.set_ylabel("Residual variance (downsampled domain)")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir / f"{tag}_pca_subtraction_eval.png")
    return out


def summarize_background_shift(diff_map: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(diff_map)),
        "median": float(np.median(diff_map)),
        "std": float(np.std(diff_map)),
        "mad": float(median_abs_deviation(diff_map.ravel(), scale=1.0, nan_policy="omit")),
    }


def choose_best_background_method(metrics: Mapping[str, ResidualResult]) -> str:
    best_name = None
    best_value = np.inf
    for method, res in metrics.items():
        if res.avg_residual_variance < best_value:
            best_value = res.avg_residual_variance
            best_name = method
    return best_name if best_name is not None else "unknown"


def run_core_analysis(stacks: Mapping[str, np.ndarray], out_dir: Path, prefix: str, run_pca: bool) -> Dict[str, object]:
    ensure_dir(out_dir)
    stats = {name: compute_recording_stats(name, stack) for name, stack in stacks.items()}
    for name, st in stats.items():
        plot_basic_stats(st, out_dir, f"{prefix}_{sanitize(name)}")

    st_no = stats["no_coverslip"]
    st_cov = stats["coverslip_static"]
    st_tap = stats["coverslip_tapped"]
    b_no = st_no.mean_img
    b_cov = st_cov.mean_img
    b_cov_robust = compute_per_pixel_median(stacks["coverslip_static"])
    b_cov_minus_no = b_cov - b_no
    plot_background_candidates(b_no, b_cov, b_cov_minus_no, out_dir, f"{prefix}_backgrounds")

    z_map = compute_significance_z_map(
        mu_cov=b_cov,
        var_cov=st_cov.var_img,
        n_cov=st_cov.n_frames,
        mu_no=b_no,
        var_no=st_no.var_img,
        n_no=st_no.n_frames,
    )
    plot_z_map(z_map, out_dir, f"{prefix}_backgrounds")

    residual_methods_cov: Dict[str, ResidualResult] = {}
    for method_name, bg in [
        ("subtract_B_no", b_no),
        ("subtract_B_cov", b_cov),
        ("subtract_B_cov_robust", b_cov_robust),
        ("fit_alpha_beta_on_B_no", b_no),
        ("fit_alpha_beta_on_B_cov", b_cov),
    ]:
        fit_ab = method_name.startswith("fit_alpha_beta")
        res, examples = compute_residuals(
            stack=stacks["coverslip_static"],
            raw_stats=st_cov,
            background=bg,
            method_name=method_name,
            fit_per_frame_gain_offset=fit_ab,
            example_frame_indices=(0, st_cov.n_frames // 2, st_cov.n_frames - 1),
        )
        residual_methods_cov[method_name] = res
        plot_residual_result(
            "coverslip_static",
            res,
            examples,
            out_dir,
            f"{prefix}_coverslip_static_{sanitize(method_name)}",
        )

    tapped_res_B_cov, _ = compute_residuals(
        stack=stacks["coverslip_tapped"],
        raw_stats=st_tap,
        background=b_cov,
        method_name="subtract_B_cov",
        fit_per_frame_gain_offset=False,
    )
    static_res_B_cov = residual_methods_cov["subtract_B_cov"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    vmax = float(
        np.percentile(
            np.concatenate([static_res_B_cov.residual_var_img.ravel(), tapped_res_B_cov.residual_var_img.ravel()]),
            98,
        )
    )
    im0 = axes[0].imshow(static_res_B_cov.residual_var_img, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title("Static residual variance (subtract B_cov)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(tapped_res_B_cov.residual_var_img, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title("Tapped residual variance (subtract B_cov)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    save_figure(fig, out_dir / f"{prefix}_static_vs_tapped_residual_variance_Bcov.png")

    var_intensity_summary: Dict[str, Dict[str, float]] = {}
    for rec_name in ["no_coverslip", "coverslip_static", "coverslip_tapped"]:
        raw = stats[rec_name]
        var_intensity_summary[f"{rec_name}:raw"] = variance_vs_intensity_analysis(
            intensity_img=raw.mean_img,
            var_img=raw.var_img,
            out_dir=out_dir,
            tag=f"{prefix}_{sanitize(rec_name)}_raw",
            title_prefix=f"{rec_name} raw",
        )

    for m_name, res in residual_methods_cov.items():
        var_intensity_summary[f"coverslip_static:{m_name}"] = variance_vs_intensity_analysis(
            intensity_img=res.residual_mean_img,
            var_img=res.residual_var_img,
            out_dir=out_dir,
            tag=f"{prefix}_coverslip_static_{sanitize(m_name)}",
            title_prefix=f"coverslip_static [{m_name}]",
        )

    grad_mag = sobel(b_cov)
    compare_metrics = plot_compare_static_tapped(
        static_stats=st_cov,
        tapped_stats=st_tap,
        b_cov_minus_no=b_cov_minus_no,
        grad_mag=grad_mag,
        out_dir=out_dir,
        tag=f"{prefix}_compare",
    )

    pca_summary: Dict[str, object] = {}
    if run_pca:
        pca_summary["per_recording"] = run_pca_analysis(stacks, out_dir, f"{prefix}_pca")
        pca_summary["subtraction_eval"] = pca_subtraction_eval(
            reference_stack=stacks["coverslip_static"],
            target_stacks={
                "coverslip_static": stacks["coverslip_static"],
                "coverslip_tapped": stacks["coverslip_tapped"],
                "no_coverslip": stacks["no_coverslip"],
            },
            out_dir=out_dir,
            tag=f"{prefix}_pca",
        )

    b_shift_metrics = summarize_background_shift(b_cov_minus_no)
    frac_significant = float(np.mean(np.abs(z_map) > 3.0))
    best_method = choose_best_background_method(residual_methods_cov)

    return {
        "background_shift_metrics": b_shift_metrics,
        "fraction_significant_shift_absZ_gt_3": frac_significant,
        "best_static_background_method_for_coverslip_static": best_method,
        "best_method_variance_reduction_fraction": float(residual_methods_cov[best_method].variance_reduction_fraction),
        "residual_methods_coverslip_static": {
            k: {
                "avg_residual_variance": float(v.avg_residual_variance),
                "median_residual_variance": float(v.median_residual_variance),
                "residual_rms": float(v.residual_rms),
                "residual_mad": float(v.residual_mad),
                "variance_reduction_fraction": float(v.variance_reduction_fraction),
            }
            for k, v in residual_methods_cov.items()
        },
        "variance_vs_intensity_fits": var_intensity_summary,
        "tapped_vs_static_metrics": compare_metrics,
        "pca_summary": pca_summary,
    }


def load_stack(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def write_text_summary(summary: Mapping[str, object], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("Background characterisation summary")
    lines.append("")
    best = summary.get("best_static_background_method_for_coverslip_static", "unknown")
    frac_sig = summary.get("fraction_significant_shift_absZ_gt_3", np.nan)
    tap = summary.get("tapped_vs_static_metrics", {})
    vv = summary.get("variance_vs_intensity_fits", {})
    lines.append(f"Best static background candidate/method: {best}")
    lines.append(f"Coverslip static pattern significance (|Z|>3 fraction): {frac_sig:.4f}")
    lines.append(
        "Tapped variance increase (median ratio, p90 ratio): "
        f"{tap.get('median_variance_ratio_tapped_over_static', np.nan):.4f}, "
        f"{tap.get('p90_variance_ratio_tapped_over_static', np.nan):.4f}"
    )
    lines.append(
        "Fraction pixels variance increase >2x/>5x/>10x: "
        f"{tap.get('frac_ratio_gt_2', np.nan):.4f}/"
        f"{tap.get('frac_ratio_gt_5', np.nan):.4f}/"
        f"{tap.get('frac_ratio_gt_10', np.nan):.4f}"
    )
    lines.append(
        "Correlation tapped variance with static gradient magnitude: "
        f"{tap.get('corr_tapped_var_with_grad_mag', np.nan):.4f}"
    )
    static_fit = vv.get("coverslip_static:raw", {})
    lines.append(
        "Variance-vs-intensity (coverslip_static raw) binned-fit R^2: "
        f"{static_fit.get('r2_binned_median', np.nan):.4f}"
    )
    lines.append("")
    lines.append("Recommended subtraction strategy:")
    lines.append(
        "1) Use B_cov (or B_cov_robust) as static baseline. "
        "2) Fit per-frame alpha+beta relative to B_cov and subtract alpha+beta*B_cov. "
        "3) Optionally remove first PCA drift modes if needed."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polarisation camera background characterisation")
    parser.add_argument("--no-coverslip", type=Path, required=True)
    parser.add_argument("--coverslip-static", type=Path, required=True)
    parser.add_argument("--coverslip-tapped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-per-channel", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    stacks = {
        "no_coverslip": load_stack(args.no_coverslip),
        "coverslip_static": load_stack(args.coverslip_static),
        "coverslip_tapped": load_stack(args.coverslip_tapped),
    }

    summary_full = run_core_analysis(stacks=stacks, out_dir=args.output_dir, prefix="full", run_pca=not args.skip_pca)
    all_summary = {"full": summary_full}

    if not args.skip_per_channel:
        per_channel = {k: split_polarisation_channels(v) for k, v in stacks.items()}
        for ch in ["90deg", "45deg", "135deg", "0deg"]:
            ch_stacks = {
                "no_coverslip": per_channel["no_coverslip"][ch],
                "coverslip_static": per_channel["coverslip_static"][ch],
                "coverslip_tapped": per_channel["coverslip_tapped"][ch],
            }
            all_summary[f"channel_{ch}"] = run_core_analysis(
                stacks=ch_stacks, out_dir=args.output_dir, prefix=f"channel_{ch}", run_pca=False
            )

    json_path = args.output_dir / "analysis_summary.json"
    json_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    txt_path = args.output_dir / "final_report.txt"
    write_text_summary(summary_full, txt_path)
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved final report: {txt_path}")


if __name__ == "__main__":
    main()
