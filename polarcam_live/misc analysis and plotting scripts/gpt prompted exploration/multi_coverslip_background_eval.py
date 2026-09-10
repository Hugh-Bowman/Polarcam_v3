from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.filters import sobel


EPS = 1e-12


def fit_alpha_beta(frame: np.ndarray, bg: np.ndarray) -> tuple[float, float]:
    x = bg.ravel().astype(np.float64)
    y = frame.ravel().astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    beta = float(np.dot(xv, yv) / (np.dot(xv, xv) + EPS))
    alpha = float(ym - beta * xm)
    return alpha, beta


def qfit(i: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * i + c * i * i


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)


def p98(arr: np.ndarray) -> float:
    v = float(np.percentile(arr, 98))
    return v if np.isfinite(v) and v > 0 else 1.0


def main() -> None:
    base = Path(r".")
    out = base / "gpt prompted exploration"
    out.mkdir(parents=True, exist_ok=True)

    coverslip_dir = base / "recordings" / "2026-05-06" / "widefield_frames" / "still coverslip recordings"
    dark_path = base / "recordings" / "2026-05-06" / "widefield_frames" / "camera_covered.npy"
    coverslip_paths = sorted(coverslip_dir.glob("*.npy"))
    if len(coverslip_paths) < 2:
        raise RuntimeError("Need at least 2 coverslip recordings for multi-location evaluation.")

    dark = np.load(dark_path, mmap_mode="r")
    dark_mean = dark.mean(axis=0, dtype=np.float64)
    dark_var = dark.var(axis=0, dtype=np.float64)

    # Load coverslip stacks as mmap, then compute dark-corrected first/second moments per recording.
    stacks = [np.load(p, mmap_mode="r") for p in coverslip_paths]
    names = [p.stem for p in coverslip_paths]
    n_rec = len(stacks)

    mu_list = []
    var_list = []
    frame_means = []
    frame_stds = []
    n_frames = []
    for s in stacks:
        n_frames.append(s.shape[0])
        mu = s.mean(axis=0, dtype=np.float64) - dark_mean
        var = s.var(axis=0, dtype=np.float64)  # dark subtraction by constant mean does not change temporal var
        mu_list.append(mu)
        var_list.append(var)
        fm = np.array([(s[t].astype(np.float64) - dark_mean).mean() for t in range(s.shape[0])], dtype=np.float64)
        fs = np.array([(s[t].astype(np.float64) - dark_mean).std() for t in range(s.shape[0])], dtype=np.float64)
        frame_means.append(fm)
        frame_stds.append(fs)

    # Static candidate from multi-location coverslip: average of dark-corrected means.
    B_cov = np.mean(mu_list, axis=0)
    grad_mag = sobel(B_cov)

    # Align each coverslip mean to B_cov with alpha+beta and compute location-specific residual structure.
    aligned_mean_resids = []
    mean_alphas = []
    mean_betas = []
    for mu in mu_list:
        a, b = fit_alpha_beta(mu, B_cov)
        mean_alphas.append(a)
        mean_betas.append(b)
        aligned_mean_resids.append(mu - (a + b * B_cov))
    aligned_mean_resid_std = np.std(np.stack(aligned_mean_resids, axis=0), axis=0)

    # Evaluate residual temporal variance for each method across recordings.
    methods = ["dark_only", "subtract_B_cov", "fit_alpha_beta_B_cov"]
    residual_metrics = {m: [] for m in methods}

    # Keep pooled points for Var(I) under best method
    pooled_I = []
    pooled_V = []
    pooled_G = []

    # For map visualization of best method variance
    best_var_maps = []

    for s, mu, var in zip(stacks, mu_list, var_list):
        # dark_only
        residual_metrics["dark_only"].append(float(np.mean(var)))
        # subtract_B_cov (fixed subtraction does not change per-pixel temporal variance)
        residual_metrics["subtract_B_cov"].append(float(np.mean(var)))

        # fit alpha+beta per frame
        t, h, w = s.shape
        mean_acc = np.zeros((h, w), dtype=np.float64)
        m2_acc = np.zeros((h, w), dtype=np.float64)
        for i in range(t):
            fr = s[i].astype(np.float64) - dark_mean
            a, b = fit_alpha_beta(fr, B_cov)
            r = fr - (a + b * B_cov)
            d = r - mean_acc
            mean_acc += d / (i + 1)
            d2 = r - mean_acc
            m2_acc += d * d2
        rv = m2_acc / t
        residual_metrics["fit_alpha_beta_B_cov"].append(float(np.mean(rv)))
        best_var_maps.append(rv)

        # Var(I) for best method using corrected mean intensity
        I = mean_acc.ravel()
        V = rv.ravel()
        G = grad_mag.ravel()
        m = np.isfinite(I) & np.isfinite(V) & np.isfinite(G)
        pooled_I.append(I[m])
        pooled_V.append(V[m])
        pooled_G.append(G[m])

    pooled_I = np.concatenate(pooled_I)
    pooled_V = np.concatenate(pooled_V)
    pooled_G = np.concatenate(pooled_G)

    # Fit Var(I) = a + bI + cI^2
    # Bin first for robust trend fit
    qs = np.linspace(0, 1, 81)
    edges = np.unique(np.quantile(pooled_I, qs))
    centers = []
    meds = []
    q10 = []
    q90 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        m = (pooled_I >= lo) & (pooled_I < hi)
        if np.count_nonzero(m) < 200:
            continue
        y = pooled_V[m]
        centers.append(0.5 * (lo + hi))
        meds.append(np.median(y))
        q10.append(np.quantile(y, 0.1))
        q90.append(np.quantile(y, 0.9))
    centers = np.asarray(centers, dtype=np.float64)
    meds = np.asarray(meds, dtype=np.float64)
    q10 = np.asarray(q10, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)

    coef, _ = curve_fit(qfit, centers, meds, maxfev=10000)
    a, b, c = [float(x) for x in coef]
    fit = qfit(centers, a, b, c)
    ss_res = float(np.sum((meds - fit) ** 2))
    ss_tot = float(np.sum((meds - meds.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)

    # Extra non-static descriptor: dependence on gradient
    corr_var_grad = float(np.corrcoef(pooled_V, pooled_G)[0, 1])

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(dark_mean, cmap="viridis", vmin=0, vmax=p98(dark_mean))
    axes[0].set_title("camera_covered mean (dark offset)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(dark_var, cmap="magma", vmin=0, vmax=p98(dark_var))
    axes[1].set_title("camera_covered variance")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    savefig(fig, out / "dark_offset_maps.png")

    # 4 coverslip means
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    vmax_mu = p98(np.stack(mu_list, axis=0))
    for ax, mu, nm in zip(axes2.ravel(), mu_list, names):
        im = ax.imshow(mu, cmap="viridis", vmin=0, vmax=vmax_mu)
        ax.set_title(f"{nm} mean-dark")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig2, out / "coverslip_means_dark_corrected.png")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    im = axes3[0].imshow(B_cov, cmap="viridis", vmin=0, vmax=p98(B_cov))
    axes3[0].set_title("B_cov (multi-location static model)")
    fig3.colorbar(im, ax=axes3[0], fraction=0.046, pad=0.04)
    im = axes3[1].imshow(aligned_mean_resid_std, cmap="magma", vmin=0, vmax=p98(aligned_mean_resid_std))
    axes3[1].set_title("Std across locations after affine align")
    fig3.colorbar(im, ax=axes3[1], fraction=0.046, pad=0.04)
    im = axes3[2].imshow(grad_mag, cmap="magma", vmin=0, vmax=p98(grad_mag))
    axes3[2].set_title("|grad(B_cov)|")
    fig3.colorbar(im, ax=axes3[2], fraction=0.046, pad=0.04)
    for ax in axes3:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    savefig(fig3, out / "static_vs_nonstatic_maps.png")

    # Residual metric comparison
    fig4, ax4 = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = np.arange(n_rec)
    w = 0.25
    ax4.bar(x - w, residual_metrics["dark_only"], width=w, label="dark_only")
    ax4.bar(x, residual_metrics["subtract_B_cov"], width=w, label="subtract_B_cov")
    ax4.bar(x + w, residual_metrics["fit_alpha_beta_B_cov"], width=w, label="fit_alpha_beta_B_cov")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=20, ha="right")
    ax4.set_ylabel("Average residual variance")
    ax4.set_title("Residual variance by method and recording")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.25)
    savefig(fig4, out / "method_variance_comparison.png")

    # Var(I) scatter + fit
    rng = np.random.default_rng(0)
    nplot = min(250_000, pooled_I.size)
    idx = rng.choice(pooled_I.size, size=nplot, replace=False)
    fig5, ax5 = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax5.scatter(pooled_I[idx], pooled_V[idx], s=2, alpha=0.05, color="tab:blue", label="Pixels")
    ax5.fill_between(centers, q10, q90, color="tab:orange", alpha=0.2, label="10-90%")
    ax5.plot(centers, meds, color="tab:orange", linewidth=2, label="Binned median")
    ax5.plot(centers, fit, color="tab:red", linewidth=2, label="a+bI+cI^2 fit")
    ax5.set_xlabel("Residual mean intensity I (after dark + affine B_cov)")
    ax5.set_ylabel("Residual variance")
    ax5.set_title("Residual Var(I) across all coverslip locations")
    ax5.grid(alpha=0.25)
    ax5.legend()
    savefig(fig5, out / "residual_var_vs_intensity_best_method.png")

    # Var vs gradient (non-static indicator)
    nplot2 = min(250_000, pooled_G.size)
    idx2 = rng.choice(pooled_G.size, size=nplot2, replace=False)
    fig6, ax6 = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax6.scatter(pooled_G[idx2], pooled_V[idx2], s=2, alpha=0.05, color="tab:green")
    ax6.set_xlabel("|grad(B_cov)|")
    ax6.set_ylabel("Residual variance")
    ax6.set_title("Residual variance vs static-gradient magnitude")
    ax6.grid(alpha=0.25)
    savefig(fig6, out / "residual_variance_vs_gradient.png")

    # Report numbers
    best_method = min(methods, key=lambda m: float(np.mean(residual_metrics[m])))
    static_rms = float(np.sqrt(np.mean(B_cov ** 2)))
    nonstatic_rms = float(np.sqrt(np.mean(aligned_mean_resid_std ** 2)))
    nonstatic_to_static_ratio = nonstatic_rms / (static_rms + EPS)

    report = {
        "recordings": [str(p) for p in coverslip_paths],
        "dark_recording": str(dark_path),
        "avg_residual_variance_per_method": {
            m: float(np.mean(v)) for m, v in residual_metrics.items()
        },
        "best_method": best_method,
        "static_component_rms": static_rms,
        "nonstatic_location_component_rms": nonstatic_rms,
        "nonstatic_to_static_ratio": nonstatic_to_static_ratio,
        "mean_affine_alpha_across_locations": float(np.mean(mean_alphas)),
        "std_affine_alpha_across_locations": float(np.std(mean_alphas)),
        "mean_affine_beta_across_locations": float(np.mean(mean_betas)),
        "std_affine_beta_across_locations": float(np.std(mean_betas)),
        "varI_fit_best_method": {
            "a": a,
            "b": b,
            "c": c,
            "r2_binned_median": r2,
        },
        "corr_residual_variance_with_gradient": corr_var_grad,
    }

    (out / "multi_coverslip_eval_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "Multi-coverslip background evaluation",
        "",
        f"Best background correction method: {best_method}",
        f"Average residual variance (dark_only): {np.mean(residual_metrics['dark_only']):.6f}",
        f"Average residual variance (subtract_B_cov): {np.mean(residual_metrics['subtract_B_cov']):.6f}",
        f"Average residual variance (fit_alpha_beta_B_cov): {np.mean(residual_metrics['fit_alpha_beta_B_cov']):.6f}",
        "",
        f"Static RMS (B_cov): {static_rms:.6f}",
        f"Non-static location RMS (post-align std): {nonstatic_rms:.6f}",
        f"Non-static/static ratio: {nonstatic_to_static_ratio:.6f}",
        "",
        "Residual Var(I) for best method:",
        f"Var = a + b I + c I^2 with a={a:.6g}, b={b:.6g}, c={c:.6g}, R^2={r2:.6f}",
        f"Residual variance correlation with |grad(B_cov)|: {corr_var_grad:.6f}",
        "",
        "Interpretation:",
        "1) Use dark correction from camera_covered first.",
        "2) Use multi-location B_cov as static background model.",
        "3) Use per-frame affine correction alpha+beta*B_cov; fixed subtraction alone does not reduce temporal variance.",
        "4) Treat remaining non-static background as heteroscedastic residual noise modeled by Var(I) (and gradient-linked excess where needed).",
    ]
    (out / "multi_coverslip_eval_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(out / "multi_coverslip_eval_summary.json")
    print(out / "multi_coverslip_eval_report.txt")


if __name__ == "__main__":
    main()

