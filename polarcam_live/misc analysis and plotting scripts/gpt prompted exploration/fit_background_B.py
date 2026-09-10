from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


def solve_alpha_beta(frame: np.ndarray, B: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Weighted LS solve for frame ~= alpha + beta * B on masked pixels."""
    m = mask.ravel().astype(bool)
    if np.count_nonzero(m) < 20:
        return 0.0, 1.0
    x = B.ravel()[m].astype(np.float64)
    y = frame.ravel()[m].astype(np.float64)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xv = x - xm
    yv = y - ym
    beta = float(np.dot(xv, yv) / (np.dot(xv, xv) + EPS))
    alpha = float(ym - beta * xm)
    return alpha, beta


def estimate_B_given_alpha_beta(
    frames: np.ndarray,
    masks: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Pixel-wise closed-form update:
      B_j = sum_t m_tj * beta_t * (F_tj - alpha_t) / sum_t m_tj * beta_t^2
    """
    t, h, w = frames.shape
    num = np.zeros((h, w), dtype=np.float64)
    den = np.zeros((h, w), dtype=np.float64)
    for i in range(t):
        m = masks[i].astype(np.float64)
        b = float(beta[i])
        a = float(alpha[i])
        f = frames[i].astype(np.float64)
        num += m * b * (f - a)
        den += m * (b * b)
    return num / (den + EPS)


def robust_initial_B(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Initialize B as masked temporal median (fallback to mean if needed)."""
    t, h, w = frames.shape
    vals = np.where(masks > 0, frames, np.nan).astype(np.float64)
    B = np.nanmedian(vals, axis=0)
    bad = ~np.isfinite(B)
    if np.any(bad):
        mean_f = np.mean(frames.astype(np.float64), axis=0)
        B[bad] = mean_f[bad]
    return B


def percentile_scale(img: np.ndarray, hi: float = 98.0) -> Tuple[float, float]:
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, hi))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit background profile B via frame-wise alpha+beta model.")
    parser.add_argument("--frames", type=Path, required=True, help="Input frame stack .npy with shape (T,H,W).")
    parser.add_argument(
        "--include-masks",
        type=Path,
        default=None,
        help="Optional include-mask stack .npy with shape (T,H,W), nonzero=include.",
    )
    parser.add_argument(
        "--exclude-masks",
        type=Path,
        default=None,
        help="Optional exclude-mask stack .npy with shape (T,H,W), nonzero=exclude.",
    )
    parser.add_argument(
        "--dark-mean",
        type=Path,
        default=None,
        help="Optional dark-mean image .npy (H,W) to subtract from every frame before fitting.",
    )
    parser.add_argument("--iters", type=int, default=8, help="ALS iterations.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = np.load(args.frames, mmap_mode="r")
    if frames.ndim != 3:
        raise ValueError(f"--frames must be 3D (T,H,W), got {frames.shape}")
    t, h, w = frames.shape

    if args.dark_mean is not None:
        dark = np.load(args.dark_mean).astype(np.float64)
        if dark.shape != (h, w):
            raise ValueError(f"dark shape {dark.shape} != frame shape {(h,w)}")
    else:
        dark = np.zeros((h, w), dtype=np.float64)

    # Build masks (include=1 means pixel is usable for fitting).
    if args.include_masks is not None:
        masks = np.load(args.include_masks)
        if masks.shape != (t, h, w):
            raise ValueError(f"include mask shape {masks.shape} != {(t,h,w)}")
        masks = (masks > 0).astype(np.uint8)
    elif args.exclude_masks is not None:
        ex = np.load(args.exclude_masks)
        if ex.shape != (t, h, w):
            raise ValueError(f"exclude mask shape {ex.shape} != {(t,h,w)}")
        masks = (ex == 0).astype(np.uint8)
    else:
        masks = np.ones((t, h, w), dtype=np.uint8)

    # Dark-corrected floating frames in RAM for speed during ALS.
    f = frames.astype(np.float64) - dark[None, :, :]

    # Initialize
    B = robust_initial_B(f, masks)
    alpha = np.zeros(t, dtype=np.float64)
    beta = np.ones(t, dtype=np.float64)
    history = []

    for it in range(max(1, args.iters)):
        # Update alpha,beta for each frame given current B
        for i in range(t):
            alpha[i], beta[i] = solve_alpha_beta(f[i], B, masks[i])

        # Gauge fix to keep scale stable: normalize beta around 1 and absorb into B.
        beta_mean = float(np.mean(beta))
        if abs(beta_mean) < EPS:
            beta_mean = 1.0
        B *= beta_mean
        beta /= beta_mean

        # Update B
        B = estimate_B_given_alpha_beta(f, masks, alpha, beta)

        # Objective on masked pixels
        sse = 0.0
        npx = 0
        for i in range(t):
            r = (f[i] - (alpha[i] + beta[i] * B))
            m = masks[i] > 0
            rr = r[m]
            sse += float(np.dot(rr, rr))
            npx += int(rr.size)
        rmse = float(np.sqrt(sse / max(1, npx)))
        history.append({"iter": it + 1, "rmse_masked": rmse})
        print(f"iter {it+1:02d}: rmse_masked={rmse:.6f}")

    # Residual variance map and global fit diagnostics
    mean_acc = np.zeros((h, w), dtype=np.float64)
    m2_acc = np.zeros((h, w), dtype=np.float64)
    for i in range(t):
        r = f[i] - (alpha[i] + beta[i] * B)
        d = r - mean_acc
        mean_acc += d / (i + 1)
        d2 = r - mean_acc
        m2_acc += d * d2
    resid_var = m2_acc / t

    # Save arrays
    np.save(args.out_dir / "B_profile.npy", B.astype(np.float32))
    np.save(args.out_dir / "alpha_per_frame.npy", alpha.astype(np.float32))
    np.save(args.out_dir / "beta_per_frame.npy", beta.astype(np.float32))
    np.save(args.out_dir / "residual_variance_map.npy", resid_var.astype(np.float32))

    # Save diagnostics
    summary = {
        "frames": str(args.frames),
        "shape": [int(t), int(h), int(w)],
        "iters": int(args.iters),
        "mask_include_fraction_mean": float(np.mean(masks)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta)),
        "rmse_history": history,
        "final_rmse_masked": history[-1]["rmse_masked"] if history else None,
    }
    (args.out_dir / "B_fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    vmin, vmax = percentile_scale(B, 98)
    im0 = axes[0].imshow(B, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Fitted background profile B")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmin2, vmax2 = percentile_scale(resid_var, 98)
    im1 = axes[1].imshow(resid_var, cmap="magma", vmin=max(0.0, vmin2), vmax=vmax2)
    axes[1].set_title("Residual variance map")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(args.out_dir / "B_and_residual_variance.png", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    x = np.arange(t)
    axes2[0].plot(x, alpha, color="tab:blue")
    axes2[0].set_title("alpha_t")
    axes2[0].set_xlabel("frame")
    axes2[0].set_ylabel("alpha")
    axes2[0].grid(alpha=0.25)

    axes2[1].plot(x, beta, color="tab:orange")
    axes2[1].set_title("beta_t")
    axes2[1].set_xlabel("frame")
    axes2[1].set_ylabel("beta")
    axes2[1].grid(alpha=0.25)

    rmse_hist = [h["rmse_masked"] for h in history]
    axes2[2].plot(np.arange(1, len(rmse_hist) + 1), rmse_hist, marker="o", color="tab:green")
    axes2[2].set_title("ALS objective convergence")
    axes2[2].set_xlabel("iteration")
    axes2[2].set_ylabel("masked RMSE")
    axes2[2].grid(alpha=0.25)
    fig2.savefig(args.out_dir / "B_fit_diagnostics.png", dpi=150)
    plt.close(fig2)

    print(f"Saved: {args.out_dir / 'B_profile.npy'}")
    print(f"Saved: {args.out_dir / 'B_fit_summary.json'}")


if __name__ == "__main__":
    main()

