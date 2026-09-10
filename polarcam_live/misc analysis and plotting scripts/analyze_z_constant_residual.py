from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (an * bn))


def _sin_model(x: np.ndarray, c: float, a: float, p: float, phi: float) -> np.ndarray:
    return c + a * np.cos(2.0 * np.pi * x / p + phi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove z-constant component and analyze residual correlations."
    )
    parser.add_argument(
        "--composite-dir",
        type=Path,
        required=True,
        help="Directory containing z1..z8 composite .npy files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_composite_bgsub.npy",
        help="Filename suffix after z index, default: _composite_bgsub.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <composite-dir>/constant_residual_analysis).",
    )
    args = parser.parse_args()

    comp_dir = args.composite_dir
    out_dir = args.out_dir or (comp_dir / "constant_residual_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    z_imgs = []
    for zi in range(1, 9):
        p = comp_dir / f"z{zi}{args.suffix}"
        if not p.exists():
            raise FileNotFoundError(f"Missing composite file: {p}")
        z_imgs.append(np.load(p).astype(np.float32))
    z_stack = np.stack(z_imgs, axis=0)  # (8, H, W)

    # z-invariant component
    z_const = z_stack.mean(axis=0)
    # z-varying residual
    z_res = z_stack - z_const[None, :, :]

    np.save(out_dir / "z_constant_component.npy", z_const.astype(np.float32))
    np.save(out_dir / "z_residual_stack.npy", z_res.astype(np.float32))

    # save per-z residual images
    for zi in range(1, 9):
        r = z_res[zi - 1]
        np.save(out_dir / f"z{zi}_residual.npy", r.astype(np.float32))
        v = float(max(np.percentile(np.abs(r), 99), 1e-6))
        plt.figure(figsize=(8, 6.5))
        plt.imshow(r, cmap="coolwarm", vmin=-v, vmax=v)
        plt.title(f"z{zi} residual (z-varying component)")
        plt.colorbar(label="Residual intensity (a.u.)")
        plt.tight_layout()
        plt.savefig(out_dir / f"z{zi}_residual.png", dpi=150)
        plt.close()

    # save constant component preview
    cmin, cmax = np.percentile(z_const, 1), np.percentile(z_const, 99)
    plt.figure(figsize=(8, 6.5))
    plt.imshow(z_const, cmap="gray", vmin=cmin, vmax=cmax)
    plt.title("z-constant component (mean across z1..z8)")
    plt.colorbar(label="Intensity (a.u.)")
    plt.tight_layout()
    plt.savefig(out_dir / "z_constant_component.png", dpi=150)
    plt.close()

    # residual correlation to z1 residual
    corr_res = np.array([_norm_corr(z_res[i], z_res[0]) for i in range(8)], dtype=np.float64)

    # constrained sinusoid fit in physically meaningful correlation bounds
    x = np.arange(1, 9, dtype=np.float64)
    y = corr_res
    p0 = [float(y.mean()), float(np.clip((y.max() - y.min()) / 2.0, 0.01, 0.6)), 4.0, 0.0]
    bounds = ([-1.0, 0.0, 2.0, -2.0 * np.pi], [1.0, 1.0, 30.0, 2.0 * np.pi])
    popt, _ = curve_fit(_sin_model, x, y, p0=p0, bounds=bounds, maxfev=500000)
    c, a, p, phi = [float(v) for v in popt]
    yhat = _sin_model(x, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    # plot residual correlation + fit
    xd = np.linspace(1, 8, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, "ok", label="Residual corr to z1 residual")
    plt.plot(xd, _sin_model(xd, c, a, p, phi), "-r", lw=2, label=f"Constrained sinusoid P={p:.3f}")
    plt.xlabel("z index")
    plt.ylabel("Correlation")
    plt.title("Residual-only correlation vs z")
    plt.xticks(np.arange(1, 9))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_correlation_to_z1_with_constrained_sinusoid.png", dpi=150)
    plt.close()

    np.savez(
        out_dir / "residual_correlation_fit_metrics.npz",
        x=x,
        correlations_to_z1_residual=corr_res,
        z_constant_component=z_const.astype(np.float32),
        constrained_params=np.array([c, a, p, phi], dtype=np.float64),
        constrained_r2=r2,
        constrained_rmse=rmse,
    )

    summary = []
    summary.append(f"composite_dir: {comp_dir}")
    summary.append("operation: z_constant = mean(z1..z8), z_residual = z - z_constant")
    summary.append(f"residual correlations to z1: {np.array2string(corr_res, precision=6, separator=', ')}")
    summary.append(f"constrained fit [C,A,P,phi]: [{c:.6f}, {a:.6f}, {p:.6f}, {phi:.6f}]")
    summary.append(f"fit R2={r2:.6f}, RMSE={rmse:.6f}")
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
