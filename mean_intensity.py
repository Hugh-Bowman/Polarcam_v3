import argparse
import re
import csv
from pathlib import Path

import numpy as np
from PIL import Image


# --- Sweep defaults ----------------------------------------------------------
DEFAULT_SWEEP_DIR = (
    r"C:\IDS recordings\X offset experiment\Sweep_to_fit_pol_angles\pol angles phone torch"
)
# For files like "15 deg.bmp" or "15degree.bmp" (plain "15.bmp" handled separately).
ANGLE_REGEX = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)\b", re.IGNORECASE)


def compute_block_means(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        # Convert RGB to luminance-like grayscale via mean.
        arr = arr.mean(axis=2)
    h, w = arr.shape
    # Trim to even dimensions.
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    arr = arr[:h2, :w2]

    # Compute means for each 2x2 position.
    means = np.zeros((2, 2), dtype=float)
    means[0, 0] = arr[0::2, 0::2].mean()
    means[0, 1] = arr[0::2, 1::2].mean()
    means[1, 0] = arr[1::2, 0::2].mean()
    means[1, 1] = arr[1::2, 1::2].mean()
    return means


def _as_gray_f64(img: Image.Image) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def _parse_angle_from_name(p: Path) -> float | None:
    # Accept simple numeric stems like "45.bmp" or "45.bitmap".
    s = p.stem.strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None

    m = ANGLE_REGEX.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def load_sweep_images(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images in `folder` with an angle encoded in the filename like '15 deg'.

    Returns:
      angles_deg: shape (N,)
      means: shape (N, 2, 2)  (raw mean intensity per 2x2 mosaic position)
    """
    if not folder.exists():
        raise SystemExit(f"Sweep folder not found: {folder}")

    paths: list[tuple[float, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".bmp", ".bitmap", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        ang = _parse_angle_from_name(p)
        if ang is None:
            continue
        paths.append((float(ang), p))

    if not paths:
        raise SystemExit(
            f"No sweep images found in {folder} (expected names like '15.bmp' or '15 deg.bmp')."
        )

    paths.sort(key=lambda t: t[0])
    angles: list[float] = []
    means: list[np.ndarray] = []
    for ang, p in paths:
        img = Image.open(p)
        arr = _as_gray_f64(img)
        m = compute_block_means(arr)
        angles.append(float(ang))
        means.append(m)

    return np.asarray(angles, dtype=float), np.asarray(means, dtype=float)


def fit_analyzer_angles(
    angles_deg: np.ndarray,
    means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-position analyzer angle alpha using a linearized Malus model:

        I(theta) = c0 + c1*cos(2theta) + c2*sin(2theta)

    which corresponds to:
        I(theta) = offset + amp * cos(2(theta - alpha))

    Returns:
      alpha_deg: shape (2,2) analyzer angles in degrees (mod 180)
      coeffs: shape (2,2,3) with [c0, c1, c2]
      pred: shape (N,2,2) fitted values at the measured angles
    """
    angles_deg = np.asarray(angles_deg, dtype=float).reshape(-1)
    means = np.asarray(means, dtype=float)
    if means.ndim != 3 or means.shape[1:] != (2, 2):
        raise ValueError(f"means must have shape (N,2,2); got {means.shape}")
    if means.shape[0] != angles_deg.shape[0]:
        raise ValueError("angles and means length mismatch")

    theta = np.deg2rad(angles_deg)
    X = np.column_stack([np.ones_like(theta), np.cos(2.0 * theta), np.sin(2.0 * theta)])

    coeffs = np.zeros((2, 2, 3), dtype=float)
    alpha = np.zeros((2, 2), dtype=float)
    pred = np.zeros_like(means, dtype=float)

    for r in range(2):
        for c in range(2):
            y = means[:, r, c]
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            coeffs[r, c, :] = beta
            pred[:, r, c] = X @ beta
            # c1 = k*cos(2a), c2 = k*sin(2a) -> 2a = atan2(c2, c1)
            a = 0.5 * np.arctan2(beta[2], beta[1])
            alpha[r, c] = (np.rad2deg(a) % 180.0)

    return alpha, coeffs, pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 2x2 block-position averages. If --sweep-dir is provided (or the default "
            "sweep directory exists), load images named like '15 deg.bmp' and fit per-position "
            "analyzer angles. The fit uses I(theta)=c0+c1*cos(2theta)+c2*sin(2theta)."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=r"C:\IDS recordings\X offset experiment\45 degree polarised rods.bmp",
        help="Path to bitmap image",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help=f"Sweep folder (default: {DEFAULT_SWEEP_DIR})",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a model-vs-data plot into the sweep folder (requires matplotlib).",
    )
    args = parser.parse_args()

    # Sweep mode if explicitly requested, or if the default sweep folder exists.
    if args.sweep_dir is not None or Path(DEFAULT_SWEEP_DIR).exists():
        sweep_dir = Path(args.sweep_dir or DEFAULT_SWEEP_DIR)
        angles_deg, means = load_sweep_images(sweep_dir)
        alpha_deg, coeffs, pred = fit_analyzer_angles(angles_deg, means)

        ext_deg = (alpha_deg + 90.0) % 180.0
        # Rebase angles so (0,0) becomes 0 deg and the others are offsets from that.
        # We wrap to [0, 180) because Malus-law polarization angles are 180-deg periodic.
        base_alpha = float(alpha_deg[0, 0])
        alpha_rel_deg = (alpha_deg - base_alpha) % 180.0
        base_ext = float(ext_deg[0, 0])
        ext_rel_deg = (ext_deg - base_ext) % 180.0
        rmse = np.sqrt(np.mean((pred - means) ** 2, axis=0))

        print("Fitted analyzer angles alpha (deg, mod 180):")
        print(f"[{alpha_deg[0,0]:.3f} {alpha_deg[0,1]:.3f}]")
        print(f"[{alpha_deg[1,0]:.3f} {alpha_deg[1,1]:.3f}]")
        print(f"Fitted analyzer angles alpha (rebased so alpha[0,0]={base_alpha:.3f} deg -> 0):")
        print(f"[{alpha_rel_deg[0,0]:.3f} {alpha_rel_deg[0,1]:.3f}]")
        print(f"[{alpha_rel_deg[1,0]:.3f} {alpha_rel_deg[1,1]:.3f}]")
        print("Implied extinction angles (alpha + 90 deg, mod 180):")
        print(f"[{ext_deg[0,0]:.3f} {ext_deg[0,1]:.3f}]")
        print(f"[{ext_deg[1,0]:.3f} {ext_deg[1,1]:.3f}]")
        print(f"Implied extinction angles (rebased so ext[0,0]={base_ext:.3f} deg -> 0):")
        print(f"[{ext_rel_deg[0,0]:.3f} {ext_rel_deg[0,1]:.3f}]")
        print(f"[{ext_rel_deg[1,0]:.3f} {ext_rel_deg[1,1]:.3f}]")
        print("RMSE (raw intensity units) per position:")
        print(f"[{rmse[0,0]:.6f} {rmse[0,1]:.6f}]")
        print(f"[{rmse[1,0]:.6f} {rmse[1,1]:.6f}]")

        # Save a simple table of data vs model so you can inspect/plot elsewhere.
        out_csv = sweep_dir / "fit_pol_angles_results.csv"
        try:
            with out_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "angle_deg",
                        "m00",
                        "m01",
                        "m10",
                        "m11",
                        "p00",
                        "p01",
                        "p10",
                        "p11",
                    ]
                )
                for i, ang in enumerate(angles_deg.tolist()):
                    m = means[i]
                    p = pred[i]
                    w.writerow(
                        [
                            float(ang),
                            float(m[0, 0]),
                            float(m[0, 1]),
                            float(m[1, 0]),
                            float(m[1, 1]),
                            float(p[0, 0]),
                            float(p[0, 1]),
                            float(p[1, 0]),
                            float(p[1, 1]),
                        ]
                    )
            print(f"Saved table: {out_csv}")
        except Exception as e:
            print(f"Could not write CSV table: {e}")

        if args.plot:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                dense_deg = np.linspace(float(np.min(angles_deg)), float(np.max(angles_deg)), 721)
                dense_th = np.deg2rad(dense_deg)
                Xd = np.column_stack(
                    [np.ones_like(dense_th), np.cos(2.0 * dense_th), np.sin(2.0 * dense_th)]
                )

                fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
                for r in range(2):
                    for c in range(2):
                        ax = axes[r][c]
                        y = means[:, r, c]
                        b = coeffs[r, c, :]
                        yhat = Xd @ b
                        ax.plot(angles_deg, y, "o", ms=4, label="data")
                        ax.plot(dense_deg, yhat, "-", lw=2, label="fit")
                        ax.set_title(
                            f"pos ({r},{c})  alpha={alpha_deg[r,c]:.1f} deg  (rebased {alpha_rel_deg[r,c]:.1f})"
                        )
                        ax.grid(True, alpha=0.3)
                        if r == 1:
                            ax.set_xlabel("Polarizer angle (deg)")
                        if c == 0:
                            ax.set_ylabel("Mean intensity")
                axes[0][0].legend(loc="best", fontsize=9)
                fig.tight_layout()
                out = sweep_dir / "fit_pol_angles_model_vs_data.png"
                fig.savefig(out, dpi=160)
                print(f"Saved plot: {out}")
            except Exception as e:
                print(f"Plot skipped (matplotlib not available or failed): {e}")

        return

    # Single-image mode.
    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    img = Image.open(path)
    arr = _as_gray_f64(img)
    means = compute_block_means(arr)

    # Print as a 2x2 grid.
    print("Measured mean intensity (raw):")
    print(f"[{means[0,0]:.6f} {means[0,1]:.6f}]")
    print(f"[{means[1,0]:.6f} {means[1,1]:.6f}]")


if __name__ == "__main__":
    main()
