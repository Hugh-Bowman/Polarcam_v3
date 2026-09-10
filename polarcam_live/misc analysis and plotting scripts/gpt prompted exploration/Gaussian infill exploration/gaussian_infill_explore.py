from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"c:\Polarcam Software\Polarcam_v3\polarcam_live\gpt prompted exploration\Background ml recon")
VAL_DIR = ROOT / "data" / "val_backgrounds_ds2"
MASK_DIR = ROOT / "data" / "val_masks_ds2"
STACK_TOKEN = "145155"
SMOOTH_SIGMAS = [12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024]
MAX_EXAMPLES = 4
OUT_DIR = Path(r"c:\Polarcam Software\Polarcam_v3\polarcam_live\gpt prompted exploration\Gaussian infill exploration\outputs")


def build_triplets() -> list[tuple[Path, Path]]:
    triplets: list[tuple[Path, Path]] = []
    for t_path in sorted(VAL_DIR.glob("*.npy")):
        if STACK_TOKEN and STACK_TOKEN not in t_path.stem:
            continue
        m_path = MASK_DIR / f"{t_path.stem}_mask.npy"
        if m_path.exists():
            triplets.append((t_path, m_path))
    return triplets


def smooth_mask_aware(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    known = 1.0 - mask
    num = cv2.GaussianBlur((img * known).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return (num / (den + 1e-8)).astype(np.float32)


def gaussian_infill(masked_input: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    known = 1.0 - mask
    num = cv2.GaussianBlur((masked_input * known).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    local_bg = num / (den + 1e-8)
    pred = masked_input.copy()
    hole = mask > 0.5
    pred[hole] = local_bg[hole]
    return pred.astype(np.float32)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    return float(np.abs(pred[sel] - target[sel]).mean()) if np.any(sel) else float("nan")


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    return float(np.sqrt(np.mean((pred[sel] - target[sel]) ** 2))) if np.any(sel) else float("nan")


def save_example_panel(out_path: Path, raw: np.ndarray, smooth_t: np.ndarray, mask: np.ndarray, masked_in: np.ndarray, pred: np.ndarray) -> None:
    err = np.abs(pred - smooth_t) * mask
    stack = np.concatenate([raw.ravel(), smooth_t.ravel(), masked_in.ravel(), pred.ravel()])
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(stack)), float(np.max(stack))
    if hi <= lo:
        hi = lo + 1e-6
    e_hi = float(np.percentile(err, 99.0))
    if e_hi <= 0:
        e_hi = float(np.max(err))
    if e_hi <= 0:
        e_hi = 1e-6

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    axes[0].imshow(raw, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Raw")
    axes[1].imshow(smooth_t, cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("Smoothed Target")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask")
    axes[3].imshow(masked_in, cmap="gray", vmin=lo, vmax=hi)
    axes[3].set_title("Smoothed Masked Input")
    axes[4].imshow(pred, cmap="gray", vmin=lo, vmax=hi)
    axes[4].set_title("Gaussian Infill")
    axes[5].imshow(err, cmap="magma", vmin=0, vmax=e_hi)
    axes[5].set_title("Masked Abs Error")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    triplets = build_triplets()
    if not triplets:
        raise RuntimeError("No validation triplets found for Gaussian exploration")

    rows: list[dict[str, float | int]] = []
    for sigma in SMOOTH_SIGMAS:
        sigma_dir = OUT_DIR / f"sigma_{sigma}"
        sigma_dir.mkdir(parents=True, exist_ok=True)
        maes: list[float] = []
        rmses: list[float] = []

        for i, (img_path, mask_path) in enumerate(triplets):
            raw = np.load(img_path).astype(np.float32)
            mask = (np.load(mask_path).astype(np.float32) > 0.5).astype(np.float32)
            smooth_t = smooth_mask_aware(raw, mask, float(sigma))
            masked_in = smooth_t * (1.0 - mask)
            pred = gaussian_infill(masked_in, mask, float(sigma))

            # Score against the unsmoothed/raw target inside masked region.
            mae = masked_mae(pred, raw, mask)
            rmse = masked_rmse(pred, raw, mask)
            maes.append(mae)
            rmses.append(rmse)

            if i < MAX_EXAMPLES:
                out_img = sigma_dir / f"example_{i:03d}_{img_path.stem}.png"
                save_example_panel(out_img, raw, smooth_t, mask, masked_in, pred)

        rows.append(
            {
                "sigma": int(sigma),
                "num_images": int(len(triplets)),
                "masked_mae_mean": float(np.nanmean(maes)),
                "masked_mae_std": float(np.nanstd(maes)),
                "masked_rmse_mean": float(np.nanmean(rmses)),
                "masked_rmse_std": float(np.nanstd(rmses)),
            }
        )

    csv_path = OUT_DIR / "gaussian_infill_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sigma", "num_images", "masked_mae_mean", "masked_mae_std", "masked_rmse_mean", "masked_rmse_std"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[done] wrote: {csv_path}")
    for row in rows:
        print(
            f"sigma={row['sigma']}: masked_mae_mean={row['masked_mae_mean']:.6f}, "
            f"masked_rmse_mean={row['masked_rmse_mean']:.6f}, n={row['num_images']}"
        )


if __name__ == "__main__":
    main()
