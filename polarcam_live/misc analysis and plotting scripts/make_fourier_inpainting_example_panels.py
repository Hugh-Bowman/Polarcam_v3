"""
Create example inpainting panels from the best saved Fourier inpainting model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_fourier_inpainting_net import (
    Config,
    DEFAULT_DATA_DIR,
    DEFAULT_MEAN_DIR,
    FourierInpaintNet,
    FourierPatchDataset,
    butterworth_lowpass,
    ifft_from_channels,
    load_blank_background,
    masked_rmse,
)


MODEL_DIR = DEFAULT_DATA_DIR / "fourier_inpainting_nn_unet"
OUT_FILE = MODEL_DIR / "best_model_inpainting_example_panels.png"


def to_adu(arr: np.ndarray, offset: float, scale: float) -> np.ndarray:
    return arr * scale + offset


def main() -> None:
    with (MODEL_DIR / "summary.json").open() as f:
        summary = json.load(f)
    cfg = Config(**summary["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_DIR / "fourier_inpainting_net.pt", map_location=device, weights_only=False)
    model = FourierInpaintNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_blank_background(DEFAULT_MEAN_DIR)
    dataset = FourierPatchDataset(image, 160, cfg, cfg.seed + 5000)

    candidates = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"][None].to(device)
            target = sample["target"][None].to(device)
            filled = sample["filled"][None].to(device)
            hidden = sample["hidden"][None].to(device)
            scale = sample["scale"][None].to(device)
            pred = ifft_from_channels(model(x))[:, None]
            nn_pix, nn_mean = masked_rmse(pred, target, hidden, scale)
            butter = torch.from_numpy(butterworth_lowpass(sample["filled"][0].numpy(), cfg)[None, None]).to(device)
            butter_pix, butter_mean = masked_rmse(butter, target, hidden, scale)
            candidates.append((butter_pix - nn_pix, idx, nn_pix, nn_mean, butter_pix, butter_mean))

    # Show good but not cherry-picked examples: take evenly spaced examples from
    # the better half of cases where the NN improves on the Butterworth baseline.
    improved = [c for c in sorted(candidates, reverse=True) if c[0] > 0]
    if len(improved) < 6:
        improved = sorted(candidates, reverse=True)
    chosen = [improved[i] for i in np.linspace(0, len(improved) - 1, 6, dtype=int)]

    fig, axes = plt.subplots(len(chosen), 6, figsize=(15, 14), constrained_layout=True)
    titles = ["true blank", "masked input", "NN infill", "Butterworth", "NN error", "Butterworth error"]

    with torch.no_grad():
        for row, (_, idx, nn_pix, nn_mean, butter_pix, butter_mean) in enumerate(chosen):
            sample = dataset[idx]
            x = sample["x"][None].to(device)
            pred = ifft_from_channels(model(x))[0].cpu().numpy()
            target = sample["target"][0].numpy()
            filled = sample["filled"][0].numpy()
            hidden = sample["hidden"][0].numpy().astype(bool)
            butter = butterworth_lowpass(filled, cfg)
            offset = float(sample["offset"])
            scale = float(sample["scale"])

            target_adu = to_adu(target, offset, scale)
            filled_adu = to_adu(filled, offset, scale)
            pred_adu = to_adu(pred, offset, scale)
            butter_adu = to_adu(butter, offset, scale)
            nn_err = np.where(hidden, pred_adu - target_adu, np.nan)
            butter_err = np.where(hidden, butter_adu - target_adu, np.nan)

            vmin, vmax = np.percentile(target_adu, [1, 99])
            err_lim = max(
                1.0,
                float(np.nanpercentile(np.abs(nn_err), 98)),
                float(np.nanpercentile(np.abs(butter_err), 98)),
            )
            panels = [target_adu, filled_adu, pred_adu, butter_adu, nn_err, butter_err]
            for col, panel in enumerate(panels):
                ax = axes[row, col]
                if col < 4:
                    shown = panel.copy()
                    if col == 1:
                        shown = np.where(hidden, np.nan, shown)
                    ax.imshow(shown, cmap="gray", vmin=vmin, vmax=vmax)
                    ax.contour(hidden, levels=[0.5], colors="tab:red", linewidths=1.0)
                else:
                    ax.imshow(panel, cmap="coolwarm", vmin=-err_lim, vmax=err_lim)
                    ax.contour(hidden, levels=[0.5], colors="black", linewidths=0.8)
                if row == 0:
                    ax.set_title(titles[col], fontsize=13)
                if col == 0:
                    ax.set_ylabel(
                        f"example {row + 1}\nNN {nn_pix:.1f} ADU\nBW {butter_pix:.1f} ADU",
                        fontsize=10,
                    )
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle(
        "Fourier neural-network inpainting of synthetic rod-sized masks on blank coverslip background",
        fontsize=15,
    )
    fig.savefig(OUT_FILE, dpi=220)
    plt.close(fig)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
