"""
Train a compact Fourier-domain inpainting network from the blank coverslip data.

The training task is synthetic: circular rod-sized regions are masked out of the
blank coverslip background, and the network learns to reconstruct the hidden
background from the surrounding field using Fourier coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "recordings" / "2026-06-19" / "background analsysis 190626"
DEFAULT_MEAN_DIR = DEFAULT_DATA_DIR / "background_noise_analysis_marker_stripped"


# Edit these when running the file directly from an IDE.
RUN_FROM_IDE = True
IDE_MEAN_DIR = DEFAULT_MEAN_DIR
IDE_OUT_DIR = DEFAULT_DATA_DIR / "fourier_inpainting_nn_unet"


@dataclass
class Config:
    patch_size: int = 256
    mask_radius_px: int = 12
    train_patches: int = 3000
    val_patches: int = 600
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    early_stop_patience: int = 5
    seed: int = 7
    pixel_um: float = 0.5 * (140.0 / 2464.0 + 120.0 / 2056.0)
    butterworth_lambda_um: float = 1.4
    butterworth_order: int = 4
    ring_outer_radius_px: int = 12
    dropout: float = 0.0
    hidden_mse_weight: float = 1.0
    hidden_smooth_l1_weight: float = 0.25
    full_mse_weight: float = 0.0
    mean_loss_weight: float = 0.05
    fft_loss_weight: float = 1.0e-4
    clip_grad_norm: float = 1.0


# These are the defaults used when you click "Run" in the IDE.
IDE_CONFIG = Config(
    patch_size=256,
    mask_radius_px=12,
    train_patches=3000,
    val_patches=600,
    epochs=20,
    batch_size=8,
    learning_rate=1.0e-3,
    weight_decay=1.0e-5,
    early_stop_patience=5,
    seed=7,
    ring_outer_radius_px=12,
    dropout=0.0,
    hidden_mse_weight=1.0,
    hidden_smooth_l1_weight=0.25,
    full_mse_weight=0.0,
    mean_loss_weight=0.05,
    fft_loss_weight=1.0e-4,
    clip_grad_norm=1.0,
)


def load_blank_background(mean_dir: Path) -> np.ndarray:
    blank = np.load(mean_dir / "blank_laser_on_mean_all150_marker_stripped.npy").astype(np.float32)
    dark = np.load(mean_dir / "laser_off_mean_dark_frame_marker_stripped.npy").astype(np.float32)
    return blank - dark


def circular_mask(size: int, radius: int, cy: int, cx: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def annulus_mask(size: int, inner: int, outer: int, cy: int, cx: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (rr2 > inner**2) & (rr2 <= outer**2)


def fft_channels(image: np.ndarray) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)


def fourier_coordinate_channels(size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2).astype(np.float32)
    return np.stack([xx, yy, rr], axis=0)


def ifft_from_channels(channels: torch.Tensor) -> torch.Tensor:
    spec = torch.complex(channels[:, 0], channels[:, 1])
    spec = torch.fft.ifftshift(spec, dim=(-2, -1))
    return torch.fft.ifft2(spec, norm="ortho").real


class FourierPatchDataset(Dataset):
    def __init__(self, image: np.ndarray, n_patches: int, cfg: Config, seed: int) -> None:
        self.image = image
        self.n_patches = n_patches
        self.cfg = cfg
        self.coord_channels = fourier_coordinate_channels(cfg.patch_size)
        rng = np.random.default_rng(seed)
        h, w = image.shape
        p = cfg.patch_size
        margin = cfg.mask_radius_px + 6
        self.samples: list[tuple[int, int, int, int]] = []
        for _ in range(n_patches):
            y0 = int(rng.integers(0, h - p))
            x0 = int(rng.integers(0, w - p))
            cy = int(rng.integers(margin, p - margin))
            cx = int(rng.integers(margin, p - margin))
            self.samples.append((y0, x0, cy, cx))

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        y0, x0, cy, cx = self.samples[idx]
        p = self.cfg.patch_size
        r = self.cfg.mask_radius_px
        patch = self.image[y0 : y0 + p, x0 : x0 + p].copy()
        hidden = circular_mask(p, r, cy, cx)
        ring = annulus_mask(p, r + 1, r + int(self.cfg.ring_outer_radius_px), cy, cx)
        known = ~hidden

        offset = float(np.mean(patch[known]))
        scale = float(np.std(patch[known]))
        if scale < 1.0:
            scale = 1.0

        target = (patch - offset) / scale
        filled = target.copy()
        fill_value = float(np.mean(target[ring])) if np.any(ring) else 0.0
        filled[hidden] = fill_value

        hidden_f = hidden.astype(np.float32)
        known_f = known.astype(np.float32)
        # Fourier-domain input: masked image spectrum, known-pixel support
        # spectrum, and explicit Fourier-coordinate channels. The coordinate
        # channels are important because low and high spatial frequencies are
        # physically different but otherwise look translation-equivalent to a CNN.
        x = np.concatenate([fft_channels(filled), fft_channels(known_f), self.coord_channels], axis=0)
        y = fft_channels(target)
        return {
            "x": torch.from_numpy(x),
            "y_fft": torch.from_numpy(y),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "filled": torch.from_numpy(filled[None].astype(np.float32)),
            "hidden": torch.from_numpy(hidden_f[None]),
            "offset": torch.tensor(offset, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.04) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FourierInpaintNet(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        # A small U-Net in shifted Fourier-coordinate space. It can model both
        # broad low-frequency structure and sharper local Fourier corrections.
        self.enc1 = ConvBlock(7, 32, dropout=dropout)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(64, 64, dropout=dropout)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.mid = ConvBlock(128, 128, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(128, 64, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(64, 32, dropout=dropout)
        self.out = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual prediction around the filled-patch Fourier spectrum.
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return x[:, :2] + self.out(d1)


def compose_prediction(pred_raw: torch.Tensor, filled: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    # Known pixels pass through exactly; the network only needs to improve the hole.
    return filled * (1.0 - hidden) + pred_raw * hidden


def butterworth_lowpass(image: np.ndarray, cfg: Config) -> np.ndarray:
    p = image.shape[-1]
    fy = np.fft.fftfreq(p, d=cfg.pixel_um)
    fx = np.fft.fftfreq(p, d=cfg.pixel_um)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    cutoff = 1.0 / cfg.butterworth_lambda_um
    filt = 1.0 / (1.0 + (radius / cutoff) ** (2 * cfg.butterworth_order))
    spec = np.fft.fft2(image, norm="ortho")
    return np.fft.ifft2(spec * filt, norm="ortho").real.astype(np.float32)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor, scale: torch.Tensor) -> tuple[float, float]:
    err = (pred - target) * hidden
    pix = torch.sqrt((err.square().sum(dim=(1, 2, 3)) / hidden.sum(dim=(1, 2, 3)).clamp_min(1.0))) * scale
    pred_mean = (pred * hidden).sum(dim=(1, 2, 3)) / hidden.sum(dim=(1, 2, 3)).clamp_min(1.0)
    targ_mean = (target * hidden).sum(dim=(1, 2, 3)) / hidden.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_err = torch.abs(pred_mean - targ_mean) * scale
    return float(pix.mean().cpu()), float(torch.sqrt((mean_err.square()).mean()).cpu())


def evaluate(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device) -> dict[str, float]:
    model.eval()
    pix_vals: list[float] = []
    mean_vals: list[float] = []
    fill_pix_vals: list[float] = []
    fill_mean_vals: list[float] = []
    butter_pix_vals: list[float] = []
    butter_mean_vals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            hidden = batch["hidden"].to(device)
            filled = batch["filled"].to(device)
            scale = batch["scale"].to(device)
            pred_raw = ifft_from_channels(model(x))[:, None]
            pred = compose_prediction(pred_raw, filled, hidden)
            pix, mean = masked_rmse(pred, target, hidden, scale)
            pix_vals.append(pix)
            mean_vals.append(mean)
            fill_pix, fill_mean = masked_rmse(filled, target, hidden, scale)
            fill_pix_vals.append(fill_pix)
            fill_mean_vals.append(fill_mean)

            butter_preds = []
            for filled in batch["filled"].numpy()[:, 0]:
                butter_preds.append(butterworth_lowpass(filled, cfg)[None])
            butter = torch.from_numpy(np.stack(butter_preds)).to(device)
            butter_pix, butter_mean = masked_rmse(butter, target, hidden, scale)
            butter_pix_vals.append(butter_pix)
            butter_mean_vals.append(butter_mean)
    return {
        "nn_pixel_rmse_adu": float(np.mean(pix_vals)),
        "nn_mask_mean_rmse_adu": float(np.mean(mean_vals)),
        "annulus_fill_pixel_rmse_adu": float(np.mean(fill_pix_vals)),
        "annulus_fill_mask_mean_rmse_adu": float(np.mean(fill_mean_vals)),
        "butterworth_pixel_rmse_adu": float(np.mean(butter_pix_vals)),
        "butterworth_mask_mean_rmse_adu": float(np.mean(butter_mean_vals)),
    }


def evaluate_nn_only(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    pix_vals: list[float] = []
    mean_vals: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            hidden = batch["hidden"].to(device)
            filled = batch["filled"].to(device)
            scale = batch["scale"].to(device)
            pred_raw = ifft_from_channels(model(x))[:, None]
            pred = compose_prediction(pred_raw, filled, hidden)
            hidden_loss = ((((pred - target) ** 2) * hidden).sum() / hidden.sum().clamp_min(1.0))
            pix, mean = masked_rmse(pred, target, hidden, scale)
            losses.append(float(hidden_loss.cpu()))
            pix_vals.append(pix)
            mean_vals.append(mean)
    return {
        "val_loss": float(np.mean(losses)),
        "val_pixel_rmse_adu": float(np.mean(pix_vals)),
        "val_mask_mean_rmse_adu": float(np.mean(mean_vals)),
    }


def save_examples(
    model: nn.Module,
    dataset: Dataset,
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    stem: str,
    split_label: str,
) -> None:
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, 6, dtype=int)
    fig, axes = plt.subplots(len(indices), 5, figsize=(12, 13), constrained_layout=True)
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[int(idx)]
            x = sample["x"][None].to(device)
            pred_raw = ifft_from_channels(model(x))[0:1]
            target = sample["target"][0].numpy()
            filled = sample["filled"][0].numpy()
            mask = sample["hidden"][0].numpy().astype(bool)
            pred = compose_prediction(
                pred_raw[:, None],
                sample["filled"][None].to(device),
                sample["hidden"][None].to(device),
            )[0, 0].cpu().numpy()
            butter = butterworth_lowpass(filled, cfg)
            vmin, vmax = np.percentile(target, [1, 99])
            panels = [
                ("target", target),
                ("masked input", filled),
                ("NN", pred),
                ("Butterworth", butter),
                ("NN error", np.where(mask, pred - target, np.nan)),
            ]
            for col, (title, arr) in enumerate(panels):
                ax = axes[row, col]
                if col == 4:
                    lim = np.nanpercentile(np.abs(arr), 98)
                    ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim)
                else:
                    shown = arr.copy()
                    if col == 1:
                        shown = np.where(mask, np.nan, shown)
                    ax.imshow(shown, cmap="gray", vmin=vmin, vmax=vmax)
                    ax.contour(mask, levels=[0.5], colors="tab:red", linewidths=0.8)
                if row == 0:
                    ax.set_title(title, fontsize=12)
                ax.set_axis_off()
    fig.suptitle(f"Best-model Fourier inpainting examples: {split_label}", fontsize=14)
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    plt.close(fig)


def plot_history(rows: list[dict[str, float]], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train loss")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], marker="o", label="validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("hidden-region MSE loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_pixel_rmse_adu"] for r in rows], marker="o", label="pixel RMSE")
    axes[1].plot(epochs, [r["val_mask_mean_rmse_adu"] for r in rows], marker="o", label="mask mean RMSE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation RMSE / ADU")
    axes[1].legend()
    fig.savefig(out_dir / "fourier_nn_training_history.png", dpi=200)
    plt.close(fig)


def train(cfg: Config, mean_dir: Path, out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = load_blank_background(mean_dir)
    train_ds = FourierPatchDataset(image, cfg.train_patches, cfg, cfg.seed)
    val_ds = FourierPatchDataset(image, cfg.val_patches, cfg, cfg.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = FourierInpaintNet(dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    history: list[dict[str, float]] = []
    best_val = math.inf

    print(f"Training FourierInpaintNet on {device}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Mean directory: {mean_dir}", flush=True)
    print(f"Config: {json.dumps(asdict(cfg), indent=2)}", flush=True)
    print("epoch | lr | train_loss | val_loss | val_pixel_RMSE_ADU | val_mask_mean_RMSE_ADU | best", flush=True)
    epochs_since_best = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, total=len(train_loader), desc=f"train epoch {epoch:03d}/{cfg.epochs:03d}", dynamic_ncols=True)
        for batch in progress:
            x = batch["x"].to(device)
            y_fft = batch["y_fft"].to(device)
            target = batch["target"].to(device)
            hidden = batch["hidden"].to(device)
            filled = batch["filled"].to(device)
            pred_fft = model(x)
            pred_raw = ifft_from_channels(pred_fft)[:, None]
            pred = compose_prediction(pred_raw, filled, hidden)
            hidden_loss = (((pred - target) ** 2) * hidden).sum() / hidden.sum().clamp_min(1.0)
            full_loss = torch.mean((pred - target) ** 2)
            mean_pred = (pred * hidden).sum(dim=(1, 2, 3)) / hidden.sum(dim=(1, 2, 3)).clamp_min(1.0)
            mean_target = (target * hidden).sum(dim=(1, 2, 3)) / hidden.sum(dim=(1, 2, 3)).clamp_min(1.0)
            mean_loss = torch.mean((mean_pred - mean_target) ** 2)
            fft_loss = torch.mean((pred_fft - y_fft) ** 2)
            smooth_l1_hidden = F.smooth_l1_loss(pred * hidden, target * hidden, beta=0.25)
            loss = (
                cfg.hidden_mse_weight * hidden_loss
                + cfg.hidden_smooth_l1_weight * smooth_l1_hidden
                + cfg.full_mse_weight * full_loss
                + cfg.mean_loss_weight * mean_loss
                + cfg.fft_loss_weight * fft_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{float(np.mean(losses)):.5f}")
        progress.close()

        val_stats = evaluate_nn_only(model, val_loader, device)
        scheduler.step(val_stats["val_loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **val_stats,
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "mean_dir": str(mean_dir),
                    "description": "Fourier-domain masked background inpainting network",
                },
                out_dir / "fourier_inpainting_net.pt",
            )
            best_mark = "*"
            epochs_since_best = 0
        else:
            best_mark = ""
            epochs_since_best += 1
        lr = opt.param_groups[0]["lr"]
        print(
            f"{epoch:05d} | {lr:.2e} | {row['train_loss']:.5f} | {row['val_loss']:.5f} | "
            f"{row['val_pixel_rmse_adu']:.3f} | {row['val_mask_mean_rmse_adu']:.3f} | {best_mark}",
            flush=True,
        )
        if epochs_since_best >= cfg.early_stop_patience:
            print(f"Early stopping: no validation improvement for {cfg.early_stop_patience} epochs.", flush=True)
            break

    checkpoint = torch.load(out_dir / "fourier_inpainting_net.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, val_loader, cfg, device)
    metrics.update({"device": str(device), "best_val_loss": best_val})

    with (out_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_pixel_rmse_adu",
                "val_mask_mean_rmse_adu",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2)
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    plot_history(history, out_dir)
    save_examples(
        model,
        train_ds,
        cfg,
        device,
        out_dir,
        stem="best_model_inpainting_examples_train",
        split_label="train set",
    )
    save_examples(
        model,
        val_ds,
        cfg,
        device,
        out_dir,
        stem="best_model_inpainting_examples_val",
        split_label="validation set",
    )
    return metrics


def build_cli_config() -> tuple[Config, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-dir", type=Path, default=DEFAULT_MEAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "fourier_inpainting_nn_unet")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--train-patches", type=int, default=Config.train_patches)
    parser.add_argument("--val-patches", type=int, default=Config.val_patches)
    parser.add_argument("--patch-size", type=int, default=Config.patch_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        train_patches=args.train_patches,
        val_patches=args.val_patches,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    return cfg, args.mean_dir, args.out_dir


def main() -> None:
    if RUN_FROM_IDE:
        cfg = IDE_CONFIG
        mean_dir = IDE_MEAN_DIR
        out_dir = IDE_OUT_DIR
    else:
        cfg, mean_dir, out_dir = build_cli_config()
    metrics = train(cfg, mean_dir, out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
