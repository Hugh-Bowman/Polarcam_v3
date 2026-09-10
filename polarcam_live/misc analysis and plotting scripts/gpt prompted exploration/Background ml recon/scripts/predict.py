from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from background_ml_recon.dataset import load_single_image
from background_ml_recon.model import build_model
from background_ml_recon.utils import save_preview


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        m = np.load(path).astype(np.float32)
    else:
        m = io.imread(path).astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
    if m.shape != expected_shape:
        raise RuntimeError(f"Mask shape {m.shape} does not match image shape {expected_shape}.")
    return (m > 0.5).astype(np.float32)


def _canonicalize_hole_mask(mask: np.ndarray) -> np.ndarray:
    hole_mask = (mask > 0.5).astype(np.float32)
    if float(np.mean(hole_mask)) > 0.5:
        hole_mask = 1.0 - hole_mask
    return hole_mask.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run background reconstruction on one masked image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Input microscopy image.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask where 1=missing region.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize-percentile", type=float, default=99.5)
    args = parser.parse_args()

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt["config"]["model"]
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image = load_single_image(args.image).astype(np.float32)
    if image.ndim != 2:
        raise RuntimeError("Expected grayscale 2D image.")
    scale = float(np.percentile(image, args.normalize_percentile))
    scale = max(scale, 1e-6)
    image_n = image / scale

    mask_raw = _load_mask(args.mask, image_n.shape)
    hole_mask = _canonicalize_hole_mask(mask_raw)
    masked = image_n * (1.0 - hole_mask)
    inp = np.stack([masked, hole_mask], axis=0)[None, ...]
    inp_t = torch.from_numpy(inp.astype(np.float32)).to(device)

    with torch.no_grad():
        pred_delta_n = model(inp_t)[0, 0].cpu().numpy().astype(np.float32)
    pred_n = masked + hole_mask * pred_delta_n

    pred = pred_n * scale
    reconstructed = image.copy()
    reconstructed[hole_mask > 0.5] = pred[hole_mask > 0.5]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    out_pred = args.out_dir / f"{stem}_pred_full_background.npy"
    out_recon = args.out_dir / f"{stem}_reconstructed.npy"
    np.save(out_pred, pred.astype(np.float32))
    np.save(out_recon, reconstructed.astype(np.float32))

    save_preview(
        args.out_dir / f"{stem}_preview.png",
        masked=masked,
        mask=hole_mask,
        pred=pred_n,
        target=image_n,
    )
    print(f"saved: {out_pred}")
    print(f"saved: {out_recon}")
    print(f"saved: {args.out_dir / f'{stem}_preview.png'}")


if __name__ == "__main__":
    main()
