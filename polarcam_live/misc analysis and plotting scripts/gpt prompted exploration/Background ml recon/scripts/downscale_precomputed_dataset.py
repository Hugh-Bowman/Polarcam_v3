from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def mean2x2(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    h, w = arr.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    if h2 != h or w2 != w:
        arr = arr[:h2, :w2]
    return arr.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def build_triplets(target_dir: Path, masked_dir: Path, mask_dir: Path) -> list[tuple[Path, Path, Path]]:
    out: list[tuple[Path, Path, Path]] = []
    for t in sorted(target_dir.glob("*.npy")):
        stem = t.stem
        mi = masked_dir / f"{stem}_masked.npy"
        m = mask_dir / f"{stem}_mask.npy"
        if mi.exists() and m.exists():
            out.append((t, mi, m))
    return out


def convert_split(
    src_target: Path,
    src_masked: Path,
    src_mask: Path,
    dst_target: Path,
    dst_masked: Path,
    dst_mask: Path,
) -> tuple[int, tuple[int, int] | None]:
    dst_target.mkdir(parents=True, exist_ok=True)
    dst_masked.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    triplets = build_triplets(src_target, src_masked, src_mask)
    n = 0
    out_shape = None

    for t_path, _mi_path, m_path in triplets:
        stem = t_path.stem

        target = np.load(t_path).astype(np.float32)
        mask = (np.load(m_path).astype(np.float32) > 0.5).astype(np.float32)

        target_ds = mean2x2(target)
        mask_ds_mean = mean2x2(mask)
        mask_ds = (mask_ds_mean >= 0.5).astype(np.float32)
        masked_ds = target_ds * (1.0 - mask_ds)

        np.save(dst_target / f"{stem}.npy", target_ds)
        np.save(dst_mask / f"{stem}_mask.npy", mask_ds)
        np.save(dst_masked / f"{stem}_masked.npy", masked_ds)

        out_shape = target_ds.shape
        n += 1

    return n, out_shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Downscale precomputed masked dataset by 2x2 block averaging.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--src-train-target", type=Path, default=Path("data/train_backgrounds"))
    parser.add_argument("--src-train-masked", type=Path, default=Path("data/train_masked_inputs"))
    parser.add_argument("--src-train-mask", type=Path, default=Path("data/train_masks"))
    parser.add_argument("--src-val-target", type=Path, default=Path("data/val_backgrounds"))
    parser.add_argument("--src-val-masked", type=Path, default=Path("data/val_masked_inputs"))
    parser.add_argument("--src-val-mask", type=Path, default=Path("data/val_masks"))

    parser.add_argument("--dst-train-target", type=Path, default=Path("data/train_backgrounds_ds2"))
    parser.add_argument("--dst-train-masked", type=Path, default=Path("data/train_masked_inputs_ds2"))
    parser.add_argument("--dst-train-mask", type=Path, default=Path("data/train_masks_ds2"))
    parser.add_argument("--dst-val-target", type=Path, default=Path("data/val_backgrounds_ds2"))
    parser.add_argument("--dst-val-masked", type=Path, default=Path("data/val_masked_inputs_ds2"))
    parser.add_argument("--dst-val-mask", type=Path, default=Path("data/val_masks_ds2"))
    args = parser.parse_args()

    root = args.root

    n_train, shape_train = convert_split(
        root / args.src_train_target,
        root / args.src_train_masked,
        root / args.src_train_mask,
        root / args.dst_train_target,
        root / args.dst_train_masked,
        root / args.dst_train_mask,
    )

    n_val, shape_val = convert_split(
        root / args.src_val_target,
        root / args.src_val_masked,
        root / args.src_val_mask,
        root / args.dst_val_target,
        root / args.dst_val_masked,
        root / args.dst_val_mask,
    )

    print(f"[done] train triplets downscaled: {n_train}")
    print(f"[done] val triplets downscaled:   {n_val}")
    print(f"[done] train output shape: {shape_train}")
    print(f"[done] val output shape:   {shape_val}")


if __name__ == "__main__":
    main()
