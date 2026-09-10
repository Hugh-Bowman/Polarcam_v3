from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val background .npy images from stack files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing .npy stacks (T,H,W).")
    parser.add_argument("--out-root", type=Path, required=True, help="Project root with data/train_backgrounds and data/val_backgrounds.")
    parser.add_argument("--mode", type=str, choices=["mean", "per_frame"], default="mean")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-frames-per-stack", type=int, default=100)
    args = parser.parse_args()

    train_dir = args.out_root / "data" / "train_backgrounds"
    val_dir = args.out_root / "data" / "val_backgrounds"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stack_files = sorted(args.input_dir.glob("*.npy"))
    if not stack_files:
        raise RuntimeError(f"No .npy files in {args.input_dir}")

    rng = np.random.default_rng(42)
    written = 0

    for p in stack_files:
        arr = np.load(p)
        if arr.ndim == 2:
            items = [arr.astype(np.float32)]
            names = [f"{p.stem}_img"]
        elif arr.ndim == 3:
            if args.mode == "mean":
                items = [arr.mean(axis=0).astype(np.float32)]
                names = [f"{p.stem}_mean"]
            else:
                t = min(arr.shape[0], args.max_frames_per_stack)
                pick = rng.choice(arr.shape[0], size=t, replace=False)
                items = [arr[i].astype(np.float32) for i in pick]
                names = [f"{p.stem}_f{i:04d}" for i in pick]
        else:
            continue

        for img, n in zip(items, names):
            dst_dir = val_dir if rng.random() < args.val_fraction else train_dir
            np.save(dst_dir / f"{n}.npy", img)
            written += 1

    print(f"[done] wrote {written} background images")
    print(f"train_dir={train_dir}")
    print(f"val_dir={val_dir}")


if __name__ == "__main__":
    main()
