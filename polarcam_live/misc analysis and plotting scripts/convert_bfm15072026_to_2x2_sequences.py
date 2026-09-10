from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "bfm15072026"
OUTPUT_DIR = SOURCE_DIR / "2x2_channel_sequences"
SPOT_WINDOW_SIZE = 19


def inspection_intensity_bounds(n: int) -> tuple[int, int, int, int]:
    # Direct copy of the angle_distribution_analysis.py convention.
    win = max(1, int(round(SPOT_WINDOW_SIZE / 2.0)))
    if win % 2 == 0:
        win += 1
    half = win // 2
    ih = max(1, int(n) // 2)
    iw = max(1, int(n) // 2)
    cx = int(round((0.5 * (n - 1)) / 2.0))
    cy = int(round((0.5 * (n - 1)) / 2.0))
    x0 = max(0, cx - half)
    x1 = min(iw, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(ih, cy + half + 1)
    return (x0, x1, y0, y1)


def reduce_stack_to_2x2_sequence(path: Path) -> tuple[np.ndarray, dict]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected 3D stack, got {arr.shape}")

    n_frames = int(arr.shape[0])
    side = min(int(arr.shape[1]), int(arr.shape[2]))
    crop = np.asarray(arr[:, :side, :side], dtype=np.float32)

    i0 = crop[:, 0::2, 0::2]
    i45 = crop[:, 0::2, 1::2]
    i135 = crop[:, 1::2, 0::2]
    i90 = crop[:, 1::2, 1::2]

    x0, x1, y0, y1 = inspection_intensity_bounds(side)
    a0 = i0[:, y0:y1, x0:x1]
    a45 = i45[:, y0:y1, x0:x1]
    a135 = i135[:, y0:y1, x0:x1]
    a90 = i90[:, y0:y1, x0:x1]

    seq = np.empty((2, 2, n_frames), dtype=np.float32)
    seq[0, 0, :] = np.mean(a0, axis=(1, 2), dtype=np.float64)
    seq[0, 1, :] = np.mean(a45, axis=(1, 2), dtype=np.float64)
    seq[1, 0, :] = np.mean(a135, axis=(1, 2), dtype=np.float64)
    seq[1, 1, :] = np.mean(a90, axis=(1, 2), dtype=np.float64)

    meta = {
        "source_shape": tuple(int(x) for x in arr.shape),
        "output_shape": tuple(int(x) for x in seq.shape),
        "crop_side": int(side),
        "channel_bounds_xy": [int(x0), int(x1), int(y0), int(y1)],
        "layout": "[[I0, I45], [I135, I90]]",
        "spot_window_size": int(SPOT_WINDOW_SIZE),
    }
    return seq, meta


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE_DIR.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {SOURCE_DIR}")

    rows = []
    for path in paths:
        print(f"Converting {path.name}")
        seq, meta = reduce_stack_to_2x2_sequence(path)
        out_path = OUTPUT_DIR / f"{path.stem}_2x2_channel_sequence.npy"
        np.save(out_path, seq, allow_pickle=False)
        rows.append(
            {
                "source": str(path),
                "output": str(out_path),
                "source_shape": meta["source_shape"],
                "output_shape": meta["output_shape"],
                "output_dtype": str(seq.dtype),
                "crop_side": meta["crop_side"],
                "channel_bounds_xy": meta["channel_bounds_xy"],
                "layout": meta["layout"],
                "spot_window_size": meta["spot_window_size"],
            }
        )

    with (OUTPUT_DIR / "conversion_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "output",
            "source_shape",
            "output_shape",
            "output_dtype",
            "crop_side",
            "channel_bounds_xy",
            "layout",
            "spot_window_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_DIR / "conversion_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    readme = [
        "BFM 2x2 channel sequences",
        "",
        f"Source folder: {SOURCE_DIR}",
        f"Output folder: {OUTPUT_DIR}",
        "",
        "Each source spotrec_*.npy was converted to one 2x2 channel sequence.",
        "Output shape is (2, 2, n_frames).",
        "Output dtype is float32 because values are averaged channel intensities.",
        "Pixel-label layout is [[I0, I45], [I135, I90]].",
        "The input frame is cropped to the first min(height,width) square, matching angle_distribution_analysis.py for padded inspection stacks.",
        f"Angle-analysis spot_window_size used for bounds: {SPOT_WINDOW_SIZE}.",
    ]
    (OUTPUT_DIR / "README_2x2_channel_sequences.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"Converted {len(rows)} files")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
