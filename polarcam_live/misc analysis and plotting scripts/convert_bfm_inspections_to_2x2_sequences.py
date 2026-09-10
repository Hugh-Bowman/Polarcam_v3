from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "tumbling 25nm glycerol" / "pending"
OUTPUT_DIR = SOURCE_DIR / "2x2_channel_sequences_from_15x256"


def is_recent_inspection_stack(path: Path) -> bool:
    if path.name != "capture_maxfps_15x15.npy":
        return False
    if "20260715-16" not in str(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return False
    return arr.ndim == 3 and int(arr.shape[1]) in (14, 15) and int(arr.shape[2]) == 256


def crop_inspection_square(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {gray.shape}")
    h, w = gray.shape
    side = min(int(h), int(w))
    return np.asarray(gray[:side, :side])


def frame_to_2x2_means(gray: np.ndarray) -> np.ndarray:
    crop = crop_inspection_square(gray)
    # Same pixel labels as angle_distribution_analysis.py:
    # I0=gray[0::2,0::2], I45=gray[0::2,1::2],
    # I135=gray[1::2,0::2], I90=gray[1::2,1::2].
    i0 = crop[0::2, 0::2]
    i45 = crop[0::2, 1::2]
    i135 = crop[1::2, 0::2]
    i90 = crop[1::2, 1::2]
    out = np.empty((2, 2), dtype=np.float32)
    out[0, 0] = float(np.mean(i0)) if i0.size else np.nan
    out[0, 1] = float(np.mean(i45)) if i45.size else np.nan
    out[1, 0] = float(np.mean(i135)) if i135.size else np.nan
    out[1, 1] = float(np.mean(i90)) if i90.size else np.nan
    return out


def convert_stack(path: Path, output_path: Path) -> dict:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"{path} is not a 3D stack: shape={arr.shape}")
    n_frames = int(arr.shape[0])
    seq = np.empty((2, 2, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seq[:, :, i] = frame_to_2x2_means(np.asarray(arr[i]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, seq, allow_pickle=False)
    return {
        "source": str(path),
        "output": str(output_path),
        "source_shape": tuple(int(x) for x in arr.shape),
        "output_shape": tuple(int(x) for x in seq.shape),
        "output_dtype": str(seq.dtype),
        "layout": "[[I0, I45], [I135, I90]]",
        "crop": "first min(height,width) x min(height,width) square, matching angle_distribution_analysis inspection crop",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in SOURCE_DIR.glob("rod_*/capture_maxfps_15x15.npy") if is_recent_inspection_stack(p))
    if not paths:
        raise FileNotFoundError(f"No recent 15x256 inspection stacks found under {SOURCE_DIR}")

    rows = []
    for path in paths:
        rod_name = path.parent.name
        output_path = OUTPUT_DIR / f"{rod_name}_2x2_channel_sequence.npy"
        print(f"Converting {path.parent.name}")
        rows.append(convert_stack(path, output_path))

    summary_csv = OUTPUT_DIR / "conversion_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "output", "source_shape", "output_shape", "output_dtype", "layout", "crop"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json = OUTPUT_DIR / "conversion_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    readme = [
        "2x2 channel sequences from BFM/inspection npy stacks",
        "",
        f"Source folder: {SOURCE_DIR}",
        f"Output folder: {OUTPUT_DIR}",
        "",
        "Inputs converted:",
        "- capture_maxfps_15x15.npy files with shape (frames, 14, 256) under recent 20260715-16 rod folders.",
        "",
        "Output format:",
        "- One .npy per source recording.",
        "- Shape is (2, 2, n_frames), as requested.",
        "- dtype is float32 because each value is an average over same-labelled pixels.",
        "- Pixel-label layout is [[I0, I45], [I135, I90]].",
        "- The input frame is cropped to the first 14x14 block before averaging, matching angle_distribution_analysis.py for padded inspection stacks.",
    ]
    (OUTPUT_DIR / "README_2x2_channel_sequences.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"Converted {len(rows)} stacks")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
