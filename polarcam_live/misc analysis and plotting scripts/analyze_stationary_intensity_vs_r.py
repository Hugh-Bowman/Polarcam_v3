from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_DIR = Path("stationary rods 25nm 02072026") / "pending"
OUTPUT_DIR = Path("stationary rods 25nm 02072026") / "plots" / "intensity_vs_r"
BACKGROUND_PROFILE_PATH = Path("background_profile.npy")


@dataclass
class RodPoint:
    rod_id: str
    folder: str
    timestamp: datetime
    set_index: int
    r_mean: float
    range_x: float
    range_y: float
    intensity_p98: float
    max_raw_value: float
    max_saved_value: float
    subtraction_drop: float


def _parse_timestamp(folder_name: str) -> datetime:
    parts = folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Could not parse timestamp from {folder_name}")
    stamp = parts[-2]
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S")


def _load_points() -> list[RodPoint]:
    meta_paths = sorted(DATASET_DIR.rglob("meta.json"))
    rows: list[tuple[datetime, Path, dict, dict]] = []
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text())
        summary = dict(meta.get("modes", {}).get("capture_maxfps_15x15_summary", {}) or {})
        actual_rel = meta.get("modes", {}).get("capture_maxfps_15x15_meta")
        if not actual_rel:
            continue
        actual = json.loads((meta_path.parent / actual_rel).read_text()).get("actual", {})
        ts = _parse_timestamp(meta_path.parent.name)
        rows.append((ts, meta_path, summary, actual))

    rows.sort(key=lambda x: x[0])
    points: list[RodPoint] = []
    for i, (ts, meta_path, summary, actual) in enumerate(rows):
        mr = float(actual.get("max_raw_value", np.nan))
        ms = float(actual.get("max_saved_value", np.nan))
        points.append(
            RodPoint(
                rod_id=str(json.loads(meta_path.read_text()).get("rod_id", meta_path.parent.name)),
                folder=meta_path.parent.name,
                timestamp=ts,
                set_index=0,
                r_mean=float(summary.get("r_mean", np.nan)),
                range_x=float(summary.get("range_x", np.nan)),
                range_y=float(summary.get("range_y", np.nan)),
                intensity_p98=float(actual.get("intensity_p98", np.nan)),
                max_raw_value=mr,
                max_saved_value=ms,
                subtraction_drop=mr - ms,
            )
        )
    return points


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = _load_points()
    if not points:
        raise RuntimeError(f"No stationary-rod metadata found under {DATASET_DIR}")

    df = pd.DataFrame(
        {
            "rod_id": [p.rod_id for p in points],
            "folder": [p.folder for p in points],
            "timestamp": [p.timestamp.isoformat(sep=" ") for p in points],
            "set_index": [p.set_index for p in points],
            "r_mean": [p.r_mean for p in points],
            "range_x": [p.range_x for p in points],
            "range_y": [p.range_y for p in points],
            "intensity_p98": [p.intensity_p98 for p in points],
            "max_raw_value": [p.max_raw_value for p in points],
            "max_saved_value": [p.max_saved_value for p in points],
            "subtraction_drop": [p.subtraction_drop for p in points],
        }
    )
    df.to_csv(OUTPUT_DIR / "intensity_vs_mean_r_points.csv", index=False)

    verify = {
        "background_profile_path": str(BACKGROUND_PROFILE_PATH.resolve()),
        "background_profile_last_modified": datetime.fromtimestamp(
            BACKGROUND_PROFILE_PATH.stat().st_mtime
        ).isoformat(sep=" "),
        "dataset_dir": str(DATASET_DIR.resolve()),
        "n_rods": int(len(df)),
        "all_saved_max_less_than_raw_max": bool(np.all(df["max_saved_value"] < df["max_raw_value"])),
        "subtraction_drop_min": float(df["subtraction_drop"].min()),
        "subtraction_drop_mean": float(df["subtraction_drop"].mean()),
        "subtraction_drop_max": float(df["subtraction_drop"].max()),
    }
    (OUTPUT_DIR / "background_subtraction_verification.json").write_text(json.dumps(verify, indent=2))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    ax.scatter(
        df["r_mean"],
        df["intensity_p98"],
        s=26,
        alpha=0.80,
        color="#1f77b4",
        edgecolors="none",
    )
    ax.set_xlabel("Mean r")
    ax.set_ylabel("Intensity p98 after background subtraction")
    ax.set_title("25nm stationary rods: intensity vs mean r")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "intensity_p98_vs_mean_r.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
