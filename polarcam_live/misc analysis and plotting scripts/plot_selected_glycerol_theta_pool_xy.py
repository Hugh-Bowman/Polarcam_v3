from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 90.0
FIG_H_MM = 85.0
FONT_PT = 7

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\selected_recordings_manifest.csv"
)
OUT_DIR = MANIFEST_CSV.parent
OUT_PNG = OUT_DIR / "xy_distribution_selected_recordings.png"
OUT_JSON = OUT_DIR / "xy_distribution_selected_recordings_summary.json"


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
        }
    )


def load_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_pooled_xy(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    pooled: list[np.ndarray] = []
    names: list[str] = []
    for row in rows:
        meta_path = Path(row["meta_path"])
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            continue
        xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
        if xy.size == 0:
            continue
        pooled.append(xy[:, :2])
        names.append(row["rod"])
    if not pooled:
        raise RuntimeError("No XY data found for selected recordings.")
    return np.vstack(pooled), names


def main() -> None:
    style_matplotlib()
    rows = load_manifest_rows()
    xy, rod_names = load_pooled_xy(rows)

    x = xy[:, 0]
    y = xy[:, 1]
    lim = float(np.nanmax(np.abs(np.concatenate([x, y]))))
    lim = max(lim, 1.0)
    lim *= 1.03

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.scatter(x, y, s=2.0, color="#1f77b4", alpha=0.28, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "Pooled X,Y distribution from selected 40x65nm glycerol recordings\n"
        f"{len(rod_names)} recordings, {xy.shape[0]} points"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    summary = {
        "manifest_csv": str(MANIFEST_CSV),
        "output_png": str(OUT_PNG),
        "n_recordings": len(rod_names),
        "n_points": int(xy.shape[0]),
        "recordings": rod_names,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PNG}")
    print(f"Summary: {OUT_JSON}")


if __name__ == "__main__":
    main()
