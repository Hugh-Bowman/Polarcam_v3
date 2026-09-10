from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 180.0
FIG_H_MM = 150.0
FONT_PT = 7
N_COLS = 4

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\selected_recordings_manifest.csv"
)
OUT_DIR = MANIFEST_CSV.parent
OUT_PNG = OUT_DIR / "xy_distribution_gallery_selected_recordings.png"
OUT_JSON = OUT_DIR / "xy_distribution_gallery_selected_recordings_summary.json"


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT - 1,
            "ytick.labelsize": FONT_PT - 1,
        }
    )


def load_rows() -> list[dict[str, str]]:
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_xy(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return xy[:, :2]


def short_title(rod_name: str) -> str:
    prefix = "rod_"
    return rod_name[len(prefix) :] if rod_name.startswith(prefix) else rod_name


def main() -> None:
    style_matplotlib()
    rows = load_rows()
    if not rows:
        raise RuntimeError("Manifest is empty.")

    per_row: list[dict[str, object]] = []
    global_lim = 0.0
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        x = xy[:, 0]
        y = xy[:, 1]
        lim = float(np.nanmax(np.abs(np.concatenate([x, y]))))
        global_lim = max(global_lim, lim)
        per_row.append(
            {
                "rod": row["rod"],
                "subset": row["subset"],
                "n_points": int(xy.shape[0]),
                "r_median": float(row["r_median"]),
                "xy": xy,
            }
        )

    if not per_row:
        raise RuntimeError("No XY data found for selected recordings.")

    global_lim = max(global_lim * 1.03, 1.0)
    n_plots = len(per_row)
    n_rows = math.ceil(n_plots / N_COLS)

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, per_row):
        xy = np.asarray(item["xy"], dtype=np.float64)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.8, color="#1f77b4", alpha=0.35, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(-global_lim, global_lim)
        ax.grid(True, alpha=0.22)
        ax.set_title(
            f"{short_title(str(item['rod']))}\n"
            f"{item['subset']} | n={item['n_points']} | med r={item['r_median']:.3f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle("Individual X,Y plots for recordings selected by median r threshold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98), pad=0.7)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    summary = {
        "manifest_csv": str(MANIFEST_CSV),
        "output_png": str(OUT_PNG),
        "n_recordings": n_plots,
        "global_axis_limit": global_lim,
        "recordings": [
            {
                "rod": str(item["rod"]),
                "subset": str(item["subset"]),
                "n_points": int(item["n_points"]),
                "r_median": float(item["r_median"]),
            }
            for item in per_row
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PNG}")
    print(f"Summary: {OUT_JSON}")


if __name__ == "__main__":
    main()
