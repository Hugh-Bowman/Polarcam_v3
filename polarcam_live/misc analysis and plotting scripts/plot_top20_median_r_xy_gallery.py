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
FIG_H_MM = 220.0
FONT_PT = 7
N_COLS = 2
TOP_N = 20

DATA_ROOT = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\background_subtracted"
)
OUT_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod"
)
OUT_PNG_1 = OUT_DIR / "xy_distribution_gallery_top20_median_r_numbered_01_to_10.png"
OUT_PNG_2 = OUT_DIR / "xy_distribution_gallery_top20_median_r_numbered_11_to_20.png"
OUT_CSV = OUT_DIR / "xy_distribution_gallery_top20_median_r_numbered.csv"
OUT_JSON = OUT_DIR / "xy_distribution_gallery_top20_median_r_numbered_summary.json"
SUBDIRS = ("pending", "good", "bad")


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


def load_xy(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return xy[:, :2]


def collect_recordings() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subset in SUBDIRS:
        subset_dir = DATA_ROOT / subset
        if not subset_dir.exists():
            continue
        for rod_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            xy = load_xy(meta_path)
            if xy.size == 0:
                continue
            r = np.hypot(xy[:, 0], xy[:, 1])
            records.append(
                {
                    "subset": subset,
                    "rod": rod_dir.name,
                    "meta_path": str(meta_path),
                    "n_points": int(xy.shape[0]),
                    "r_median": float(np.median(r)),
                    "r_mean": float(np.mean(r)),
                    "r_max": float(np.max(r)),
                    "xy": xy,
                }
            )
    records.sort(key=lambda item: float(item["r_median"]), reverse=True)
    return records[:TOP_N]


def write_csv(records: list[dict[str, object]]) -> None:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "subset", "rod", "meta_path", "n_points", "r_median", "r_mean", "r_max"],
        )
        writer.writeheader()
        for idx, item in enumerate(records, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "subset": item["subset"],
                    "rod": item["rod"],
                    "meta_path": item["meta_path"],
                    "n_points": item["n_points"],
                    "r_median": item["r_median"],
                    "r_mean": item["r_mean"],
                    "r_max": item["r_max"],
                }
            )


def render_gallery(records: list[dict[str, object]], out_png: Path, start_index: int, global_lim: float) -> None:
    n_plots = len(records)
    n_rows = math.ceil(n_plots / N_COLS)
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx_offset, (ax, item) in enumerate(zip(axes_flat, records)):
        idx = start_index + idx_offset
        xy = np.asarray(item["xy"], dtype=np.float64)
        ax.scatter(xy[:, 0], xy[:, 1], s=3.4, color="#1f77b4", alpha=0.38, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(-global_lim, global_lim)
        ax.grid(True, alpha=0.22)
        ax.text(
            0.03,
            0.97,
            f"{idx}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_PT + 3,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.6},
        )
        ax.set_title(
            f"{item['subset']} | n={item['n_points']} | med r={item['r_median']:.3f}",
            pad=4.0,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    end_index = start_index + n_plots - 1
    fig.suptitle(f"Top 20 recordings by median r: individual X,Y plots ({start_index}-{end_index})", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), pad=0.9)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    style_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_recordings()
    if not records:
        raise RuntimeError("No valid recordings found.")

    global_lim = 0.0
    for item in records:
        xy = np.asarray(item["xy"], dtype=np.float64)
        lim = float(np.nanmax(np.abs(np.concatenate([xy[:, 0], xy[:, 1]]))))
        global_lim = max(global_lim, lim)
    global_lim = max(global_lim * 1.03, 1.0)

    render_gallery(records[:10], OUT_PNG_1, 1, global_lim)
    render_gallery(records[10:20], OUT_PNG_2, 11, global_lim)

    write_csv(records)
    summary = {
        "data_root": str(DATA_ROOT),
        "output_png_01_to_10": str(OUT_PNG_1),
        "output_png_11_to_20": str(OUT_PNG_2),
        "output_csv": str(OUT_CSV),
        "n_recordings": len(records),
        "global_axis_limit": global_lim,
        "records": [
            {
                "index": idx,
                "subset": str(item["subset"]),
                "rod": str(item["rod"]),
                "meta_path": str(item["meta_path"]),
                "n_points": int(item["n_points"]),
                "r_median": float(item["r_median"]),
                "r_mean": float(item["r_mean"]),
                "r_max": float(item["r_max"]),
            }
            for idx, item in enumerate(records, start=1)
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PNG_1}")
    print(f"Saved: {OUT_PNG_2}")
    print(f"CSV: {OUT_CSV}")
    print(f"Summary: {OUT_JSON}")


if __name__ == "__main__":
    main()
