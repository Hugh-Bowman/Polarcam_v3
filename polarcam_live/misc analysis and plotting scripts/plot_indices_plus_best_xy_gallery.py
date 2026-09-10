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
FONT_PT = 7
FIG_W_MM = 180.0
FIG_H_MM = 220.0
N_COLS = 2

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\indices_5_8_9_11_19_plus_best_rod_glycerol095_manifest.csv"
)
OUT_DIR = MANIFEST_CSV.parent
OUT_GALLERY_PNG = OUT_DIR / "indices_5_8_9_11_19_plus_best_xy_gallery.png"
OUT_COMBINED_PNG = OUT_DIR / "indices_5_8_9_11_19_plus_best_xy_all.png"
OUT_SUMMARY_JSON = OUT_DIR / "indices_5_8_9_11_19_plus_best_xy_gallery_summary.json"


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


def read_rows() -> list[dict[str, str]]:
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_xy(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return xy[:, :2]


def build_records(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], float]:
    records: list[dict[str, object]] = []
    global_lim = 0.0
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        lim = float(np.nanmax(np.abs(np.concatenate([xy[:, 0], xy[:, 1]]))))
        global_lim = max(global_lim, lim)
        label = str(row["index"]).strip()
        records.append(
            {
                "label": label,
                "subset": row["subset"],
                "rod": row["rod"],
                "n_points": int(xy.shape[0]),
                "xy": xy,
            }
        )
    if not records:
        raise RuntimeError("No valid XY data found in manifest.")
    global_lim = max(global_lim * 1.03, 1.0)
    return records, global_lim


def scatter_panel(ax: plt.Axes, xy: np.ndarray, label: str, lim: float, title_suffix: str = "") -> None:
    ax.scatter(xy[:, 0], xy[:, 1], s=2.3, color="#1f77b4", alpha=0.32, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_PT + 3,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.4},
    )
    ax.set_title(title_suffix, pad=4.0)


def plot_gallery(records: list[dict[str, object]], lim: float) -> np.ndarray:
    pooled_xy = np.vstack([np.asarray(item["xy"], dtype=np.float64) for item in records])
    panels = [*records, {"label": "all", "subset": "pooled", "rod": "all", "n_points": int(pooled_xy.shape[0]), "xy": pooled_xy}]
    n_rows = math.ceil(len(panels) / N_COLS)

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, panels):
        xy = np.asarray(item["xy"], dtype=np.float64)
        title = f"n={int(item['n_points'])}"
        scatter_panel(ax, xy, str(item["label"]), lim, title)

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    fig.suptitle("X,Y plots for selected recordings and pooled total", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), pad=0.9)
    fig.savefig(OUT_GALLERY_PNG, dpi=300)
    plt.close(fig)
    return pooled_xy


def plot_combined_only(pooled_xy: np.ndarray, lim: float) -> None:
    fig, ax = plt.subplots(figsize=(90.0 * MM_TO_IN, 85.0 * MM_TO_IN))
    scatter_panel(ax, pooled_xy, "all", lim, f"n={pooled_xy.shape[0]}")
    ax.set_title("Pooled X,Y plot", pad=4.0)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_COMBINED_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    style_matplotlib()
    rows = read_rows()
    records, lim = build_records(rows)
    pooled_xy = plot_gallery(records, lim)
    plot_combined_only(pooled_xy, lim)

    summary = {
        "source_manifest_csv": str(MANIFEST_CSV),
        "gallery_png": str(OUT_GALLERY_PNG),
        "combined_png": str(OUT_COMBINED_PNG),
        "global_axis_limit": lim,
        "records": [
            {
                "label": str(item["label"]),
                "subset": str(item["subset"]),
                "rod": str(item["rod"]),
                "n_points": int(item["n_points"]),
            }
            for item in records
        ],
        "pooled_n_points": int(pooled_xy.shape[0]),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_GALLERY_PNG}")
    print(f"Saved: {OUT_COMBINED_PNG}")
    print(f"Summary: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
