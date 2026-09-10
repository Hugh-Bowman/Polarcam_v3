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
FIG_H_MM = 240.0
N_COLS = 2
THETA_BIN_WIDTH_DEG = 10.0

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\indices_5_8_9_11_19_best_21_24_26_27_glycerol095_manifest.csv"
)
OUT_DIR = MANIFEST_CSV.parent
OUT_PNG = OUT_DIR / "indices_5_8_9_11_19_best_21_24_26_27_glycerol095_theta_gallery.png"
OUT_JSON = OUT_DIR / "indices_5_8_9_11_19_best_21_24_26_27_glycerol095_theta_gallery_summary.json"

A_GLY = 0.633617
B_GLY = 0.831926
C_GLY = 0.241089


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT - 1,
            "ytick.labelsize": FONT_PT - 1,
            "legend.fontsize": FONT_PT - 1,
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


def theta_from_r_model(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rr = np.asarray(r, dtype=np.float64)
    r_max = float(B_GLY / (A_GLY + C_GLY))
    theta = np.full(rr.shape, np.pi / 2.0, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    denom = B_GLY - (C_GLY * rr[valid])
    val = (A_GLY * rr[valid]) / np.maximum(1e-12, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    clipped = np.isfinite(rr) & (rr >= r_max)
    return theta, clipped, r_max


def main() -> None:
    style_matplotlib()
    rows = read_rows()
    if not rows:
        raise RuntimeError("Manifest is empty.")

    bins = np.arange(0.0, 90.0 + THETA_BIN_WIDTH_DEG, THETA_BIN_WIDTH_DEG, dtype=np.float64)
    theta_grid_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theta_grid_rad = np.radians(theta_grid_deg)
    sin_pdf_per_deg = np.sin(theta_grid_rad) * (np.pi / 180.0)

    records: list[dict[str, object]] = []
    ymax = 0.0
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        theta_rad, clipped_mask, r_max = theta_from_r_model(r)
        theta_deg = np.degrees(theta_rad)
        hist, _ = np.histogram(theta_deg, bins=bins, density=True)
        ymax = max(ymax, float(np.max(hist)))
        records.append(
            {
                "label": row["index"],
                "theta_deg": theta_deg,
                "n_points": int(theta_deg.size),
                "theta_mean_deg": float(np.mean(theta_deg)),
                "theta_median_deg": float(np.median(theta_deg)),
                "n_clipped_to_90deg": int(np.count_nonzero(clipped_mask)),
                "r_max": r_max,
            }
        )

    ymax = max(ymax, float(np.max(sin_pdf_per_deg))) * 1.1
    n_plots = len(records)
    n_rows = math.ceil(n_plots / N_COLS)
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, records):
        theta_deg = np.asarray(item["theta_deg"], dtype=np.float64)
        ax.hist(
            theta_deg,
            bins=bins,
            density=True,
            color="#c26f22",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
        )
        ax.plot(theta_grid_deg, sin_pdf_per_deg, color="black", lw=1.0, ls="--")
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Density")
        ax.text(
            0.03,
            0.97,
            str(item["label"]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_PT + 3,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.4},
        )
        ax.set_title(
            f"n={int(item['n_points'])} | med {float(item['theta_median_deg']):.1f} deg",
            pad=4.0,
        )

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle("Theta distributions for selected rods", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), pad=0.9)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    summary = {
        "source_manifest_csv": str(MANIFEST_CSV),
        "output_png": str(OUT_PNG),
        "theta_model": {
            "definition": "theta = asin(sqrt((A*r)/(B-C*r)))",
            "A": A_GLY,
            "B": B_GLY,
            "C": C_GLY,
            "r_max": float(B_GLY / (A_GLY + C_GLY)),
        },
        "theta_histogram_bin_width_deg": THETA_BIN_WIDTH_DEG,
        "records": [
            {
                "label": str(item["label"]),
                "n_points": int(item["n_points"]),
                "theta_mean_deg": float(item["theta_mean_deg"]),
                "theta_median_deg": float(item["theta_median_deg"]),
                "n_clipped_to_90deg": int(item["n_clipped_to_90deg"]),
            }
            for item in records
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PNG}")
    print(f"Summary: {OUT_JSON}")


if __name__ == "__main__":
    main()
