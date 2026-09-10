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
FIG_H_MM = 180.0
N_COLS = 2
BIN_WIDTH_DEG = 5.0

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\tumbling rods in glycerol collection\selection_manifest.csv"
)
OUT_DIR = MANIFEST_CSV.parent / "plots"
OUT_PNG = OUT_DIR / "selected_rods_theta_gallery_bin5deg.png"
OUT_JSON = OUT_DIR / "selected_rods_theta_gallery_bin5deg_summary.json"

A_GLY = 0.633617
B_GLY = 0.831926
C_GLY = 0.241089


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
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


def theta_from_r_model(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(r, dtype=np.float64)
    r_max = float(B_GLY / (A_GLY + C_GLY))
    theta = np.full(rr.shape, np.pi / 2.0, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    denom = B_GLY - (C_GLY * rr[valid])
    val = (A_GLY * rr[valid]) / np.maximum(1e-12, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    clipped = np.isfinite(rr) & (rr >= r_max)
    return np.degrees(theta), clipped


def main() -> None:
    style_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()

    bins = np.arange(0.0, 90.0 + BIN_WIDTH_DEG, BIN_WIDTH_DEG, dtype=np.float64)
    theory_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theory_density = np.sin(np.radians(theory_deg)) * (np.pi / 180.0)

    records: list[dict[str, object]] = []
    ymax = 0.0
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        theta_deg, clipped = theta_from_r_model(r)
        hist, _ = np.histogram(theta_deg, bins=bins, density=True)
        ymax = max(ymax, float(np.max(hist)))
        records.append(
            {
                "label": row["index"],
                "theta_deg": theta_deg,
                "n_points": int(theta_deg.size),
                "theta_median_deg": float(np.median(theta_deg)),
                "theta_mean_deg": float(np.mean(theta_deg)),
                "n_clipped_to_90deg": int(np.count_nonzero(clipped)),
            }
        )

    ymax = max(ymax, float(np.max(theory_density))) * 1.08
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
        ax.plot(theory_deg, theory_density, color="black", lw=1.0, ls="--")
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

    fig.suptitle("Selected rods theta distributions (5 deg bins)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), pad=0.9)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    summary = {
        "source_manifest_csv": str(MANIFEST_CSV),
        "output_png": str(OUT_PNG),
        "bin_width_deg": BIN_WIDTH_DEG,
        "theta_model": {
            "definition": "theta = asin(sqrt((A*r)/(B-C*r)))",
            "A": A_GLY,
            "B": B_GLY,
            "C": C_GLY,
            "r_max": float(B_GLY / (A_GLY + C_GLY)),
        },
        "records": [
            {
                "label": str(item["label"]),
                "n_points": int(item["n_points"]),
                "theta_median_deg": float(item["theta_median_deg"]),
                "theta_mean_deg": float(item["theta_mean_deg"]),
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
