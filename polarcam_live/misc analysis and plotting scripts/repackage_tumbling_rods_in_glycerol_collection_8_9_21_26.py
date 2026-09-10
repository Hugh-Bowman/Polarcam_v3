from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MM_TO_IN = 1.0 / 25.4
FONT_PT = 7
XY_FIG_W_MM = 90.0
XY_FIG_H_MM = 85.0
THETA_FIG_W_MM = 90.0
THETA_FIG_H_MM = 62.0

SOURCE_SELECTION_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\tumbling rods in glycerol collection\selection_manifest.csv"
)
OUT_ROOT = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\tumbling rods in glycerol collection"
)
SELECTED_LABELS = ["8", "9", "21", "26"]

SELECTED_RODS_DIR = OUT_ROOT / "selected_rods"
PLOTS_DIR = OUT_ROOT / "plots"
SELECTION_CSV = OUT_ROOT / "selection_manifest.csv"
POINTS_CSV = OUT_ROOT / "pooled_xy_theta_phi_points.csv"
SUMMARY_JSON = OUT_ROOT / "collection_summary.json"
JOINT_XY_PNG = PLOTS_DIR / "joint_xy_selected_rods.png"
THETA_10_PNG = PLOTS_DIR / "theta_distribution_relative_bin10p0deg.png"

# Corrected guide-derived glycerol coefficients for:
# n = 1.47, NA_out = 1.3, NA_in = 0.39
A_GLY = 0.4649796110130161
B_GLY = 0.7369490542844699
C_GLY = 0.2972242960658035


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
        }
    )


def read_rows() -> list[dict[str, str]]:
    with SOURCE_SELECTION_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    selected = [row for row in rows if row["index"] in SELECTED_LABELS]
    selected.sort(key=lambda row: SELECTED_LABELS.index(row["index"]))
    found = {row["index"] for row in selected}
    missing = [label for label in SELECTED_LABELS if label not in found]
    if missing:
        raise RuntimeError(f"Missing labels in source selection manifest: {missing}")
    return selected


def reset_output_dirs() -> None:
    if SELECTED_RODS_DIR.exists():
        shutil.rmtree(SELECTED_RODS_DIR)
    if PLOTS_DIR.exists():
        shutil.rmtree(PLOTS_DIR)
    SELECTED_RODS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def copy_selected_rods(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    copied_rows: list[dict[str, str]] = []
    for row in rows:
        src_meta = Path(row["meta_path"])
        src_dir = src_meta.parent
        dst_dir = SELECTED_RODS_DIR / f"{row['index']}_{src_dir.name}"
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        copied = dict(row)
        copied["copied_dir"] = str(dst_dir)
        copied_rows.append(copied)
    return copied_rows


def write_selection_manifest(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "index",
        "subset",
        "rod",
        "meta_path",
        "copied_dir",
        "n_points",
        "r_median",
        "r_mean",
        "r_max",
    ]
    with SELECTION_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def write_points_csv(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, int]:
    pooled_xy: list[np.ndarray] = []
    pooled_theta_deg: list[np.ndarray] = []
    n_clipped_total = 0

    with POINTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "source_label",
                "source_subset",
                "source_rod",
                "point_index",
                "x",
                "y",
                "r",
                "theta_rad",
                "theta_deg",
                "phi_rad",
                "phi_deg",
                "phi_deg_0_360",
            ]
        )
        for row in rows:
            xy = load_xy(Path(row["meta_path"]))
            if xy.size == 0:
                continue
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.hypot(x, y)
            theta_rad, clipped_mask, _ = theta_from_r_model(r)
            theta_deg = np.degrees(theta_rad)
            phi_rad = np.arctan2(y, x)
            phi_deg = np.degrees(phi_rad)
            phi_deg_0_360 = np.mod(phi_deg, 360.0)
            pooled_xy.append(xy)
            pooled_theta_deg.append(theta_deg)
            n_clipped_total += int(np.count_nonzero(clipped_mask))

            for idx in range(r.size):
                writer.writerow(
                    [
                        row["index"],
                        row["subset"],
                        row["rod"],
                        idx,
                        f"{x[idx]:.12g}",
                        f"{y[idx]:.12g}",
                        f"{r[idx]:.12g}",
                        f"{theta_rad[idx]:.12g}",
                        f"{theta_deg[idx]:.12g}",
                        f"{phi_rad[idx]:.12g}",
                        f"{phi_deg[idx]:.12g}",
                        f"{phi_deg_0_360[idx]:.12g}",
                    ]
                )

    return np.vstack(pooled_xy), np.concatenate(pooled_theta_deg), n_clipped_total


def plot_joint_xy(pooled_xy: np.ndarray) -> dict[str, float | int | str]:
    x = pooled_xy[:, 0]
    y = pooled_xy[:, 1]
    lim = 1.0
    ticks = np.arange(-1.0, 1.0 + 0.001, 0.5, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(XY_FIG_W_MM * MM_TO_IN, XY_FIG_H_MM * MM_TO_IN))
    ax.scatter(x, y, s=2.0, color="#1f77b4", alpha=0.28, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.7)
    fig.savefig(JOINT_XY_PNG, dpi=300)
    plt.close(fig)

    return {
        "plot_path": str(JOINT_XY_PNG),
        "n_points": int(pooled_xy.shape[0]),
        "axis_limit": lim,
    }


def plot_theta_hist(theta_deg: np.ndarray, n_clipped: int, bin_width_deg: float, out_png: Path) -> dict[str, object]:
    bins = np.arange(0.0, 90.0 + bin_width_deg, bin_width_deg, dtype=np.float64)
    if bins[-1] < 90.0:
        bins = np.append(bins, 90.0)

    hist_density, edges = np.histogram(theta_deg, bins=bins, density=True)
    theory_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theory_rad = np.radians(theory_deg)
    theory_density = np.sin(theory_rad) * (np.pi / 180.0)
    scale = float(np.max(theory_density))
    hist_relative = hist_density / scale
    theory_relative = theory_density / scale
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(THETA_FIG_W_MM * MM_TO_IN, THETA_FIG_H_MM * MM_TO_IN))
    ax.bar(
        centers,
        hist_relative,
        width=np.diff(edges) * 0.92,
        color="#c26f22",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        label="Tumbling Rods in Glycerol",
    )
    ax.plot(
        theory_deg,
        theory_relative,
        color="black",
        lw=1.2,
        ls="--",
        label="Theoretical uniform occupancy\nof all orientations",
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(1.08, float(np.max(hist_relative)) * 1.05))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Relative density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return {
        "plot_path": str(out_png),
        "bin_width_deg": bin_width_deg,
        "n_points": int(theta_deg.size),
        "n_clipped_to_90deg": int(n_clipped),
        "histogram_centers_deg": centers.tolist(),
        "histogram_relative_density": hist_relative.tolist(),
    }


def main() -> None:
    style_matplotlib()
    rows = read_rows()
    reset_output_dirs()
    copied_rows = copy_selected_rods(rows)
    write_selection_manifest(copied_rows)

    pooled_xy, theta_deg, n_clipped = write_points_csv(copied_rows)
    xy_summary = plot_joint_xy(pooled_xy)
    hist_10 = plot_theta_hist(theta_deg, n_clipped, 10.0, THETA_10_PNG)

    r = np.hypot(pooled_xy[:, 0], pooled_xy[:, 1])
    summary = {
        "source_selection_csv": str(SOURCE_SELECTION_CSV),
        "collection_root": str(OUT_ROOT),
        "selection_manifest_csv": str(SELECTION_CSV),
        "pooled_points_csv": str(POINTS_CSV),
        "selected_labels": SELECTED_LABELS,
        "theta_model": {
            "definition": "theta = asin(sqrt((A*r)/(B-C*r)))",
            "A": A_GLY,
            "B": B_GLY,
            "C": C_GLY,
            "r_max": float(B_GLY / (A_GLY + C_GLY)),
        },
        "pooled_stats": {
            "n_points": int(pooled_xy.shape[0]),
            "r_median": float(np.median(r)),
            "r_mean": float(np.mean(r)),
            "theta_median_deg": float(np.median(theta_deg)),
            "theta_mean_deg": float(np.mean(theta_deg)),
        },
        "joint_xy": xy_summary,
        "theta_histograms": [hist_10],
        "selected_rods": [
            {
                "index": row["index"],
                "subset": row["subset"],
                "rod": row["rod"],
                "copied_dir": row["copied_dir"],
                "meta_path": row["meta_path"],
            }
            for row in copied_rows
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selection: {SELECTION_CSV}")
    print(f"Points: {POINTS_CSV}")
    print(f"XY: {JOINT_XY_PNG}")
    print(f"Theta 10: {THETA_10_PNG}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
