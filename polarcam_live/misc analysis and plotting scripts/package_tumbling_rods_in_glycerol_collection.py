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
FIG_W_MM = 90.0
FIG_H_MM = 62.0

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\indices_5_8_9_best_21_26_glycerol095_manifest.csv"
)
OUT_ROOT = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\tumbling rods in glycerol collection"
)
RODS_DIR = OUT_ROOT / "selected_rods"
PLOTS_DIR = OUT_ROOT / "plots"
POINTS_CSV = OUT_ROOT / "pooled_xy_theta_phi_points.csv"
SELECTION_CSV = OUT_ROOT / "selection_manifest.csv"
SUMMARY_JSON = OUT_ROOT / "collection_summary.json"

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
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
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


def ensure_dirs() -> None:
    RODS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def copy_selected_rods(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    copied_rows: list[dict[str, str]] = []
    for row in rows:
        src_meta = Path(row["meta_path"])
        src_dir = src_meta.parent
        label = str(row["index"]).strip()
        dst_dir = RODS_DIR / f"{label}_{src_dir.name}"
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        copied_row = dict(row)
        copied_row["copied_dir"] = str(dst_dir)
        copied_rows.append(copied_row)
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


def write_points_csv(rows: list[dict[str, str]]) -> dict[str, float | int]:
    total_points = 0
    pooled_r: list[np.ndarray] = []
    pooled_theta_deg: list[np.ndarray] = []
    pooled_theta_rad: list[np.ndarray] = []
    pooled_phi_deg: list[np.ndarray] = []
    pooled_phi_rad: list[np.ndarray] = []

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
            meta_path = Path(row["meta_path"])
            xy = load_xy(meta_path)
            if xy.size == 0:
                continue
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.hypot(x, y)
            theta_rad, _, _ = theta_from_r_model(r)
            theta_deg = np.degrees(theta_rad)
            phi_rad = np.arctan2(y, x)
            phi_deg = np.degrees(phi_rad)
            phi_deg_0_360 = np.mod(phi_deg, 360.0)

            pooled_r.append(r)
            pooled_theta_rad.append(theta_rad)
            pooled_theta_deg.append(theta_deg)
            pooled_phi_rad.append(phi_rad)
            pooled_phi_deg.append(phi_deg)
            total_points += int(r.size)

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

    r_all = np.concatenate(pooled_r)
    theta_deg_all = np.concatenate(pooled_theta_deg)
    theta_rad_all = np.concatenate(pooled_theta_rad)
    phi_deg_all = np.concatenate(pooled_phi_deg)
    phi_rad_all = np.concatenate(pooled_phi_rad)
    return {
        "n_points": int(total_points),
        "r_median": float(np.median(r_all)),
        "r_mean": float(np.mean(r_all)),
        "theta_median_deg": float(np.median(theta_deg_all)),
        "theta_mean_deg": float(np.mean(theta_deg_all)),
        "theta_median_rad": float(np.median(theta_rad_all)),
        "theta_mean_rad": float(np.mean(theta_rad_all)),
        "phi_mean_deg": float(np.mean(phi_deg_all)),
        "phi_mean_rad": float(np.mean(phi_rad_all)),
    }


def plot_theta_hist(rows: list[dict[str, str]], bin_width_deg: float) -> dict[str, float | str]:
    pooled_theta_deg: list[np.ndarray] = []
    n_clipped = 0
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        theta_rad, clipped_mask, _ = theta_from_r_model(r)
        pooled_theta_deg.append(np.degrees(theta_rad))
        n_clipped += int(np.count_nonzero(clipped_mask))

    theta_deg = np.concatenate(pooled_theta_deg)
    bins = np.arange(0.0, 90.0 + bin_width_deg, bin_width_deg, dtype=np.float64)
    if bins[-1] < 90.0:
        bins = np.append(bins, 90.0)

    hist_density, edges = np.histogram(theta_deg, bins=bins, density=True)
    theory_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theory_rad = np.radians(theory_deg)
    theory_density_per_deg = np.sin(theory_rad) * (np.pi / 180.0)
    density_scale = float(np.max(theory_density_per_deg))
    hist_relative = hist_density / density_scale
    theory_relative = theory_density_per_deg / density_scale

    centers = 0.5 * (edges[:-1] + edges[1:])
    out_png = PLOTS_DIR / f"theta_distribution_relative_bin{str(bin_width_deg).replace('.', 'p')}deg.png"

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    widths = np.diff(edges)
    ax.bar(
        centers,
        hist_relative,
        width=widths * 0.92,
        color="#c26f22",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        align="center",
        label="Selected rod data",
    )
    ax.plot(
        theory_deg,
        theory_relative,
        color="black",
        lw=1.2,
        ls="--",
        label="Theoretical uniform occupancy of all orientations",
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(1.08, float(np.max(hist_relative)) * 1.05))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Relative density")
    ax.set_title(f"Theta distribution from selected rods ({int(bin_width_deg)} deg bins)")
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
    ensure_dirs()
    rows = read_rows()
    copied_rows = copy_selected_rods(rows)
    write_selection_manifest(copied_rows)
    pooled_stats = write_points_csv(copied_rows)
    hist_10 = plot_theta_hist(copied_rows, 10.0)
    hist_5 = plot_theta_hist(copied_rows, 5.0)

    summary = {
        "source_manifest_csv": str(MANIFEST_CSV),
        "collection_root": str(OUT_ROOT),
        "theta_model": {
            "definition": "theta = asin(sqrt((A*r)/(B-C*r)))",
            "A": A_GLY,
            "B": B_GLY,
            "C": C_GLY,
            "r_max": float(B_GLY / (A_GLY + C_GLY)),
        },
        "selection_manifest_csv": str(SELECTION_CSV),
        "pooled_points_csv": str(POINTS_CSV),
        "pooled_stats": pooled_stats,
        "theta_histograms": [hist_10, hist_5],
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

    print(f"Collection: {OUT_ROOT}")
    print(f"Manifest: {SELECTION_CSV}")
    print(f"Points: {POINTS_CSV}")
    print(f"Plot: {hist_10['plot_path']}")
    print(f"Plot: {hist_5['plot_path']}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
