from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MM_TO_IN = 1.0 / 25.4
FONT_PT = 7
FIG_W_MM = 90.0
FIG_H_MM = 62.0
R_BIN_WIDTH = 0.025
THETA_BIN_WIDTH_DEG = 10.0

SELECTED_INDICES = [5, 8, 9, 11, 19]

TOP20_MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\xy_distribution_gallery_top20_median_r_numbered.csv"
)
BEST_ROD_META = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\best tumbling rod\rod_x1019_y97_20260810-114604_1786358764605838700\capture_maxfps_15x15_meta.json"
)
OUT_DIR = TOP20_MANIFEST_CSV.parent
OUT_PREFIX = "indices_5_8_9_11_19_plus_best_rod_glycerol095"
OUT_MANIFEST_CSV = OUT_DIR / f"{OUT_PREFIX}_manifest.csv"
OUT_R_PDF_PNG = OUT_DIR / f"{OUT_PREFIX}_r_pdf.png"
OUT_R_CDF_PNG = OUT_DIR / f"{OUT_PREFIX}_r_cdf.png"
OUT_THETA_PNG = OUT_DIR / f"{OUT_PREFIX}_theta_distribution.png"
OUT_SUMMARY_JSON = OUT_DIR / f"{OUT_PREFIX}_summary.json"

# New glycerol model with r_max ~ 0.95
A_GLY = 0.633617
B_GLY = 0.831926
C_GLY = 0.241089


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_xy_from_meta(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return xy[:, :2]


def theta_from_r_model(r: np.ndarray, A: float, B: float, C: float) -> tuple[np.ndarray, np.ndarray, float]:
    rr = np.asarray(r, dtype=np.float64)
    r_max = float(B / (A + C))
    theta = np.full(rr.shape, np.pi / 2.0, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    denom = B - (C * rr[valid])
    val = (A * rr[valid]) / np.maximum(1e-12, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    clipped = np.isfinite(rr) & (rr >= r_max)
    return theta, clipped, r_max


def theory_theta_from_r_grid(r_grid: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    val = (A * r_grid) / np.maximum(1e-12, B - C * r_grid)
    val = np.clip(val, 0.0, 1.0)
    return np.arcsin(np.sqrt(val))


def theory_r_pdf_cdf(r_max: float, A: float, B: float, C: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_grid = np.linspace(0.0, r_max, 3000, dtype=np.float64)
    theta = theory_theta_from_r_grid(r_grid, A, B, C)
    cdf = 1.0 - np.cos(theta)
    pdf = np.gradient(cdf, r_grid)
    pdf = np.clip(pdf, 0.0, None)
    return r_grid, pdf, cdf


def select_index_rows(rows: list[dict[str, str]], indices: list[int]) -> list[dict[str, str]]:
    wanted = {idx for idx in indices}
    selected = [row for row in rows if int(row["index"]) in wanted]
    selected.sort(key=lambda row: int(row["index"]))
    missing = sorted(wanted.difference(int(row["index"]) for row in selected))
    if missing:
        raise RuntimeError(f"Missing requested indices: {missing}")
    return selected


def build_selection_rows() -> list[dict[str, str]]:
    selected = select_index_rows(read_csv_rows(TOP20_MANIFEST_CSV), SELECTED_INDICES)
    best_row = {
        "index": "best",
        "subset": "best_tumbling_rod",
        "rod": BEST_ROD_META.parent.name,
        "meta_path": str(BEST_ROD_META),
        "n_points": "",
        "r_median": "",
        "r_mean": "",
        "r_max": "",
    }
    return [*selected, best_row]


def load_pool(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    pooled_r: list[np.ndarray] = []
    details: list[dict[str, object]] = []
    for row in rows:
        meta_path = Path(row["meta_path"])
        xy = load_xy_from_meta(meta_path)
        if xy.size == 0:
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        pooled_r.append(r)
        details.append(
            {
                "index": row["index"],
                "subset": row["subset"],
                "rod": row["rod"],
                "meta_path": row["meta_path"],
                "n_points": int(r.size),
                "r_median": float(np.median(r)),
                "r_mean": float(np.mean(r)),
                "r_max": float(np.max(r)),
            }
        )
    if not pooled_r:
        raise RuntimeError("No usable recordings in selected pool.")
    return np.concatenate(pooled_r), details


def write_manifest(details: list[dict[str, object]]) -> None:
    with OUT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "subset", "rod", "meta_path", "n_points", "r_median", "r_mean", "r_max"],
        )
        writer.writeheader()
        writer.writerows(details)


def plot_r_pdf(r_raw: np.ndarray, r_clipped: np.ndarray, r_max: float, theory_r: np.ndarray, theory_pdf: np.ndarray) -> dict[str, object]:
    bins = np.arange(0.0, r_max + R_BIN_WIDTH, R_BIN_WIDTH, dtype=np.float64)
    if bins[-1] < r_max:
        bins = np.append(bins, r_max)
    hist, edges = np.histogram(r_clipped, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cap = 1.2 * max(np.max(hist), 1e-9)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.hist(
        r_clipped,
        bins=bins,
        density=True,
        color="#5aa0c8",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
        label="Measured data (clipped at $r_{\\max}$)",
    )
    ax.plot(
        theory_r,
        np.minimum(theory_pdf, cap),
        color="#c26f22",
        lw=1.4,
        label="Theory: uniform cos($\\theta$)",
    )
    ax.set_xlim(0.0, r_max)
    ax.set_ylim(0.0, cap)
    ax.set_xlabel(r"$r=\sqrt{X^2+Y^2}$")
    ax.set_ylabel("Density")
    ax.set_title("Measured and theoretical r PDF\nindices 5, 8, 9, 11, 19 + best tumbling rod")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_R_PDF_PNG, dpi=300)
    plt.close(fig)

    return {
        "plot_path": str(OUT_R_PDF_PNG),
        "bin_width_r": R_BIN_WIDTH,
        "r_max": r_max,
        "measured_histogram_max_density": float(np.max(hist)),
        "theory_curve_plot_cap_density": float(cap),
        "histogram_bins": edges.tolist(),
        "histogram_centers": centers.tolist(),
        "histogram_density": hist.tolist(),
        "n_points_raw": int(r_raw.size),
        "n_points_clipped": int(r_clipped.size),
    }


def plot_r_cdf(r_clipped: np.ndarray, r_max: float, theory_r: np.ndarray, theory_cdf: np.ndarray) -> dict[str, object]:
    r_sorted = np.sort(r_clipped)
    y = np.arange(1, r_sorted.size + 1, dtype=np.float64) / float(r_sorted.size)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.plot(r_sorted, y, color="#1f77b4", lw=1.3, label="Measured data CDF (clipped)")
    ax.plot(theory_r, theory_cdf, color="#c26f22", lw=1.3, label="Theory CDF")
    ax.set_xlim(0.0, r_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$r=\sqrt{X^2+Y^2}$")
    ax.set_ylabel("CDF")
    ax.set_title("Measured and theoretical r CDF\nindices 5, 8, 9, 11, 19 + best tumbling rod")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_R_CDF_PNG, dpi=300)
    plt.close(fig)

    return {
        "plot_path": str(OUT_R_CDF_PNG),
        "n_points_clipped": int(r_sorted.size),
        "r_max": r_max,
    }


def plot_theta_distribution(r_raw: np.ndarray, r_max: float) -> dict[str, object]:
    theta_rad, clipped_mask, _ = theta_from_r_model(r_raw, A_GLY, B_GLY, C_GLY)
    theta_deg = np.degrees(theta_rad)
    bins = np.arange(0.0, 90.0 + THETA_BIN_WIDTH_DEG, THETA_BIN_WIDTH_DEG, dtype=np.float64)
    theta_grid_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theta_grid_rad = np.radians(theta_grid_deg)
    sin_pdf_per_deg = np.sin(theta_grid_rad) * (np.pi / 180.0)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.hist(
        theta_deg,
        bins=bins,
        density=True,
        color="#c26f22",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        label="Selected rod data",
    )
    ax.plot(
        theta_grid_deg,
        sin_pdf_per_deg,
        color="black",
        lw=1.2,
        ls="--",
        label="Theoretical uniform occupancy of all orientations",
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Theta distribution from selected recordings\n"
        "new glycerol model, indices 5, 8, 9, 11, 19 + best tumbling rod"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_THETA_PNG, dpi=300)
    plt.close(fig)

    return {
        "plot_path": str(OUT_THETA_PNG),
        "theta_histogram_bin_width_deg": THETA_BIN_WIDTH_DEG,
        "theta_mean_deg": float(np.mean(theta_deg)),
        "theta_median_deg": float(np.median(theta_deg)),
        "n_clipped_to_90deg": int(np.count_nonzero(clipped_mask)),
        "r_max": r_max,
    }


def main() -> None:
    style_matplotlib()
    rows = build_selection_rows()
    r_raw, details = load_pool(rows)
    write_manifest(details)

    r_max = float(B_GLY / (A_GLY + C_GLY))
    r_clipped = np.clip(r_raw, 0.0, r_max)
    theory_r, theory_pdf, theory_cdf = theory_r_pdf_cdf(r_max, A_GLY, B_GLY, C_GLY)

    pdf_summary = plot_r_pdf(r_raw, r_clipped, r_max, theory_r, theory_pdf)
    cdf_summary = plot_r_cdf(r_clipped, r_max, theory_r, theory_cdf)
    theta_summary = plot_theta_distribution(r_raw, r_max)

    summary = {
        "selected_indices": SELECTED_INDICES,
        "included_best_rod_meta": str(BEST_ROD_META),
        "selected_manifest_csv": str(OUT_MANIFEST_CSV),
        "theta_model": {
            "definition": "theta = asin(sqrt((A*r)/(B-C*r)))",
            "A": A_GLY,
            "B": B_GLY,
            "C": C_GLY,
            "r_max": r_max,
        },
        "n_recordings": len(details),
        "n_points_raw": int(r_raw.size),
        "n_points_clipped_for_r_plots": int(r_clipped.size),
        "pooled_r_median_raw": float(np.median(r_raw)),
        "pooled_r_mean_raw": float(np.mean(r_raw)),
        "r_pdf_output": pdf_summary,
        "r_cdf_output": cdf_summary,
        "theta_output": theta_summary,
        "recordings": details,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_R_PDF_PNG}")
    print(f"Saved: {OUT_R_CDF_PNG}")
    print(f"Saved: {OUT_THETA_PNG}")
    print(f"Manifest: {OUT_MANIFEST_CSV}")
    print(f"Summary: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
