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
XY_FIG_W_MM = 90.0
XY_FIG_H_MM = 85.0
THETA_FIG_W_MM = 90.0
THETA_FIG_H_MM = 62.0
THETA_BIN_WIDTH_DEG = 10.0

TOP20_INDICES = [5, 8, 9, 11, 19]
RANK20_30 = [21, 24, 26, 27]

TOP20_MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\xy_distribution_gallery_top20_median_r_numbered.csv"
)
RANK20_30_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\xy_distribution_gallery_candidates_20_to_30.csv"
)
BEST_ROD_META = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\best tumbling rod\rod_x1019_y97_20260810-114604_1786358764605838700\capture_maxfps_15x15_meta.json"
)
OUT_DIR = TOP20_MANIFEST_CSV.parent
OUT_PREFIX = "indices_5_8_9_11_19_best_21_24_26_27_glycerol095"
OUT_XY_PNG = OUT_DIR / f"{OUT_PREFIX}_xy.png"
OUT_THETA_PNG = OUT_DIR / f"{OUT_PREFIX}_theta_distribution.png"
OUT_MANIFEST_CSV = OUT_DIR / f"{OUT_PREFIX}_manifest.csv"
OUT_SUMMARY_JSON = OUT_DIR / f"{OUT_PREFIX}_summary.json"

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


def load_xy(meta_path: Path) -> np.ndarray:
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


def select_rows(rows: list[dict[str, str]], key: str, values: list[int]) -> list[dict[str, str]]:
    wanted = {int(v) for v in values}
    selected = [row for row in rows if int(row[key]) in wanted]
    selected.sort(key=lambda row: int(row[key]))
    found = {int(row[key]) for row in selected}
    missing = sorted(wanted.difference(found))
    if missing:
        raise RuntimeError(f"Missing requested {key} values: {missing}")
    return selected


def build_selection_rows() -> list[dict[str, str]]:
    rows_top20 = select_rows(read_csv_rows(TOP20_MANIFEST_CSV), "index", TOP20_INDICES)
    rows_20_30 = select_rows(read_csv_rows(RANK20_30_CSV), "rank", RANK20_30)
    rows_20_30_norm = [
        {
            "index": row["rank"],
            "subset": row["subset"],
            "rod": row["rod"],
            "meta_path": row["meta_path"],
            "n_points": row["n_points"],
            "r_median": row["r_median"],
            "r_mean": row["r_mean"],
            "r_max": row["r_max"],
        }
        for row in rows_20_30
    ]
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
    return [*rows_top20, best_row, *rows_20_30_norm]


def build_pool(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    pooled_xy: list[np.ndarray] = []
    pooled_r: list[np.ndarray] = []
    details: list[dict[str, object]] = []
    for row in rows:
        xy = load_xy(Path(row["meta_path"]))
        if xy.size == 0:
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        pooled_xy.append(xy)
        pooled_r.append(r)
        details.append(
            {
                "index": row["index"],
                "subset": row["subset"],
                "rod": row["rod"],
                "meta_path": row["meta_path"],
                "n_points": int(xy.shape[0]),
                "r_median": float(np.median(r)),
                "r_mean": float(np.mean(r)),
                "r_max": float(np.max(r)),
            }
        )
    if not pooled_xy:
        raise RuntimeError("No usable XY data found.")
    return np.vstack(pooled_xy), np.concatenate(pooled_r), details


def write_manifest(details: list[dict[str, object]]) -> None:
    with OUT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "subset", "rod", "meta_path", "n_points", "r_median", "r_mean", "r_max"],
        )
        writer.writeheader()
        writer.writerows(details)


def plot_joint_xy(xy: np.ndarray, n_recordings: int) -> None:
    x = xy[:, 0]
    y = xy[:, 1]
    lim = float(np.nanmax(np.abs(np.concatenate([x, y]))))
    lim = max(lim * 1.03, 1.0)

    fig, ax = plt.subplots(figsize=(XY_FIG_W_MM * MM_TO_IN, XY_FIG_H_MM * MM_TO_IN))
    ax.scatter(x, y, s=2.0, color="#1f77b4", alpha=0.28, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "Joint X,Y distribution for selected glycerol tumbling recordings\n"
        "5, 8, 9, 11, 19, best, 21, 24, 26, 27"
        f" | {n_recordings} recordings | {xy.shape[0]} points"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_XY_PNG, dpi=300)
    plt.close(fig)


def plot_theta_distribution(r: np.ndarray, n_recordings: int) -> dict[str, float | int]:
    bins = np.arange(0.0, 90.0 + THETA_BIN_WIDTH_DEG, THETA_BIN_WIDTH_DEG, dtype=np.float64)
    theta_grid_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theta_grid_rad = np.radians(theta_grid_deg)
    sin_pdf_per_deg = np.sin(theta_grid_rad) * (np.pi / 180.0)

    theta_rad, clipped_mask, r_max = theta_from_r_model(r, A_GLY, B_GLY, C_GLY)
    theta_deg = np.degrees(theta_rad)
    theta_mean = float(np.mean(theta_deg))
    theta_median = float(np.median(theta_deg))
    clipped_n = int(np.count_nonzero(clipped_mask))

    fig, ax = plt.subplots(figsize=(THETA_FIG_W_MM * MM_TO_IN, THETA_FIG_H_MM * MM_TO_IN))
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
    ax.axvline(theta_mean, color="black", lw=0.9, ls="--", alpha=0.9, label=f"mean={theta_mean:.2f} deg")
    ax.axvline(theta_median, color="#666666", lw=0.9, ls=":", alpha=0.95, label=f"median={theta_median:.2f} deg")
    ax.set_xlim(0.0, 90.0)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Theta distribution for selected glycerol tumbling recordings\n"
        "5, 8, 9, 11, 19, best, 21, 24, 26, 27 | glycerol theta(r)"
        f" | n={n_recordings}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_THETA_PNG, dpi=300)
    plt.close(fig)

    return {
        "A": A_GLY,
        "B": B_GLY,
        "C": C_GLY,
        "r_max": r_max,
        "theta_mean_deg": theta_mean,
        "theta_median_deg": theta_median,
        "n_clipped_to_90deg": clipped_n,
    }


def main() -> None:
    style_matplotlib()
    rows = build_selection_rows()
    xy, r, details = build_pool(rows)
    write_manifest(details)
    plot_joint_xy(xy, len(details))
    theta_summary = plot_theta_distribution(r, len(details))

    summary = {
        "selected_top20_indices": TOP20_INDICES,
        "selected_ranks_20_30": RANK20_30,
        "included_best_rod_meta": str(BEST_ROD_META),
        "selected_manifest_csv": str(OUT_MANIFEST_CSV),
        "joint_xy_plot": str(OUT_XY_PNG),
        "theta_distribution_plot": str(OUT_THETA_PNG),
        "n_recordings": len(details),
        "n_points": int(xy.shape[0]),
        "pooled_r_median": float(np.median(r)),
        "pooled_r_mean": float(np.mean(r)),
        "theta_model": theta_summary,
        "recordings": details,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_XY_PNG}")
    print(f"Saved: {OUT_THETA_PNG}")
    print(f"Manifest: {OUT_MANIFEST_CSV}")
    print(f"Summary: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
