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

SELECTED_INDICES = [3, 5, 7, 8, 9, 11, 19]

MANIFEST_CSV = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod\xy_distribution_gallery_top20_median_r_numbered.csv"
)
OUT_DIR = MANIFEST_CSV.parent
OUT_XY_PNG = OUT_DIR / "xy_distribution_indices_3_5_7_8_9_11_19.png"
OUT_THETA_PNG = OUT_DIR / "theta_distribution_indices_3_5_7_8_9_11_19_glycerol.png"
OUT_SEL_CSV = OUT_DIR / "selected_indices_3_5_7_8_9_11_19_manifest.csv"
OUT_SUMMARY_JSON = OUT_DIR / "selected_indices_3_5_7_8_9_11_19_summary.json"

# Current glycerol theta(r) curve used in the pooled glycerol analysis.
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


def read_top20_manifest() -> list[dict[str, str]]:
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_xy(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return xy[:, :2]


def select_rows(all_rows: list[dict[str, str]], indices: list[int]) -> list[dict[str, str]]:
    wanted = {idx for idx in indices}
    selected = [row for row in all_rows if int(row["index"]) in wanted]
    selected.sort(key=lambda row: int(row["index"]))
    missing = sorted(wanted.difference(int(row["index"]) for row in selected))
    if missing:
        raise RuntimeError(f"Missing requested indices: {missing}")
    return selected


def write_selected_manifest(rows: list[dict[str, str]]) -> None:
    with OUT_SEL_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "subset", "rod", "meta_path", "n_points", "r_median", "r_mean", "r_max"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_pooled_xy(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
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
                "index": int(row["index"]),
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
        raise RuntimeError("No usable XY data found for selected indices.")
    return np.vstack(pooled_xy), np.concatenate(pooled_r), details


def plot_joint_xy(xy: np.ndarray, n_recordings: int) -> None:
    x = xy[:, 0]
    y = xy[:, 1]
    lim = float(np.nanmax(np.abs(np.concatenate([x, y]))))
    lim = max(lim * 1.03, 1.0)

    fig, ax = plt.subplots(figsize=(XY_FIG_W_MM * MM_TO_IN, XY_FIG_H_MM * MM_TO_IN))
    ax.scatter(x, y, s=2.2, color="#1f77b4", alpha=0.28, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "Joint X,Y distribution for selected glycerol tumbling recordings\n"
        f"indices {', '.join(str(v) for v in SELECTED_INDICES)} | {n_recordings} recordings | {xy.shape[0]} points"
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
        f"indices {', '.join(str(v) for v in SELECTED_INDICES)} | glycerol theta(r) | n={n_recordings}"
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
    rows = select_rows(read_top20_manifest(), SELECTED_INDICES)
    write_selected_manifest(rows)
    xy, r, details = build_pooled_xy(rows)
    plot_joint_xy(xy, len(details))
    theta_summary = plot_theta_distribution(r, len(details))

    summary = {
        "selected_indices": SELECTED_INDICES,
        "source_top20_manifest_csv": str(MANIFEST_CSV),
        "selected_manifest_csv": str(OUT_SEL_CSV),
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
    print(f"Manifest: {OUT_SEL_CSV}")
    print(f"Summary: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
