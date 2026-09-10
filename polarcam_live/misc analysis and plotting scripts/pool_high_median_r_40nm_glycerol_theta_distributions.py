from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 90.0
FIG_H_MM = 62.0
FONT_PT = 7

DATA_ROOT = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\background_subtracted"
)
BEST_ROD_SUMMARY = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\best tumbling rod\rod_x1019_y97_20260810-114604_1786358764605838700\capture_maxfps_15x15_r_density_theory_overlays_summary.json"
)
OUT_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\glycerol_tumbling_background_subtracted_20260715\40x65nm\plots\theta_distribution_median_r_above_best_rod"
)
MANIFEST_CSV = OUT_DIR / "selected_recordings_manifest.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"

SUBDIRS = ("pending", "good", "bad")
THETA_BIN_WIDTH_DEG = 10.0

MODELS: dict[str, dict[str, float | str]] = {
    "glycerol_n1p396": {
        "label": "glycerol n=1.396",
        "A": 0.633617,
        "B": 0.831926,
        "C": 0.241089,
        "color": "#c26f22",
        "png_name": "theta_distribution_glycerol_n1p396.png",
    },
    "water_new_abc": {
        "label": "water",
        "A": 0.894504,
        "B": 0.944236,
        "C": 0.128466,
        "color": "#2d6f95",
        "png_name": "theta_distribution_water_new_abc.png",
    },
}


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


def load_best_rod_threshold() -> float:
    payload = json.loads(BEST_ROD_SUMMARY.read_text(encoding="utf-8"))
    return float(payload["r_median"])


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


def load_selected_points(threshold: float) -> tuple[np.ndarray, list[dict[str, object]]]:
    pooled_r: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    for src in SUBDIRS:
        src_dir = DATA_ROOT / src
        if not src_dir.exists():
            continue
        for rod_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            xy = np.asarray(payload.get("xy_series") or [], dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
            if xy.size == 0:
                continue
            r = np.hypot(xy[:, 0], xy[:, 1])
            r_median = float(np.median(r))
            if not (r_median > threshold):
                continue
            pooled_r.append(r)
            manifest.append(
                {
                    "subset": src,
                    "rod": rod_dir.name,
                    "meta_path": str(meta_path),
                    "n_points": int(r.size),
                    "r_median": r_median,
                    "r_mean": float(np.mean(r)),
                    "r_max": float(np.max(r)),
                }
            )
    if not pooled_r:
        raise RuntimeError("No recordings passed the median-r threshold.")
    return np.concatenate(pooled_r), manifest


def write_manifest(rows: list[dict[str, object]]) -> None:
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["subset", "rod", "meta_path", "n_points", "r_median", "r_mean", "r_max"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_theta_distribution(r: np.ndarray, threshold: float, manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    bins = np.arange(0.0, 90.0 + THETA_BIN_WIDTH_DEG, THETA_BIN_WIDTH_DEG, dtype=np.float64)
    theta_grid_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theta_grid_rad = np.radians(theta_grid_deg)
    sin_pdf_per_deg = np.sin(theta_grid_rad) * (np.pi / 180.0)

    for model_key, cfg in MODELS.items():
        A = float(cfg["A"])
        B = float(cfg["B"])
        C = float(cfg["C"])
        label = str(cfg["label"])
        color = str(cfg["color"])
        out_png = OUT_DIR / str(cfg["png_name"])

        theta_rad, clipped_mask, r_max = theta_from_r_model(r, A, B, C)
        theta_deg = np.degrees(theta_rad)
        theta_mean = float(np.mean(theta_deg))
        theta_median = float(np.median(theta_deg))
        clipped_n = int(np.count_nonzero(clipped_mask))

        fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
        ax.hist(
            theta_deg,
            bins=bins,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            label="Pooled rod data",
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
            f"Theta distribution from pooled 40x65nm glycerol recordings\n"
            f"{label} theta(r), selected rods: median r > {threshold:.3f} | n={len(manifest)} recordings"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout(pad=0.7)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        outputs.append(
            {
                "model_key": model_key,
                "model_label": label,
                "plot_path": str(out_png),
                "curve_parameters": {"A": A, "B": B, "C": C},
                "r_max": r_max,
                "theta_mean_deg": theta_mean,
                "theta_median_deg": theta_median,
                "n_clipped_to_90deg": clipped_n,
            }
        )
    return outputs


def main() -> None:
    style_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = load_best_rod_threshold()
    r, manifest = load_selected_points(threshold)
    write_manifest(manifest)
    outputs = plot_theta_distribution(r, threshold, manifest)

    summary = {
        "data_root": str(DATA_ROOT),
        "best_rod_threshold_source": str(BEST_ROD_SUMMARY),
        "best_rod_r_median_threshold": threshold,
        "n_selected_recordings": int(len(manifest)),
        "n_selected_points": int(r.size),
        "pooled_r_median": float(np.median(r)),
        "pooled_r_mean": float(np.mean(r)),
        "manifest_csv": str(MANIFEST_CSV),
        "theta_histogram_bin_width_deg": THETA_BIN_WIDTH_DEG,
        "outputs": outputs,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selected recordings: {len(manifest)}")
    print(f"Selected points: {r.size}")
    print(f"Manifest: {MANIFEST_CSV}")
    for item in outputs:
        print(f"Plot: {item['plot_path']}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
