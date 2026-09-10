from __future__ import annotations

import argparse
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

ROD_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\best tumbling rod\rod_x1019_y97_20260810-114604_1786358764605838700"
)
META_PATH = ROD_DIR / "capture_maxfps_15x15_meta.json"
CAPTURE_JSON_PATH = ROD_DIR / "capture_maxfps_15x15.json"
OUT_PNG = ROD_DIR / "capture_maxfps_15x15_theta_distribution_glycerol99p5_new_abc.png"
OUT_JSON = ROD_DIR / "capture_maxfps_15x15_theta_distribution_glycerol99p5_new_abc_summary.json"

MODEL_PRESETS: dict[str, dict[str, object]] = {
    "glycerol_n1p396": {
        "label": "glycerol n=1.396",
        "A": 0.633617,
        "B": 0.831926,
        "C": 0.241089,
        "png_name": "capture_maxfps_15x15_theta_distribution_glycerol_n1p396.png",
        "json_name": "capture_maxfps_15x15_theta_distribution_glycerol_n1p396_summary.json",
    },
    "water_new_abc": {
        "label": "water",
        "A": 0.894504,
        "B": 0.944236,
        "C": 0.128466,
        "png_name": "capture_maxfps_15x15_theta_distribution_water_new_abc.png",
        "json_name": "capture_maxfps_15x15_theta_distribution_water_new_abc_summary.json",
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


def theta_from_r_model(r: np.ndarray, A: float, B: float, C: float) -> tuple[np.ndarray, np.ndarray, float]:
    r = np.asarray(r, dtype=np.float64)
    r_max = float(B / (A + C))
    theta = np.full(r.shape, np.pi / 2.0, dtype=np.float64)
    valid = np.isfinite(r) & (r >= 0.0) & (r < r_max)
    denom = B - (C * r[valid])
    val = (A * r[valid]) / np.maximum(1e-12, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    clipped = np.isfinite(r) & (r >= r_max)
    return theta, clipped, r_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS.keys()), default="glycerol_n1p396")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style_matplotlib()
    model = dict(MODEL_PRESETS[str(args.model)])
    A = float(model["A"])
    B = float(model["B"])
    C = float(model["C"])
    label = str(model["label"])
    out_png = ROD_DIR / str(model["png_name"])
    out_json = ROD_DIR / str(model["json_name"])
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    capture_meta = json.loads(CAPTURE_JSON_PATH.read_text(encoding="utf-8"))
    xy = np.asarray(meta.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise RuntimeError("No xy_series found in capture_maxfps_15x15_meta.json")

    x = xy[:, 0]
    y = xy[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    r = np.hypot(x, y)

    theta_rad, clipped_mask, r_max = theta_from_r_model(r, A, B, C)
    theta_deg = np.degrees(theta_rad)
    theta_mean = float(np.mean(theta_deg))
    theta_median = float(np.median(theta_deg))
    clipped_n = int(np.count_nonzero(clipped_mask))

    bins = np.arange(0.0, 100.0, 10.0, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.hist(
        theta_deg,
        bins=bins,
        density=True,
        color="#c26f22",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        label="Rod data",
    )

    theta_grid_deg = np.linspace(0.0, 90.0, 600, dtype=np.float64)
    theta_grid_rad = np.radians(theta_grid_deg)
    sin_pdf_per_deg = np.sin(theta_grid_rad) * (np.pi / 180.0)
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
        "Theta distribution from rod recording\n"
        "rod_x1019_y97_20260810-114604_1786358764605838700"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    summary = {
        "model_key": str(args.model),
        "model_label": label,
        "rod_dir": str(ROD_DIR),
        "meta_path": str(META_PATH),
        "recording_json_path": str(CAPTURE_JSON_PATH),
        "recording_npy": str(Path(meta.get("npy_file", "capture_maxfps_15x15.npy"))),
        "recording_original_npy_path": str(capture_meta.get("npy_path", "")),
        "formula": "theta(r) = asin(sqrt((A*r)/(B-C*r)))",
        "curve_parameters": {"A": A, "B": B, "C": C},
        "r_max": r_max,
        "n_points": int(theta_deg.size),
        "n_clipped_to_90deg": clipped_n,
        "theta_mean_deg": theta_mean,
        "theta_median_deg": theta_median,
        "theta_histogram_bin_width_deg": 10.0,
        "r_from_saved_xy_series": True,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Plot: {out_png}")
    print(f"Summary: {out_json}")
    print(f"r_max={r_max:.6f}")
    print(f"n_clipped_to_90deg={clipped_n}")
    print(f"theta_mean_deg={theta_mean:.4f}")
    print(f"theta_median_deg={theta_median:.4f}")


if __name__ == "__main__":
    main()
