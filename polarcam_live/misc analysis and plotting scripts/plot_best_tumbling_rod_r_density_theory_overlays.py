from __future__ import annotations

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
SUMMARY_PATH = ROD_DIR / "capture_maxfps_15x15_r_density_theory_overlays_summary.json"

MODELS: dict[str, dict[str, float | str]] = {
    "water": {
        "label": "Water theory",
        "A": 0.894504,
        "B": 0.944236,
        "C": 0.128466,
        "color": "#d95f02",
    },
    "glycerol_first_new": {
        "label": "Glycerol theory",
        "A": 0.464980,
        "B": 0.736949,
        "C": 0.297224,
        "color": "#1b9e77",
    },
}

BIN_WIDTHS = (0.05, 0.025)


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


def load_r_values() -> np.ndarray:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    xy = np.asarray(meta.get("xy_series") or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise RuntimeError("No valid xy_series in capture_maxfps_15x15_meta.json")
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    return np.hypot(xy[:, 0], xy[:, 1])


def theory_r_density(r: np.ndarray, A: float, B: float, C: float) -> tuple[np.ndarray, float]:
    # theta(r) = asin(sqrt((A*r)/(B-C*r)))
    # With uniform occupancy of all orientations:
    # p(theta) = sin(theta), 0 <= theta <= pi/2
    # p(r) = p(theta(r)) * |dtheta/dr|
    rr = np.asarray(r, dtype=np.float64)
    r_max = float(B / (A + C))
    out = np.full(rr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    if np.any(valid):
        rv = rr[valid]
        denom = B - (C * rv)
        v = (A * rv) / denom
        v = np.clip(v, 0.0, 1.0 - 1e-12)
        out[valid] = (A * B) / (2.0 * denom * denom * np.sqrt(1.0 - v))
    return out, r_max


def theory_r_cdf(r: np.ndarray, A: float, B: float, C: float) -> tuple[np.ndarray, float]:
    rr = np.asarray(r, dtype=np.float64)
    r_max = float(B / (A + C))
    out = np.ones(rr.shape, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    out[rr < 0.0] = 0.0
    if np.any(valid):
        rv = rr[valid]
        denom = B - (C * rv)
        v = (A * rv) / denom
        v = np.clip(v, 0.0, 1.0)
        theta = np.arcsin(np.sqrt(v))
        out[valid] = 1.0 - np.cos(theta)
    return out, r_max


def make_plot(r: np.ndarray, bin_width: float) -> tuple[Path, dict[str, object]]:
    bins = np.arange(0.0, 1.0 + 0.5 * bin_width, bin_width, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist_density, _ = np.histogram(r, bins=bins, density=True)
    y_cap = 1.2 * float(np.max(hist_density)) if hist_density.size else 1.0

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.hist(
        r,
        bins=bins,
        density=True,
        color="#3b7ba5",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
        label=f"Rod data, Δr={bin_width:g}",
    )

    model_summary: dict[str, object] = {}
    for key, cfg in MODELS.items():
        A = float(cfg["A"])
        B = float(cfg["B"])
        C = float(cfg["C"])
        color = str(cfg["color"])
        label = str(cfg["label"])
        r_max = float(B / (A + C))
        r_grid = np.linspace(0.0, max(0.0, r_max - 1e-5), 1200, dtype=np.float64)
        theory_density, _ = theory_r_density(r_grid, A, B, C)
        theory_density_clipped = np.minimum(theory_density, y_cap)
        ax.plot(r_grid, theory_density_clipped, color=color, lw=1.6, label=f"{label} (r_max={r_max:.3f})")
        ax.axvline(r_max, color=color, lw=0.9, ls="--", alpha=0.85)
        model_summary[key] = {
            "A": A,
            "B": B,
            "C": C,
            "r_max": r_max,
        }

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_cap)
    ax.set_xlabel(r"$r = \sqrt{X^2 + Y^2}$")
    ax.set_ylabel("Density")
    ax.set_title(
        "r density from rod recording with theory overlays\n"
        "rod_x1019_y97_20260810-114604_1786358764605838700"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.7)

    suffix = str(bin_width).replace(".", "p")
    out_path = ROD_DIR / f"capture_maxfps_15x15_r_density_with_water_glycerol_theory_bin{suffix}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path, {
        "bin_width_r": bin_width,
        "measured_histogram_max_density": float(np.max(hist_density)) if hist_density.size else 0.0,
        "theory_curve_plot_cap_density": float(y_cap),
        "histogram_bins": [float(v) for v in bins],
        "histogram_density": [float(v) for v in hist_density],
        "histogram_centers": [float(v) for v in centers],
        "models": model_summary,
    }


def make_cdf_plot(r: np.ndarray) -> tuple[Path, dict[str, object]]:
    r_sorted = np.sort(np.asarray(r, dtype=np.float64))
    n = int(r_sorted.size)
    y_emp = np.arange(1, n + 1, dtype=np.float64) / max(n, 1)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    ax.plot(r_sorted, y_emp, color="#3b7ba5", lw=1.6, label="Rod data CDF")

    model_summary: dict[str, object] = {}
    r_grid = np.linspace(0.0, 1.0, 1600, dtype=np.float64)
    for key, cfg in MODELS.items():
        A = float(cfg["A"])
        B = float(cfg["B"])
        C = float(cfg["C"])
        color = str(cfg["color"])
        label = str(cfg["label"])
        cdf, r_max = theory_r_cdf(r_grid, A, B, C)
        ax.plot(r_grid, cdf, color=color, lw=1.6, label=f"{label} CDF (r_max={r_max:.3f})")
        ax.axvline(r_max, color=color, lw=0.9, ls="--", alpha=0.85)
        model_summary[key] = {
            "A": A,
            "B": B,
            "C": C,
            "r_max": r_max,
        }

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$r = \sqrt{X^2 + Y^2}$")
    ax.set_ylabel("CDF")
    ax.set_title(
        "r CDF from rod recording with theory overlays\n"
        "rod_x1019_y97_20260810-114604_1786358764605838700"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.7)

    out_path = ROD_DIR / "capture_maxfps_15x15_r_cdf_with_water_glycerol_theory.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path, {
        "models": model_summary,
        "measured_cdf_points": n,
    }


def main() -> None:
    style_matplotlib()
    r = load_r_values()
    outputs: list[dict[str, object]] = []
    for bin_width in BIN_WIDTHS:
        out_path, meta = make_plot(r, bin_width)
        outputs.append({"plot_path": str(out_path), **meta})
        print(f"Plot: {out_path}")
    cdf_path, cdf_meta = make_cdf_plot(r)
    print(f"Plot: {cdf_path}")

    summary = {
        "rod_dir": str(ROD_DIR),
        "meta_path": str(META_PATH),
        "n_points": int(r.size),
        "r_mean": float(np.mean(r)),
        "r_median": float(np.median(r)),
        "r_max_sample": float(np.max(r)),
        "outputs": outputs,
        "cdf_output": {"plot_path": str(cdf_path), **cdf_meta},
        "theory_definition": "p(r) = sin(theta(r)) * |dtheta/dr| with theta(r) = asin(sqrt((A*r)/(B-C*r)))",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
