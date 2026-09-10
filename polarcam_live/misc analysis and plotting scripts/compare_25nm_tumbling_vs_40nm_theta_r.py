from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURVE_25_CSV = (
    Path("datasets") / "tumbling 25nm glycerol"
    / "plots"
    / "balanced_phi_xy_points"
    / "theta_vs_r_balanced_sampled_points_recording_bootstrap_uncertainty_band.csv"
)
CURVE_40_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
CURVE_40_NEW_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "balanced_phi_xy_points_all_recordings"
    / "theta_vs_r_balanced_sampled_points_recording_bootstrap_uncertainty_band.csv"
)
CURVE_40_SUBSET_NEW_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "balanced_phi_xy_points_selected_subset"
    / "theta_vs_r_balanced_sampled_points_uncertainty_band.csv"
)
OUTPUT_DIR = Path("datasets") / "tumbling 25nm glycerol" / "plots" / "balanced_phi_xy_points"


def _load_curve(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {
        "r": np.asarray([float(row["r"]) for row in rows], dtype=np.float64),
        "center": np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64),
        "lo": np.asarray([float(row["theta_deg_lo_1sigma"]) for row in rows], dtype=np.float64),
        "hi": np.asarray([float(row["theta_deg_hi_1sigma"]) for row in rows], dtype=np.float64),
        "std": np.asarray([float(row["theta_deg_std"]) for row in rows], dtype=np.float64),
    }


def main() -> None:
    curve25_path = Path.cwd() / CURVE_25_CSV
    curve40_path = Path.cwd() / CURVE_40_CSV
    curve40_new_path = Path.cwd() / CURVE_40_NEW_CSV
    curve40_subset_new_path = Path.cwd() / CURVE_40_SUBSET_NEW_CSV
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    c25 = _load_curve(curve25_path)
    c40 = _load_curve(curve40_path)
    c40n = _load_curve(curve40_new_path)
    c40s = _load_curve(curve40_subset_new_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(c25["r"], c25["lo"], c25["hi"], color="#2ca02c", alpha=0.20, label="25nm tumbling 1 sigma")
    ax.plot(c25["r"], c25["center"], color="#1b7f3a", lw=2.3, label="25nm tumbling theta(r)")
    ax.fill_between(c40["r"], c40["lo"], c40["hi"], color="#1f77b4", alpha=0.18, label="40nm rods 1 sigma")
    ax.plot(c40["r"], c40["center"], color="#144d7a", lw=2.1, label="40nm rods theta(r)")
    ax.fill_between(c40n["r"], c40n["lo"], c40n["hi"], color="#d62728", alpha=0.16, label="40nm balanced 1 sigma")
    ax.plot(c40n["r"], c40n["center"], color="#a51c30", lw=2.1, label="40nm balanced theta(r)")
    ax.fill_between(c40s["r"], c40s["lo"], c40s["hi"], color="#9467bd", alpha=0.16, label="40nm selected-subset 1 sigma")
    ax.plot(c40s["r"], c40s["center"], color="#6f42c1", lw=2.1, label="40nm selected-subset theta(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r): 25nm tumbling vs old and new 40nm curves")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_25nm_tumbling_vs_40nm.png", dpi=220)
    plt.close(fig)

    summary = {
        "curve_25_csv": str(curve25_path.resolve()),
        "curve_40_csv": str(curve40_path.resolve()),
        "curve_40_new_csv": str(curve40_new_path.resolve()),
        "curve_40_subset_new_csv": str(curve40_subset_new_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "curve_25_r_range": [float(np.min(c25["r"])), float(np.max(c25["r"]))],
        "curve_40_r_range": [float(np.min(c40["r"])), float(np.max(c40["r"]))],
        "curve_40_new_r_range": [float(np.min(c40n["r"])), float(np.max(c40n["r"]))],
        "curve_40_subset_new_r_range": [float(np.min(c40s["r"])), float(np.max(c40s["r"]))],
    }
    (out_dir / "theta_vs_r_25nm_tumbling_vs_40nm_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
