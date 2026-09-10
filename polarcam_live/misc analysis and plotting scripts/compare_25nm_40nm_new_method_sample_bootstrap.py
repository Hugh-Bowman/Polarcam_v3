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
    / "theta_vs_r_balanced_sampled_points_uncertainty_band.csv"
)
CURVE_40_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "balanced_phi_xy_points_all_recordings"
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
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    c25 = _load_curve(curve25_path)
    c40 = _load_curve(curve40_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(c25["r"], c25["lo"], c25["hi"], color="#2ca02c", alpha=0.20, label="25x65nm 1 sigma")
    ax.plot(c25["r"], c25["center"], color="#1b7f3a", lw=2.3, label="25x65nm theta(r)")
    ax.fill_between(c40["r"], c40["lo"], c40["hi"], color="#d62728", alpha=0.18, label="40x65nm 1 sigma")
    ax.plot(c40["r"], c40["center"], color="#a51c30", lw=2.2, label="40x65nm theta(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) with sample-bootstrap error: 25x65nm vs 40x65nm")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_25nm_40nm_new_method_sample_bootstrap.png", dpi=220)
    plt.close(fig)

    summary = {
        "output_dir": str(out_dir.resolve()),
        "curve_25_csv": str(curve25_path.resolve()),
        "curve_40_csv": str(curve40_path.resolve()),
        "method": [
            "Use only recordings with range_x > 1 and range_y > 1.",
            "Pool all xy points from the accepted recordings.",
            "Bin pooled points by phi in 5 degree bins from 0 to 180 degrees.",
            "Let n be half of the least populated phi-bin size.",
            "Randomly sample n points without replacement from each phi bin.",
            "Fit theta(r) so that the sampled r distribution corresponds to uniform cos(theta).",
            "Estimate uncertainty by sample-level bootstrap on the balanced sampled point set, with replacement, refitting theta(r) for each replicate.",
            "Use the 16th and 84th percentiles of the bootstrap curves as the 1 sigma band.",
        ],
        "datasets": {
            "25x65nm": {
                "data_root": str((Path.cwd() / "tumbling 25nm glycerol").resolve()),
                "curve_r_range": [float(np.min(c25["r"])), float(np.max(c25["r"]))],
            },
            "40x65nm": {
                "data_root": str((Path.cwd() / "glycerol suspended rods 17062026").resolve()),
                "curve_r_range": [float(np.min(c40["r"])), float(np.max(c40["r"]))],
            },
        },
        "parameters": {
            "recording_filter": "range_x > 1 and range_y > 1",
            "phi_bin_deg": 5.0,
            "sample_rule": "half of least populated phi bin",
            "bootstrap_type": "sample-level bootstrap on balanced sampled points",
            "bootstrap_replicates": 250,
        },
    }
    (out_dir / "theta_vs_r_25nm_40nm_new_method_sample_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
