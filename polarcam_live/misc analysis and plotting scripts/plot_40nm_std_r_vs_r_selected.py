from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_40nm_bg_subtracted_sound_precision_new as base


OUT_DIR = base.OUTPUT_DIR
R_BIN_WIDTH = 0.05
TOP_N_PER_R_BIN = 3

# Use the current requested "second sigma for a normal distribution" definition.
R_SIGMA_LO_PCT = 2.2750131948179195
R_SIGMA_HI_PCT = 97.72498680518208
R_SIGMA_DIVISOR = 4.0


def select_best_per_r_bin(rows: list[base.RodDatum]) -> list[base.RodDatum]:
    bins: dict[int, list[base.RodDatum]] = {}
    for row in rows:
        if not np.isfinite(row.r_mean) or not np.isfinite(row.sigma_r_percentile):
            continue
        bin_idx = int(np.floor(row.r_mean / R_BIN_WIDTH))
        bins.setdefault(bin_idx, []).append(row)
    selected: list[base.RodDatum] = []
    for bin_idx in sorted(bins):
        selected.extend(
            sorted(bins[bin_idx], key=lambda r: (r.sigma_r_percentile, r.sigma_xy, r.rod))[:TOP_N_PER_R_BIN]
        )
    return selected


def write_selected_csv(path: Path, rows: list[base.RodDatum]) -> None:
    fields = [
        "label",
        "rod",
        "path",
        "r_mean",
        "r_bin_start",
        "sigma_r",
        "sigma_r_percentile",
        "theta_deg",
        "sigma_theta_deg",
        "sigma_phi_deg",
        "intensity_mean",
        "background_subtracted",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "label": r.label,
                    "rod": r.rod,
                    "path": str(r.path),
                    "r_mean": r.r_mean,
                    "r_bin_start": np.floor(r.r_mean / R_BIN_WIDTH) * R_BIN_WIDTH,
                    "sigma_r": r.sigma_r,
                    "sigma_r_percentile": r.sigma_r_percentile,
                    "theta_deg": r.theta_deg,
                    "sigma_theta_deg": r.sigma_theta_deg,
                    "sigma_phi_deg": r.sigma_phi_deg,
                    "intensity_mean": r.intensity_mean,
                    "background_subtracted": r.background_subtracted,
                }
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.R_SIGMA_LO_PCT = R_SIGMA_LO_PCT
    base.R_SIGMA_HI_PCT = R_SIGMA_HI_PCT
    base.R_SIGMA_DIVISOR = R_SIGMA_DIVISOR
    curve = base.build_curve()

    colors = {"0.6 ms": "#1f77b4", "1.2 ms": "#ff7f0e"}
    selected_by_label: dict[str, list[base.RodDatum]] = {}
    summary = {
        "method": (
            f"Bin rods by mean r in width {R_BIN_WIDTH:g}; select top {TOP_N_PER_R_BIN} "
            f"lowest sigma_r candidates per bin for each exposure. sigma_r is "
            f"(p{R_SIGMA_HI_PCT:g}(r)-p{R_SIGMA_LO_PCT:g}(r))/{R_SIGMA_DIVISOR:g}."
        ),
        "datasets": {},
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for label, root in base.DATASETS.items():
        rows = base.load_dataset(label, root, curve)
        selected = select_best_per_r_bin(rows)
        selected_by_label[label] = selected
        safe = label.replace(" ", "").replace(".", "p")
        write_selected_csv(OUT_DIR / f"{safe}_std_r_vs_r_selected_rods.csv", selected)
        ax.scatter(
            [r.r_mean for r in selected],
            [r.sigma_r_percentile for r in selected],
            s=40,
            alpha=0.78,
            edgecolors="none",
            color=colors.get(label),
            label=f"{label}",
        )
        summary["datasets"][label] = {
            "source_dir": str(root),
            "n_rods_loaded": len(rows),
            "n_selected": len(selected),
            "max_r_loaded": max((r.r_mean for r in rows), default=float("nan")),
            "max_r_selected": max((r.r_mean for r in selected), default=float("nan")),
        }

    ax.set_xlabel("mean r")
    ax.set_ylabel("std r")
    ax.set_title("40nm rods: r uncertainty vs r")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, title="Exposure")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_std_r_vs_r_selected_by_r_bin.png", dpi=220)
    plt.close(fig)

    (OUT_DIR / "std_r_vs_r_selected_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {OUT_DIR}")
    for label, rows in selected_by_label.items():
        print(f"{label}: selected {len(rows)} rods")


if __name__ == "__main__":
    main()
