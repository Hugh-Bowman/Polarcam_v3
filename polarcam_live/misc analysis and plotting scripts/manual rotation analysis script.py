"""Reproduce the final manual-rotation validation plots.

This script intentionally does not introduce a new matching method.  It uses
the accepted matched-rod table already produced by the manual-rotation analysis
and applies only the final filtering/plotting step:

- default phi-change filter: +/-8 degrees from constellation rotation
- stricter filter for record pair 7->8, the 140 to 160 degree point: +/-5 degrees

Inputs are the CSV outputs from the established analysis in
``stationary_rod_dataset_11062026_manual_rotation/plots/manual_box_rotation_gui``.
The raw recordings remain in the ``pending/field_roi_*`` folders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = (
    Path("stationary_rod_dataset_11062026_manual_rotation")
    / "plots"
    / "manual_box_rotation_gui"
)
DEFAULT_SOURCE_PREFIX = "multipoint_constellation_relaxed_phi_filter_8deg"
DEFAULT_OUTPUT_PREFIX = "multipoint_constellation_relaxed_pair78_strict5deg"


def keep_match(row: pd.Series, default_phi_limit_deg: float, strict_pair: tuple[int, int], strict_phi_limit_deg: float, max_residual_px: float) -> bool:
    pair = (int(row["from_record_index"]), int(row["to_record_index"]))
    phi_limit = strict_phi_limit_deg if pair == strict_pair else default_phi_limit_deg
    return (
        abs(float(row["phi_error_from_multipoint_constellation_deg"])) <= phi_limit
        and float(row["fit_residual_px"]) <= max_residual_px
    )


def make_outputs(
    analysis_dir: Path,
    source_prefix: str,
    output_prefix: str,
    default_phi_limit_deg: float,
    strict_pair: tuple[int, int],
    strict_phi_limit_deg: float,
    max_residual_px: float,
) -> dict:
    matches_path = analysis_dir / f"{source_prefix}_matches.csv"
    pair_report_path = analysis_dir / f"{source_prefix}_pair_report.csv"
    if not matches_path.exists():
        raise FileNotFoundError(matches_path)
    if not pair_report_path.exists():
        raise FileNotFoundError(pair_report_path)

    matches = pd.read_csv(matches_path)
    pair_report_in = pd.read_csv(pair_report_path)

    used = matches[
        matches.apply(
            keep_match,
            axis=1,
            default_phi_limit_deg=default_phi_limit_deg,
            strict_pair=strict_pair,
            strict_phi_limit_deg=strict_phi_limit_deg,
            max_residual_px=max_residual_px,
        )
    ].copy()
    used.to_csv(analysis_dir / f"{output_prefix}_matches.csv", index=False)

    mean_rows = [
        {
            "manual_rotation_from_constellation_deg": 0.0,
            "cumulative_mean_phi_change_deg": 0.0,
            "pair_manual_rotation_from_constellation_deg": 0.0,
            "pair_mean_phi_delta_deg": 0.0,
            "pair_variance_phi_delta_deg2": 0.0,
            "pair_sem_phi_delta_deg": 0.0,
            "cumulative_sem_phi_change_deg": 0.0,
            "n_rods": 0,
            "phi_filter_limit_deg": 0.0,
        }
    ]
    pair_rows = []
    cum_manual = 0.0
    cum_phi = 0.0
    cum_var_sum = 0.0

    for _, pair in pair_report_in.iterrows():
        i = int(pair["from_record_index"])
        j = int(pair["to_record_index"])
        g = used[(used["from_record_index"] == i) & (used["to_record_index"] == j)]
        manual_step = float(pair["constellation_manual_rotation_deg"])
        cum_manual += manual_step

        if len(g):
            deltas = g["phi_delta_deg"].to_numpy(float)
            mean_delta = float(np.mean(deltas))
            var_delta = float(np.var(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            sem_delta = float(math.sqrt(var_delta / len(deltas)))
        else:
            mean_delta = 0.0
            var_delta = 0.0
            sem_delta = 0.0

        cum_phi += mean_delta
        cum_var_sum += sem_delta**2
        phi_limit = strict_phi_limit_deg if (i, j) == strict_pair else default_phi_limit_deg

        mean_rows.append(
            {
                "manual_rotation_from_constellation_deg": cum_manual,
                "cumulative_mean_phi_change_deg": cum_phi,
                "pair_manual_rotation_from_constellation_deg": manual_step,
                "pair_mean_phi_delta_deg": mean_delta,
                "pair_variance_phi_delta_deg2": var_delta,
                "pair_sem_phi_delta_deg": sem_delta,
                "cumulative_sem_phi_change_deg": float(math.sqrt(cum_var_sum)),
                "n_rods": int(len(g)),
                "phi_filter_limit_deg": phi_limit,
            }
        )
        pair_rows.append(
            {
                "from_record_index": i,
                "to_record_index": j,
                "n_input_matches": int(pair["n_input_matches"]),
                "n_used_after_mixed_filter": int(len(g)),
                "phi_filter_limit_deg": phi_limit,
                "image_rotation_deg": float(pair["image_rotation_deg"]),
                "constellation_manual_rotation_deg": manual_step,
                "shift_x_px": float(pair["shift_x_px"]),
                "shift_y_px": float(pair["shift_y_px"]),
                "median_fit_residual_px_all": float(pair["median_fit_residual_px_all"]),
                "median_fit_residual_px_used": float(np.median(g["fit_residual_px"])) if len(g) else np.nan,
                "mean_phi_error_from_constellation_deg_used": float(np.mean(np.abs(g["phi_error_from_multipoint_constellation_deg"]))) if len(g) else np.nan,
            }
        )

    mean_df = pd.DataFrame(mean_rows)
    pair_df = pd.DataFrame(pair_rows)
    mean_df.to_csv(analysis_dir / f"{output_prefix}_mean_phi.csv", index=False)
    pair_df.to_csv(analysis_dir / f"{output_prefix}_pair_report.csv", index=False)

    manual = mean_df["manual_rotation_from_constellation_deg"].to_numpy(float)
    phi = mean_df["cumulative_mean_phi_change_deg"].to_numpy(float)
    err = mean_df["cumulative_sem_phi_change_deg"].to_numpy(float)
    resid = phi - manual

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.errorbar(manual, phi, yerr=err, fmt="o-", color="#1f77b4", ecolor="#1f77b4", capsize=4, lw=2, ms=6, label="Matched rods")
    ax.plot([0, 180], [0, 180], "--", color="0.45", lw=1.5, label=r"$y=x$")
    ax.set_xlim(-5, 185)
    ax.set_ylim(-5, 185)
    ax.set_xticks(np.arange(0, 181, 20))
    ax.set_yticks(np.arange(0, 181, 20))
    ax.set_xlabel("manual rotation angle (degrees)")
    ax.set_ylabel(r"mean $\phi$ change from anisotropy (degrees)")
    ax.set_title("Verification of Phi reconstruction by manual rotation of gold nanorods")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(analysis_dir / f"{output_prefix}_phi_plot_0to180.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.errorbar(manual, resid, yerr=err, fmt="o-", color="#d62728", ecolor="#d62728", capsize=4, lw=2, ms=6)
    ax.axhline(0, color="0.35", ls="--", lw=1.3)
    ax.set_xlim(-5, 185)
    ax.set_xticks(np.arange(0, 181, 20))
    ax.set_xlabel("manual rotation angle (degrees)")
    ax.set_ylabel("residual (degrees)")
    ax.set_title("Residuals after stricter 140 to 160 degree phi filtering")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(analysis_dir / f"{output_prefix}_residuals_0to180.png")
    plt.close(fig)

    rods_lines = [
        "Number of rods used for each data point",
        f"Analysis: {source_prefix}; pair {strict_pair[0]}->{strict_pair[1]} tightened to +/-{strict_phi_limit_deg:g} degrees",
        "",
        "manual rotation angle (deg)\trods used",
    ]
    for _, row in mean_df.iterrows():
        rods_lines.append(f"{float(row['manual_rotation_from_constellation_deg']):.2f}\t{int(row['n_rods'])}")
    (analysis_dir / f"{output_prefix}_rods_used_by_point.txt").write_text("\n".join(rods_lines) + "\n", encoding="utf-8")

    summary = {
        "source_matches": str(matches_path),
        "source_pair_report": str(pair_report_path),
        "final_manual_deg": float(manual[-1]),
        "final_phi_deg": float(phi[-1]),
        "final_sem_deg": float(err[-1]),
        "final_residual_deg": float(resid[-1]),
        "rods_per_data_point": [int(v) for v in mean_df["n_rods"].to_list()],
    }
    (analysis_dir / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--phi-filter-deg", type=float, default=8.0)
    parser.add_argument("--strict-pair", default="7,8")
    parser.add_argument("--strict-phi-filter-deg", type=float, default=5.0)
    parser.add_argument("--max-residual-px", type=float, default=4.0)
    args = parser.parse_args()

    strict_pair = tuple(int(v.strip()) for v in args.strict_pair.split(",", 1))
    summary = make_outputs(
        analysis_dir=args.analysis_dir.resolve(),
        source_prefix=args.source_prefix,
        output_prefix=args.output_prefix,
        default_phi_limit_deg=args.phi_filter_deg,
        strict_pair=strict_pair,
        strict_phi_limit_deg=args.strict_phi_filter_deg,
        max_residual_px=args.max_residual_px,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
