from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_40nm_bg_subtracted_sound_precision_new as base


DATASETS = {
    "0.6 ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\stationary rods 25nm with sound 0.6ms\pending"),
    "1.2 ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\stationary rods 25nm with sound 1.2ms\pending"),
}
OUTPUT_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\stationary rods 25nm combined precision general"
)


def prune_phi_selection_general(rows: list[base.RodDatum]) -> tuple[list[base.RodDatum], list[base.RodDatum]]:
    kept: list[base.RodDatum] = []
    removed: list[base.RodDatum] = []
    for row in rows:
        sigma_phi = float(row.sigma_phi_deg)
        if np.isfinite(sigma_phi) and sigma_phi > 0.0:
            kept.append(row)
        else:
            row.selected_for_phi = False
            removed.append(row)
    return kept, removed


def plot_combined_25nm(results: dict[str, dict], out_dir: Path) -> None:
    colors = {"0.6 ms": "#1f77b4", "1.2 ms": "#ff7f0e"}
    markers = {"0.6 ms": "o", "1.2 ms": "s"}
    plot_defs = [
        ("theta", "sigma_theta_deg", "std theta (deg)", "combined_std_theta_vs_theta.png"),
        ("phi", "sigma_phi_deg", "std phi (deg)", "combined_std_phi_vs_theta.png"),
    ]
    for kind, attr, ylabel, fname in plot_defs:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        base.add_low_theta_divergence_region(ax)
        plotted_y_parts: list[np.ndarray] = []
        for label, res in results.items():
            rows = res[kind]["fit_rows"]
            plotted_y_parts.append(np.asarray([getattr(r, attr) for r in rows], dtype=np.float64))
            ax.scatter(
                [r.theta_deg for r in rows],
                [getattr(r, attr) for r in rows],
                s=44,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
                color=colors.get(label),
                marker=markers.get(label, "o"),
                label=f"{label} data",
            )
        plotted_y = np.concatenate([p for p in plotted_y_parts if p.size]) if plotted_y_parts else np.asarray([])
        finite_y = plotted_y[np.isfinite(plotted_y)]
        y_cap = float(np.max(finite_y) * 1.12) if finite_y.size else base.MODEL_ERROR_PLOT_MAX_DEG
        if not np.isfinite(y_cap) or y_cap <= 0.0:
            y_cap = 1.0
        if y_cap >= base.MODEL_ERROR_PLOT_MAX_DEG:
            ax.axhline(base.MODEL_ERROR_PLOT_MAX_DEG, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.set_xlim(base.PLOT_THETA_MIN_DEG, 90.0)
        ax.set_ylim(0.0, y_cap)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"25nm rods: {ylabel} vs theta")
        ax.grid(True, alpha=0.24)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = base.build_curve()
    results: dict[str, dict] = {}
    summaries = []
    for label, root in DATASETS.items():
        rows = base.load_dataset(label, root, curve)
        selected_theta = base.select_best_per_theta_bin(rows, metric="sigma_r_percentile", flag="selected_for_theta")
        selected_phi = base.select_best_per_theta_bin(rows, metric="sigma_phi_deg", flag="selected_for_phi")
        selected_theta, theta_pruned = base.prune_theta_selection(label, selected_theta)
        selected_phi, phi_pruned = prune_phi_selection_general(selected_phi)
        fit_rows_theta = [
            r
            for r in selected_theta
            if base.THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_theta_deg)
            and float(r.sigma_theta_deg) > 0.0
        ]
        fit_rows_phi = [
            r
            for r in selected_phi
            if base.THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_phi_deg)
            and float(r.sigma_phi_deg) > 0.0
        ]
        for row in fit_rows_theta:
            row.used_for_theta_fit = True
        for row in fit_rows_phi:
            row.used_for_phi_fit = True
        if len(fit_rows_theta) < 3:
            raise SystemExit(f"Too few theta fit rows for {label}: {len(fit_rows_theta)}")
        if len(fit_rows_phi) < 3:
            raise SystemExit(f"Too few phi fit rows for {label}: {len(fit_rows_phi)}")

        shared_noise_params = base.fit_constant_sigma_xy(fit_rows_theta, fit_rows_phi, curve)
        results[label] = {
            "all_rows": rows,
            "theta": {
                "selected": selected_theta,
                "fit_rows": fit_rows_theta,
                "fit_params": {"b_cos2_intensity": 1.0, "c_sin2_intensity": 1.0, **shared_noise_params},
            },
            "phi": {
                "selected": selected_phi,
                "fit_rows": fit_rows_phi,
                "fit_params": {"b_cos2_intensity": 1.0, "c_sin2_intensity": 1.0, **shared_noise_params},
            },
        }

        safe_label = label.replace(" ", "").replace(".", "p")
        base.write_csv(OUTPUT_DIR / f"{safe_label}_all_rods.csv", base.rows_to_dicts(rows))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_theta_selected_rods.csv", base.rows_to_dicts(selected_theta))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_theta_fit_rods.csv", base.rows_to_dicts(fit_rows_theta))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_theta_pruned_rods.csv", base.rows_to_dicts(theta_pruned))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_phi_selected_rods.csv", base.rows_to_dicts(selected_phi))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_phi_fit_rods.csv", base.rows_to_dicts(fit_rows_phi))
        base.write_csv(OUTPUT_DIR / f"{safe_label}_phi_pruned_rods.csv", base.rows_to_dicts(phi_pruned))

        summaries.append(
            {
                "label": label,
                "source_dir": str(root),
                "n_rods_loaded": len(rows),
                "theta_selection": {
                    "n_selected_best_per_bin": len(selected_theta),
                    "n_used_for_fit": len(fit_rows_theta),
                    "pruned_after_selection": len(theta_pruned),
                    "prune_rule": "remove only non-finite or non-positive sigma_theta",
                },
                "phi_selection": {
                    "n_selected_best_per_bin": len(selected_phi),
                    "n_used_for_fit": len(fit_rows_phi),
                    "pruned_after_selection": len(phi_pruned),
                    "prune_rule": "remove only non-finite or non-positive sigma_phi",
                },
                "selection": {
                    "theta_bin_width_deg": base.THETA_BIN_WIDTH_DEG,
                    "top_n_per_bin": base.TOP_N_PER_BIN,
                    "high_angle_filter_start_deg": base.HIGH_ANGLE_FILTER_START_DEG,
                    "high_angle_top_n_per_bin": base.HIGH_ANGLE_TOP_N_PER_BIN,
                    "very_high_angle_filter_start_deg": base.VERY_HIGH_ANGLE_FILTER_START_DEG,
                    "very_high_angle_top_n_per_bin": base.VERY_HIGH_ANGLE_TOP_N_PER_BIN,
                    "extra_selection_fraction": base.EXTRA_SELECTION_FRACTION,
                    "low_theta_rule": "for theta 0-9.999 deg can include up to 3 best rods if available",
                },
                "sigma_xy_fit": shared_noise_params,
                "theta_curve": base.FOURKAS_WATER,
                "largest_mean_r_loaded": max((r.r_mean for r in rows), default=float("nan")),
            }
        )
        print(
            f"{label}: loaded={len(rows)} "
            f"theta_fit={len(fit_rows_theta)} "
            f"phi_fit={len(fit_rows_phi)} "
            f"theta_pruned={len(theta_pruned)} "
            f"phi_pruned={len(phi_pruned)} "
            f"sigma_xy={shared_noise_params['sigma_xy_const']:.6g}"
        )

    plot_combined_25nm(results, OUTPUT_DIR)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
