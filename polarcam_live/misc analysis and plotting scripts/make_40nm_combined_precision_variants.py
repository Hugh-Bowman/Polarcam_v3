from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_40nm_bg_subtracted_sound_precision_new as base


VARIANTS = [
    {
        "tag": "p16_p84_div2",
        "label": "p16-p84 divided by 2",
        "lo_pct": 16.0,
        "hi_pct": 84.0,
        "divisor": 2.0,
    },
    {
        "tag": "p2p275_p97p725_div4",
        "label": "p2.275-p97.725 divided by 4",
        "lo_pct": 2.2750131948179195,
        "hi_pct": 97.72498680518208,
        "divisor": 4.0,
    },
]


def _safe_label(label: str) -> str:
    return label.replace(" ", "").replace(".", "p")


def _apply_phi_percentile_sigma(rows: list[base.RodDatum], lo_pct: float, hi_pct: float, divisor: float) -> None:
    for row in rows:
        meta_path = row.path / "capture_maxfps_15x15_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xy = base.load_xy(meta, meta_path)
        if xy.shape[0] < 5:
            continue
        phi = 0.5 * np.arctan2(xy[:, 1], xy[:, 0])
        mean_phi = 0.5 * float(np.angle(np.mean(np.exp(2.0j * phi))))
        delta = 0.5 * np.angle(np.exp(2.0j * (phi - mean_phi)))
        lo = float(np.percentile(np.degrees(delta), lo_pct))
        hi = float(np.percentile(np.degrees(delta), hi_pct))
        row.sigma_phi_deg = abs(hi - lo) / divisor


def _fit_results(curve: dict[str, np.ndarray], lo_pct: float, hi_pct: float, divisor: float) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for label, root in base.DATASETS.items():
        rows = base.load_dataset(label, root, curve)
        _apply_phi_percentile_sigma(rows, lo_pct, hi_pct, divisor)
        selected_theta = base.select_best_per_theta_bin(rows, metric="sigma_r_percentile", flag="selected_for_theta")
        selected_phi = base.select_best_per_theta_bin(rows, metric="sigma_phi_deg", flag="selected_for_phi")
        selected_theta, theta_pruned = base.prune_theta_selection(label, selected_theta)

        fit_rows_theta = [
            r
            for r in selected_theta
            if base.THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_theta_deg)
            and not bool(r.r_clipped_to_90)
        ]
        fit_rows_phi = [
            r
            for r in selected_phi
            if base.THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_phi_deg)
        ]
        for row in fit_rows_theta:
            row.used_for_theta_fit = True
        for row in fit_rows_phi:
            row.used_for_phi_fit = True

        b_theta, c_theta = base.fit_intensity(fit_rows_theta)
        a_theta = base.fit_variance(fit_rows_theta, curve, b_theta, c_theta, kind="theta")
        b_phi, c_phi = base.fit_intensity(fit_rows_phi)
        a_phi = base.fit_variance(fit_rows_phi, curve, b_phi, c_phi, kind="phi")

        results[label] = {
            "all_rows": rows,
            "theta": {
                "selected": selected_theta,
                "fit_rows": fit_rows_theta,
                "fit_params": {
                    "a_var_xy_over_I2": a_theta,
                    "b_cos2_intensity": b_theta,
                    "c_sin2_intensity": c_theta,
                },
                "pruned": theta_pruned,
            },
            "phi": {
                "selected": selected_phi,
                "fit_rows": fit_rows_phi,
                "fit_params": {
                    "a_var_xy_over_I2": a_phi,
                    "b_cos2_intensity": b_phi,
                    "c_sin2_intensity": c_phi,
                },
            },
        }
    return results


def _theta_70_80_summary(rows: list[base.RodDatum], fit_rows: list[base.RodDatum]) -> dict:
    all_band = [r for r in rows if 70.0 <= float(r.theta_deg) < 80.0]
    fit_band = [r for r in fit_rows if 70.0 <= float(r.theta_deg) < 80.0]
    nearest = sorted(rows, key=lambda r: min(abs(float(r.theta_deg) - 70.0), abs(float(r.theta_deg) - 80.0)))[:5]
    return {
        "n_all_70_to_80": len(all_band),
        "n_fit_70_to_80": len(fit_band),
        "fit_theta_values_70_to_80_deg": [float(r.theta_deg) for r in sorted(fit_band, key=lambda r: r.theta_deg)],
        "nearest_theta_values_deg": [float(r.theta_deg) for r in nearest],
    }


def _plot_combined_variant(
    results: dict[str, dict],
    curve: dict[str, np.ndarray],
    out_dir: Path,
    variant_tag: str,
    variant_label: str,
) -> None:
    colors = {"0.6 ms": "#1f77b4", "1.2 ms": "#ff7f0e"}
    theta_grid = np.linspace(base.THETA_FIT_MIN_DEG, 89.999, 900)
    plot_defs = [
        ("theta", "sigma_theta_deg", "sigma_theta_deg", "std theta (deg)", f"combined_std_theta_vs_theta_{variant_tag}.png"),
        ("phi", "sigma_phi_deg", "sigma_phi_deg", "std phi (deg)", f"combined_std_phi_vs_theta_{variant_tag}.png"),
    ]
    for kind, attr, pred_key, ylabel, fname in plot_defs:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        all_y = [getattr(r, attr) for res in results.values() for r in res[kind]["fit_rows"]]
        finite_y = np.asarray(all_y, dtype=np.float64)
        finite_y = finite_y[np.isfinite(finite_y)]
        y_cap = float(max(np.max(finite_y) * 1.18, base.MODEL_ERROR_PLOT_MAX_DEG)) if finite_y.size else base.MODEL_ERROR_PLOT_MAX_DEG
        for label, res in results.items():
            rows = res[kind]["fit_rows"]
            params = res[kind]["fit_params"]
            pred = base.predict(
                theta_grid,
                curve,
                params["b_cos2_intensity"],
                params["c_sin2_intensity"],
                params["a_var_xy_over_I2"],
            )
            color = colors.get(label)
            ax.scatter(
                [r.theta_deg for r in rows],
                [getattr(r, attr) for r in rows],
                s=38,
                alpha=0.78,
                edgecolors="none",
                color=color,
                label=f"{label} data",
            )
            ax.plot(
                theta_grid,
                base.model_visible_until_10_deg(theta_grid, np.asarray(pred[pred_key], dtype=np.float64)),
                lw=2.0,
                color=color,
                label=f"{label} fit",
            )
        ax.axhline(base.MODEL_ERROR_PLOT_MAX_DEG, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.set_ylim(0.0, y_cap)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"40nm rods: {ylabel} vs theta ({variant_label})")
        ax.grid(True, alpha=0.24)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)


def main() -> None:
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "theta_curve": base.FOURKAS_WATER,
        "datasets": {label: str(path) for label, path in base.DATASETS.items()},
        "variants": [],
    }
    for variant in VARIANTS:
        base.R_SIGMA_LO_PCT = float(variant["lo_pct"])
        base.R_SIGMA_HI_PCT = float(variant["hi_pct"])
        base.R_SIGMA_DIVISOR = float(variant["divisor"])
        curve = base.build_curve()
        results = _fit_results(curve, float(variant["lo_pct"]), float(variant["hi_pct"]), float(variant["divisor"]))
        _plot_combined_variant(results, curve, base.OUTPUT_DIR, str(variant["tag"]), str(variant["label"]))

        variant_summary = {
            "tag": variant["tag"],
            "sigma_method": {
                "lower_percentile": variant["lo_pct"],
                "upper_percentile": variant["hi_pct"],
                "divisor": variant["divisor"],
            },
            "datasets": {},
        }
        for label, res in results.items():
            theta_rows = res["theta"]["fit_rows"]
            phi_rows = res["phi"]["fit_rows"]
            variant_summary["datasets"][label] = {
                "n_rods_loaded": len(res["all_rows"]),
                "n_theta_fit": len(theta_rows),
                "n_phi_fit": len(phi_rows),
                "theta_70_to_80_deg": _theta_70_80_summary(res["all_rows"], theta_rows),
                "theta_fit_params": res["theta"]["fit_params"],
                "phi_fit_params": res["phi"]["fit_params"],
                "largest_mean_r_loaded": max((r.r_mean for r in res["all_rows"]), default=float("nan")),
            }
        summary["variants"].append(variant_summary)
        print(
            f"{variant['tag']}: "
            + "; ".join(
                f"{label} theta_fit={len(res['theta']['fit_rows'])} phi_fit={len(res['phi']['fit_rows'])} "
                f"70-80fit={variant_summary['datasets'][label]['theta_70_to_80_deg']['n_fit_70_to_80']}"
                for label, res in results.items()
            )
        )

    (base.OUTPUT_DIR / "combined_precision_variant_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Output: {base.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
