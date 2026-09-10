from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = {
    "40x65nm": {
        "summary": Path("stationary rod data 01072026") / "plots" / "pixel_error_model_fit" / "summary.json",
        "points": Path("stationary rod data 01072026") / "plots" / "pixel_error_model_fit" / "pixel_error_model_fitted_points.csv",
        "curve": Path("theta_r_curves_for_analysis") / "theta_r_curve_40x65nm_recording_bootstrap.csv",
        "color_points_theta": "#1f77b4",
        "color_curve_theta": "#0b4f8a",
        "color_points_phi": "#2ca02c",
        "color_curve_phi": "#1b6f1b",
        "kind": "40nm",
    },
    "25x65nm": {
        "summary": Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400_sigma_theta_pruned" / "plots" / "pixel_error_c_over_I2_fit_r_gt_0p25" / "summary.json",
        "points": Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400_sigma_theta_pruned" / "plots" / "pixel_error_c_over_I2_fit_r_gt_0p25" / "pruned_filtered_25nm_r_gt_0p25_fitted_points.csv",
        "curve": Path("theta_r_curves_for_analysis") / "theta_r_curve_25x65nm_recording_bootstrap.csv",
        "color_points_theta": "#d62728",
        "color_curve_theta": "#8c1d1d",
        "color_points_phi": "#ff7f0e",
        "color_curve_phi": "#b35900",
        "kind": "25nm",
    },
}

OUTPUT_DIR = Path("outputs") / "40nm_vs_25nm_std_comparison_remade_curves"
STD_CURVE_MAX_DEG = 10.0


def _load_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {"r": r, "theta_deg": theta_deg, "dtheta_dr_rad": dtheta_dr_rad}


def _load_points(points_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(points_csv.open("r", encoding="utf-8", newline="")))
    return {
        "theta_deg": np.asarray([float(r["theta_deg"]) for r in rows], dtype=np.float64),
        "sigma_theta_measured_deg": np.asarray([float(r["sigma_theta_measured_deg"]) for r in rows], dtype=np.float64),
        "sigma_phi_measured_deg": np.asarray([float(r["sigma_phi_measured_deg"]) for r in rows], dtype=np.float64),
    }


def _predict_curve(summary: dict, curve: dict[str, np.ndarray], theta_grid: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "40nm":
        best = summary["best_model_by_bic"]
        a_sin2 = float(best["intensity_a_sin2"])
        b_cos2 = float(best["intensity_b_cos2"])
        c_over_i2 = float(best["c"])
    else:
        a_sin2 = float(summary["params"]["a_sin2"])
        b_cos2 = float(summary["params"]["b_cos2"])
        c_over_i2 = float(summary["params"]["c_over_I2"])

    theta_curve = np.asarray(curve["theta_deg"], dtype=np.float64)
    r_curve = np.asarray(curve["r"], dtype=np.float64)
    dtheta_dr_deg_curve = np.degrees(np.abs(np.asarray(curve["dtheta_dr_rad"], dtype=np.float64)))
    r_grid = np.interp(theta_grid, theta_curve, r_curve, left=float(r_curve[0]), right=float(r_curve[-1]))
    dtheta_dr_deg_grid = np.interp(
        theta_grid,
        theta_curve,
        dtheta_dr_deg_curve,
        left=float(dtheta_dr_deg_curve[0]),
        right=float(dtheta_dr_deg_curve[-1]),
    )
    tg = np.radians(theta_grid)
    I_grid = a_sin2 * (np.sin(tg) ** 2) + b_cos2 * (np.cos(tg) ** 2)
    I_grid = np.maximum(I_grid, 1e-12)
    sigma_xy_grid = np.sqrt(c_over_i2 / (I_grid * I_grid))
    sigma_theta_grid = dtheta_dr_deg_grid * sigma_xy_grid
    sigma_phi_grid = np.degrees(sigma_xy_grid / np.maximum(2.0 * r_grid, 1e-12))
    return sigma_theta_grid, sigma_phi_grid


def _clip_curve(theta_grid: np.ndarray, y_grid: np.ndarray, max_std: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(y_grid <= float(max_std), dtype=bool)
    return theta_grid[valid], y_grid[valid]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {}
    theta_min = 180.0
    theta_max = -180.0
    for label, cfg in DATASETS.items():
        summary = json.loads((Path.cwd() / cfg["summary"]).read_text(encoding="utf-8"))
        points = _load_points(Path.cwd() / cfg["points"])
        curve = _load_curve(Path.cwd() / cfg["curve"])
        theta_min = min(theta_min, float(np.min(points["theta_deg"])), float(np.min(curve["theta_deg"])))
        theta_max = max(theta_max, float(np.max(points["theta_deg"])), float(np.max(curve["theta_deg"])))
        payloads[label] = {
            "summary_data": summary,
            "points_data": points,
            "curve_data": curve,
            **cfg,
        }

    theta_grid = np.linspace(theta_min, theta_max, 800)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, p in payloads.items():
        ax.scatter(
            p["points_data"]["theta_deg"],
            p["points_data"]["sigma_theta_measured_deg"],
            s=28,
            alpha=0.78,
            color=p["color_points_theta"],
            edgecolors="none",
            label=f"{label} points",
        )
        sigma_theta_grid, _sigma_phi_grid = _predict_curve(p["summary_data"], p["curve_data"], theta_grid, p["kind"])
        tx, ty = _clip_curve(theta_grid, sigma_theta_grid, STD_CURVE_MAX_DEG)
        ax.plot(tx, ty, lw=2.0, color=p["color_curve_theta"], label=f"{label} theory")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("Theta std vs theta: 40x65nm and 25x65nm with remade theta(r) curves")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "theta_std_40nm_vs_25nm_remade_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, p in payloads.items():
        ax.scatter(
            p["points_data"]["theta_deg"],
            p["points_data"]["sigma_phi_measured_deg"],
            s=28,
            alpha=0.78,
            color=p["color_points_phi"],
            edgecolors="none",
            label=f"{label} points",
        )
        _sigma_theta_grid, sigma_phi_grid = _predict_curve(p["summary_data"], p["curve_data"], theta_grid, p["kind"])
        tx, ty = _clip_curve(theta_grid, sigma_phi_grid, STD_CURVE_MAX_DEG)
        ax.plot(tx, ty, lw=2.0, color=p["color_curve_phi"], label=f"{label} theory")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("Phi std vs theta: 40x65nm and 25x65nm with remade theta(r) curves")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phi_std_40nm_vs_25nm_remade_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "std_curve_clip_deg": STD_CURVE_MAX_DEG,
        "datasets": {
            label: {
                "summary": str((Path.cwd() / p["summary"]).resolve()),
                "points": str((Path.cwd() / p["points"]).resolve()),
                "curve": str((Path.cwd() / p["curve"]).resolve()),
                "kind": p["kind"],
            }
            for label, p in payloads.items()
        },
        "method": "Reuse the same filtered stationary rod point sets and intensity/noise-model summaries as before, but replace the theta(r) calibration curves with the new remade curves saved in theta_r_curves_for_analysis.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
