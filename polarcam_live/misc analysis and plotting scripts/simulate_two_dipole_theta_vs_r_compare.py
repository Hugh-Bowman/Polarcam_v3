from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")

from dipole_finite_na_cutout import AngularIntegrals, angular_integrals, anisotropy_from_intensities, intensities_from_orientation


EMPIRICAL_BAND_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
OUTPUT_DIR = Path("stationary rod data 01072026") / "plots" / "two_dipole_theta_vs_r_compare"
NA_OUT = 1.3
N_WATER = 1.333
HOLE_NA = 0.39
TRANSVERSE_INTENSITY_RATIO = 2.0 / 3.0
N_THETA_SAMPLES = 1200
FIT_RATIO_GRID = np.linspace(0.0, 3.0, 3001, dtype=np.float64)


def _load_empirical_band(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {
        "r": np.asarray([float(row["r"]) for row in rows], dtype=np.float64),
        "theta_center": np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64),
        "theta_lo": np.asarray([float(row["theta_deg_lo_1sigma"]) for row in rows], dtype=np.float64),
        "theta_hi": np.asarray([float(row["theta_deg_hi_1sigma"]) for row in rows], dtype=np.float64),
        "theta_std": np.asarray([float(row["theta_deg_std"]) for row in rows], dtype=np.float64),
        "theta_fourkas_glycerol": np.asarray([float(row["theta_deg_fourkas"]) for row in rows], dtype=np.float64),
    }


def _single_dipole_r(theta_rad: np.ndarray, coeffs: AngularIntegrals) -> np.ndarray:
    i0, i90, i45, i135 = intensities_from_orientation(theta_rad, np.zeros_like(theta_rad), coeffs)
    _x, _y, r = anisotropy_from_intensities(i0, i90, i45, i135)
    return np.asarray(r, dtype=np.float64)


def _r_to_i0_i90(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip(np.asarray(r, dtype=np.float64), 0.0, 1.0)
    i0 = 0.5 * (1.0 - r)
    i90 = 0.5 * (1.0 + r)
    return i0, i90


def _two_mode_projected_r(theta_rad: np.ndarray, coeffs: AngularIntegrals, transverse_ratio: float) -> np.ndarray:
    r_long = _single_dipole_r(theta_rad, coeffs)
    i0_long, i90_long = _r_to_i0_i90(r_long)

    theta_short = (0.5 * np.pi) - theta_rad
    r_short = _single_dipole_r(theta_short, coeffs)
    i0_short, i90_short = _r_to_i0_i90(r_short)

    w_long = np.sin(theta_rad) ** 2
    w_short = float(transverse_ratio) * (np.cos(theta_rad) ** 2)

    i0 = (w_long * i0_long) + (w_short * i0_short)
    i90 = (w_long * i90_long) + (w_short * i90_short)
    return np.abs(i90 - i0) / np.maximum(i90 + i0, 1e-12)


def _fit_transverse_ratio(
    empirical: dict[str, np.ndarray],
    coeffs: AngularIntegrals,
) -> tuple[float, float, np.ndarray]:
    theta_emp_rad = np.radians(empirical["theta_center"])
    r_emp = empirical["r"]
    best_ratio = 0.0
    best_rss = float("inf")
    best_curve = np.zeros_like(theta_emp_rad, dtype=np.float64)
    for ratio in FIT_RATIO_GRID:
        r_model = _two_mode_projected_r(theta_emp_rad, coeffs, float(ratio))
        rss = float(np.sum((r_model - r_emp) ** 2))
        if rss < best_rss:
            best_ratio = float(ratio)
            best_rss = rss
            best_curve = r_model
    return best_ratio, best_rss, best_curve


def _make_plot(
    empirical: dict[str, np.ndarray],
    theta_deg_grid: np.ndarray,
    r_single: np.ndarray,
    r_two: np.ndarray,
    fitted_ratio: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.fill_betweenx(
        empirical["theta_center"],
        empirical["r"],
        empirical["r"],
        color="#2ca02c",
        alpha=0.0,
    )
    ax.fill_between(
        empirical["r"],
        empirical["theta_lo"],
        empirical["theta_hi"],
        color="#2ca02c",
        alpha=0.22,
        label="Empirical 1 sigma band",
    )
    ax.plot(empirical["r"], empirical["theta_center"], color="#1b7f3a", lw=2.3, label="Empirical center")
    ax.plot(r_single, theta_deg_grid, color="#f58518", lw=2.0, ls="--", label="Fourkas single dipole, water")
    ax.plot(
        r_two,
        theta_deg_grid,
        color="#1f77b4",
        lw=2.4,
        label=f"Two-mode projected fit (transverse={fitted_ratio:.3f})",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) comparison with two-dipole finite-NA model")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    empirical = _load_empirical_band(Path.cwd() / EMPIRICAL_BAND_CSV)
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    coeffs_water = angular_integrals(
        na_out=NA_OUT,
        n_medium=N_WATER,
        hole=HOLE_NA,
        hole_mode="na",
        medium_name="water",
    )

    theta_deg_grid = np.linspace(0.0, 89.95, N_THETA_SAMPLES, dtype=np.float64)
    theta_rad_grid = np.radians(theta_deg_grid)
    r_single = _single_dipole_r(theta_rad_grid, coeffs_water)
    fitted_ratio, fit_rss, r_fit_on_empirical_theta = _fit_transverse_ratio(empirical, coeffs_water)
    r_two = _two_mode_projected_r(theta_rad_grid, coeffs_water, fitted_ratio)

    rows = []
    for theta_deg, rs, rt in zip(theta_deg_grid, r_single, r_two):
        rows.append(
            {
                "theta_deg": float(theta_deg),
                "r_single_dipole_water": float(rs),
                "r_two_mode_fit": float(rt),
            }
        )
    _write_csv(
        out_dir / "two_dipole_theta_vs_r_curves.csv",
        rows,
        ["theta_deg", "r_single_dipole_water", "r_two_mode_fit"],
    )

    fit_rows = []
    for theta_deg, r_emp, r_model in zip(empirical["theta_center"], empirical["r"], r_fit_on_empirical_theta):
        fit_rows.append(
            {
                "theta_deg_empirical": float(theta_deg),
                "r_empirical": float(r_emp),
                "r_model_fit": float(r_model),
                "r_residual": float(r_model - r_emp),
            }
        )
    _write_csv(
        out_dir / "two_dipole_fit_to_empirical.csv",
        fit_rows,
        ["theta_deg_empirical", "r_empirical", "r_model_fit", "r_residual"],
    )

    _make_plot(
        empirical=empirical,
        theta_deg_grid=theta_deg_grid,
        r_single=r_single,
        r_two=r_two,
        fitted_ratio=fitted_ratio,
        output_path=out_dir / "two_dipole_theta_vs_r_compare.png",
    )

    summary = {
        "empirical_band_csv": str(Path.cwd() / EMPIRICAL_BAND_CSV),
        "output_dir": str(out_dir),
        "water_model": {
            "na_out": float(NA_OUT),
            "n_medium": float(N_WATER),
            "hole_na": float(HOLE_NA),
            "J1": float(coeffs_water.J1),
            "J2": float(coeffs_water.J2),
            "J3": float(coeffs_water.J3),
            "r_max_single_dipole": float(coeffs_water.r_max),
        },
        "two_dipole_model": {
            "primary_relative_intensity": 1.0,
            "transverse_relative_intensity_best_fit": float(fitted_ratio),
            "construction": "map each single-mode r to normalized I0/I90, reflect theta about 45deg for short mode, combine weighted I0/I90, reconstruct r from the resulting I0/I90 pair",
            "projection_weighting": {
                "long_mode": "sin^2(theta)",
                "transverse_mode": "c cos^2(theta), with c fit to empirical curve",
            },
            "i0_i90_mapping": {
                "I0": "(1-r)/2",
                "I90": "(1+r)/2",
            },
            "r_max_two_dipole": float(np.max(r_two)),
            "fit_method": "grid search in c over [0, 3] using empirical center curve residuals in r(theta)",
            "fit_grid_size": int(FIT_RATIO_GRID.size),
            "fit_rss": float(fit_rss),
            "fit_rmse_r": float(np.sqrt(np.mean((r_fit_on_empirical_theta - empirical["r"]) ** 2))),
        },
        "empirical_curve": {
            "r_min": float(np.min(empirical["r"])),
            "r_max": float(np.max(empirical["r"])),
            "theta_min_deg": float(np.min(empirical["theta_center"])),
            "theta_max_deg": float(np.max(empirical["theta_center"])),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
