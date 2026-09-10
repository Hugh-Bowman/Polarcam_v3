from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:
    least_squares = None


GOOD_DIR = Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400" / "good"
CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
OUTPUT_DIR = (
    Path("stationary rods 25nm 02072026")
    / "filtered_intensity_p98_50_to_400"
    / "plots"
    / "pixel_error_c_over_I2_fit"
)


@dataclass
class RodDatum:
    rod: str
    n_frames: int
    r_mean: float
    sigma_xy: float
    intensity_mean: float
    theta_deg: float = float("nan")
    dtheta_dr_rad: float = float("nan")
    sigma_theta_xy_only_deg: float = float("nan")
    sigma_phi_deg: float = float("nan")
    intensity_model: float = float("nan")


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy_series = payload.get("xy_series")
    if not isinstance(xy_series, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape in {meta_path}: {arr.shape}")
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _load_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {
        "r": r,
        "theta_deg": theta_deg,
        "dtheta_dr_rad": dtheta_dr_rad,
    }


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _estimate_intensity_mean(npy_path: Path) -> float:
    arr = np.load(npy_path, allow_pickle=False)
    return float(np.mean(np.asarray(arr, dtype=np.float64)))


def _load_data(good_dir: Path) -> list[RodDatum]:
    out: list[RodDatum] = []
    for rod_dir in sorted([p for p in good_dir.iterdir() if p.is_dir()]):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        npy_path = rod_dir / "capture_maxfps_15x15.npy"
        if not meta_path.exists() or not npy_path.exists():
            continue
        xy = _load_xy_series(meta_path)
        if xy.size == 0:
            continue
        x = xy[:, 0]
        y = xy[:, 1]
        r = np.sqrt((x * x) + (y * y))
        r_mean = float(np.mean(r))
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
        sigma_xy = float(np.hypot(sigma_x, sigma_y))
        intensity_mean = _estimate_intensity_mean(npy_path)
        if not np.isfinite(intensity_mean) or intensity_mean <= 0.0:
            continue
        out.append(
            RodDatum(
                rod=rod_dir.name,
                n_frames=int(xy.shape[0]),
                r_mean=r_mean,
                sigma_xy=sigma_xy,
                intensity_mean=intensity_mean,
            )
        )
    return out


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], r_max_dataset: float) -> tuple[dict[str, np.ndarray], float]:
    r_max_empirical = float(np.max(curve["r"]))
    scale = float(r_max_dataset / max(r_max_empirical, 1e-12))
    return (
        {
            "r": curve["r"] * scale,
            "theta_deg": curve["theta_deg"].copy(),
            "dtheta_dr_rad": curve["dtheta_dr_rad"] / max(scale, 1e-12),
        },
        scale,
    )


def _apply_curve_to_data(data: list[RodDatum], curve: dict[str, np.ndarray]) -> None:
    for d in data:
        d.theta_deg = _interp(curve["r"], curve["theta_deg"], d.r_mean)
        d.dtheta_dr_rad = _interp(curve["r"], curve["dtheta_dr_rad"], d.r_mean)
        d.sigma_theta_xy_only_deg = float(np.degrees(abs(d.dtheta_dr_rad) * d.sigma_xy))
        d.sigma_phi_deg = float(np.degrees(d.sigma_xy / max(2.0 * d.r_mean, 1e-12)))


def _predict_from_params(data: list[RodDatum], inten_a_sin2: float, inten_b_cos2: float, coeff_c: float) -> dict[str, np.ndarray]:
    theta_deg = np.asarray([d.theta_deg for d in data], dtype=np.float64)
    r_mean = np.asarray([d.r_mean for d in data], dtype=np.float64)
    dtheta_dr_deg = np.asarray([np.degrees(abs(d.dtheta_dr_rad)) for d in data], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    intensity_model = (inten_a_sin2 * (np.sin(theta_rad) ** 2)) + (inten_b_cos2 * (np.cos(theta_rad) ** 2))
    intensity_model = np.maximum(intensity_model, 1e-12)
    var_xy = float(coeff_c) / (intensity_model * intensity_model)
    sigma_xy = np.sqrt(np.maximum(var_xy, 0.0))
    sigma_theta = dtheta_dr_deg * sigma_xy
    sigma_phi = np.degrees(sigma_xy / np.maximum(2.0 * r_mean, 1e-12))
    return {
        "intensity_model": intensity_model,
        "var_xy": var_xy,
        "sigma_xy": sigma_xy,
        "sigma_theta": sigma_theta,
        "sigma_phi": sigma_phi,
    }


def _initial_guess(data: list[RodDatum]) -> tuple[float, float, float]:
    theta_rad = np.radians(np.asarray([d.theta_deg for d in data], dtype=np.float64))
    intensity = np.asarray([d.intensity_mean for d in data], dtype=np.float64)
    X = np.column_stack([np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2])
    beta, *_ = np.linalg.lstsq(X, intensity, rcond=None)
    beta = np.maximum(beta, 1e-6)
    sigma_xy2 = np.asarray([d.sigma_xy * d.sigma_xy for d in data], dtype=np.float64)
    I_guess = np.maximum((beta[0] * (np.sin(theta_rad) ** 2)) + (beta[1] * (np.cos(theta_rad) ** 2)), 1e-12)
    c0 = float(np.median(sigma_xy2 * I_guess * I_guess))
    return float(beta[0]), float(beta[1]), max(c0, 1e-12)


def _fit_model(data: list[RodDatum]) -> dict:
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is required for this fit.")
    sigma_theta_meas = np.asarray([d.sigma_theta_xy_only_deg for d in data], dtype=np.float64)
    sigma_phi_meas = np.asarray([d.sigma_phi_deg for d in data], dtype=np.float64)
    x0 = np.asarray(_initial_guess(data), dtype=np.float64)
    lb = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    ub = np.asarray([np.inf, np.inf, np.inf], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = _predict_from_params(data, float(params[0]), float(params[1]), float(params[2]))
        return np.concatenate(
            [
                pred["sigma_theta"] - sigma_theta_meas,
                pred["sigma_phi"] - sigma_phi_meas,
            ]
        )

    result = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
    pred = _predict_from_params(data, float(result.x[0]), float(result.x[1]), float(result.x[2]))
    resid = residuals(result.x)
    return {
        "intensity_a_sin2": float(result.x[0]),
        "intensity_b_cos2": float(result.x[1]),
        "c_over_I2": float(result.x[2]),
        "rss": float(np.sum(resid * resid)),
        "rmse_joint": float(np.sqrt(np.mean(resid * resid))),
        "rmse_theta_deg": float(np.sqrt(np.mean((pred["sigma_theta"] - sigma_theta_meas) ** 2))),
        "rmse_phi_deg": float(np.sqrt(np.mean((pred["sigma_phi"] - sigma_phi_meas) ** 2))),
        "pred": pred,
        "initial_guess": {
            "intensity_a_sin2": float(x0[0]),
            "intensity_b_cos2": float(x0[1]),
            "c_over_I2": float(x0[2]),
        },
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    good_dir = Path.cwd() / GOOD_DIR
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_curve(Path.cwd() / CURVE_CSV)
    data = _load_data(good_dir)
    if not data:
        raise SystemExit("No usable rods found in the filtered 25nm good folder.")

    r_max_dataset = float(max(d.r_mean for d in data))
    stretched_curve, stretch_scale = _stretch_curve_for_dataset(curve, r_max_dataset)
    _apply_curve_to_data(data, stretched_curve)
    fit = _fit_model(data)
    pred = fit["pred"]

    fitted_rows = []
    for datum, intensity_model, var_xy_pred, sigma_xy_pred, sigma_theta_pred_deg, sigma_phi_pred_deg in zip(
        data,
        pred["intensity_model"],
        pred["var_xy"],
        pred["sigma_xy"],
        pred["sigma_theta"],
        pred["sigma_phi"],
    ):
        datum.intensity_model = float(intensity_model)
        fitted_rows.append(
            {
                "rod": datum.rod,
                "theta_deg": datum.theta_deg,
                "r_mean": datum.r_mean,
                "intensity_mean": datum.intensity_mean,
                "intensity_model": float(intensity_model),
                "sigma_xy_measured": datum.sigma_xy,
                "sigma_xy_pred": float(sigma_xy_pred),
                "sigma_phi_measured_deg": datum.sigma_phi_deg,
                "sigma_phi_pred_deg": float(sigma_phi_pred_deg),
                "sigma_theta_measured_deg": datum.sigma_theta_xy_only_deg,
                "sigma_theta_pred_deg": float(sigma_theta_pred_deg),
                "sigma_xy2_pred": float(var_xy_pred),
            }
        )

    _write_csv(
        out_dir / "filtered_25nm_c_over_I2_fitted_points.csv",
        fitted_rows,
        [
            "rod",
            "theta_deg",
            "r_mean",
            "intensity_mean",
            "intensity_model",
            "sigma_xy_measured",
            "sigma_xy_pred",
            "sigma_phi_measured_deg",
            "sigma_phi_pred_deg",
            "sigma_theta_measured_deg",
            "sigma_theta_pred_deg",
            "sigma_xy2_pred",
        ],
    )

    th = np.asarray([r["theta_deg"] for r in fitted_rows], dtype=np.float64)
    sth_meas = np.asarray([r["sigma_theta_measured_deg"] for r in fitted_rows], dtype=np.float64)
    sth_pred = np.asarray([r["sigma_theta_pred_deg"] for r in fitted_rows], dtype=np.float64)
    sphi_meas = np.asarray([r["sigma_phi_measured_deg"] for r in fitted_rows], dtype=np.float64)
    sphi_pred = np.asarray([r["sigma_phi_pred_deg"] for r in fitted_rows], dtype=np.float64)
    intensity_meas = np.asarray([r["intensity_mean"] for r in fitted_rows], dtype=np.float64)
    intensity_model_pts = np.asarray([r["intensity_model"] for r in fitted_rows], dtype=np.float64)

    theta_grid = np.linspace(float(np.min(stretched_curve["theta_deg"])), float(np.max(stretched_curve["theta_deg"])), 500)
    theta_curve = stretched_curve["theta_deg"]
    r_curve = stretched_curve["r"]
    dtheta_dr_deg_curve = np.degrees(np.abs(stretched_curve["dtheta_dr_rad"]))
    r_grid = np.interp(theta_grid, theta_curve, r_curve, left=float(r_curve[0]), right=float(r_curve[-1]))
    dtheta_dr_deg_grid = np.interp(theta_grid, theta_curve, dtheta_dr_deg_curve, left=float(dtheta_dr_deg_curve[0]), right=float(dtheta_dr_deg_curve[-1]))

    inten_a = float(fit["intensity_a_sin2"])
    inten_b = float(fit["intensity_b_cos2"])
    coeff_c = float(fit["c_over_I2"])
    theta_grid_rad = np.radians(theta_grid)
    I_grid = (inten_a * (np.sin(theta_grid_rad) ** 2)) + (inten_b * (np.cos(theta_grid_rad) ** 2))
    I_grid = np.maximum(I_grid, 1e-12)
    sigma_xy_grid = np.sqrt(coeff_c / (I_grid * I_grid))
    sigma_theta_grid = dtheta_dr_deg_grid * sigma_xy_grid
    sigma_phi_grid = np.degrees(sigma_xy_grid / np.maximum(2.0 * r_grid, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, intensity_meas, s=40, alpha=0.85, color="#9467bd", edgecolors="none", label="Measured rod intensity")
    ax.plot(theta_grid, I_grid, color="#111111", lw=2.0, label=r"Fit: $a\sin^2\theta + b\cos^2\theta$")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Mean pixel intensity I")
    ax.set_title("25nm rods filtered: intensity vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_vs_theta.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sth_meas, s=40, alpha=0.85, color="#1f77b4", edgecolors="none", label="Measured propagated points")
    ax.plot(theta_grid, sigma_theta_grid, color="#d62728", lw=2.0, label=r"Fit: Var$=c/I(\theta)^2$")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("25nm rods filtered: theta std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sphi_meas, s=40, alpha=0.85, color="#2ca02c", edgecolors="none", label="Measured propagated points")
    ax.plot(theta_grid, sigma_phi_grid, color="#ff7f0e", lw=2.0, label=r"Fit: Var$=c/I(\theta)^2$")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("25nm rods filtered: phi std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    theta_lo = float(min(np.min(sth_meas), np.min(sth_pred)))
    theta_hi = float(max(np.max(sth_meas), np.max(sth_pred)))
    ax.scatter(sth_meas, sth_pred, s=40, alpha=0.82, color="#d62728", edgecolors="none")
    ax.plot([theta_lo, theta_hi], [theta_lo, theta_hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Actual propagated std dev theta (deg)")
    ax.set_ylabel("Predicted std dev theta (deg)")
    ax.set_title("25nm rods filtered: predicted vs actual theta std")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    phi_lo = float(min(np.min(sphi_meas), np.min(sphi_pred)))
    phi_hi = float(max(np.max(sphi_meas), np.max(sphi_pred)))
    ax.scatter(sphi_meas, sphi_pred, s=40, alpha=0.82, color="#ff7f0e", edgecolors="none")
    ax.plot([phi_lo, phi_hi], [phi_lo, phi_hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Actual propagated std dev phi (deg)")
    ax.set_ylabel("Predicted std dev phi (deg)")
    ax.set_title("25nm rods filtered: predicted vs actual phi std")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    summary = {
        "good_dir": str(good_dir),
        "curve_csv": str(Path.cwd() / CURVE_CSV),
        "output_dir": str(out_dir),
        "n_rods": int(len(data)),
        "curve_r_stretch": {
            "empirical_r_max": float(np.max(curve["r"])),
            "dataset_r_max": r_max_dataset,
            "stretch_scale": float(stretch_scale),
        },
        "fit_target": "joint sigma_theta(theta) and sigma_phi(theta), XY propagation only",
        "fit_model": "I(theta) = a sin^2(theta) + b cos^2(theta), sigma_xy(theta)^2 = c / I(theta)^2",
        "ignores_theta_r_curve_uncertainty": True,
        "params": {
            "a_sin2": inten_a,
            "b_cos2": inten_b,
            "c_over_I2": coeff_c,
        },
        "initial_guess": fit["initial_guess"],
        "rmse_joint": float(fit["rmse_joint"]),
        "rmse_theta_deg": float(fit["rmse_theta_deg"]),
        "rmse_phi_deg": float(fit["rmse_phi_deg"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
